
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
    struct memblock mem_138221;
    struct memblock mem_138222;
    struct memblock mem_138223;
    struct memblock mem_138224;
    struct memblock mem_138225;
    struct memblock mem_138226;
    struct memblock mem_138227;
    struct memblock mem_138228;
    struct memblock mem_138229;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12499(struct futhark_context *ctx, struct memblock *mem_out_p_140868, struct memblock *mem_out_p_140869, struct memblock *mem_out_p_140870, struct memblock w_mem_138230, struct memblock mw_mem_138231, struct memblock vw_mem_138232, struct memblock dw_mem_138233, int64_t n_101673, int64_t m_101674, int64_t step_101679, double lt_r_101680);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12500(struct futhark_context *ctx, struct memblock *mem_out_p_140873, struct memblock *mem_out_p_140874, struct memblock *mem_out_p_140875, struct memblock w_mem_138230, struct memblock mw_mem_138231, struct memblock vw_mem_138232, struct memblock dw_mem_138233, int64_t n_102706, int64_t m_102707, int64_t step_102712, double lt_r_102713);
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_140878, double *out_prim_out_140879, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock tokens_mem_138239, struct memblock target_mem_138240, struct memblock mask_mem_138241);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_140937, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock tokens_mem_138239, struct memblock mask_mem_138240);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_140994, struct memblock *mem_out_p_140995, struct memblock *mem_out_p_140996, struct memblock *mem_out_p_140997, struct memblock *mem_out_p_140998, struct memblock *mem_out_p_140999, struct memblock *mem_out_p_141000, struct memblock *mem_out_p_141001, struct memblock *mem_out_p_141002, struct memblock wte_mem_138230, struct memblock wpe_mem_138231, struct memblock wqry_mem_138232, struct memblock wkey_mem_138233, struct memblock wval_mem_138234, struct memblock wout_mem_138235, struct memblock wup_mem_138236, struct memblock wdown_mem_138237, struct memblock wvoc_mem_138238);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_141003, struct memblock *mem_out_p_141004, struct memblock *mem_out_p_141005, struct memblock *mem_out_p_141006, struct memblock *mem_out_p_141007, struct memblock *mem_out_p_141008, struct memblock *mem_out_p_141009, struct memblock *mem_out_p_141010, struct memblock *mem_out_p_141011, struct memblock *mem_out_p_141012, struct memblock *mem_out_p_141013, struct memblock *mem_out_p_141014, struct memblock *mem_out_p_141015, struct memblock *mem_out_p_141016, struct memblock *mem_out_p_141017, struct memblock *mem_out_p_141018, struct memblock *mem_out_p_141019, struct memblock *mem_out_p_141020, struct memblock *mem_out_p_141021, struct memblock *mem_out_p_141022, struct memblock *mem_out_p_141023, struct memblock *mem_out_p_141024, struct memblock *mem_out_p_141025, struct memblock *mem_out_p_141026, struct memblock *mem_out_p_141027, struct memblock *mem_out_p_141028, struct memblock *mem_out_p_141029, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock wdown_mem_138239, struct memblock wkey_mem_138240, struct memblock wout_mem_138241, struct memblock wpe_mem_138242, struct memblock wqry_mem_138243, struct memblock wte_mem_138244, struct memblock wup_mem_138245, struct memblock wval_mem_138246, struct memblock wvoc_mem_138247, struct memblock wdown_mem_138248, struct memblock wkey_mem_138249, struct memblock wout_mem_138250, struct memblock wpe_mem_138251, struct memblock wqry_mem_138252, struct memblock wte_mem_138253, struct memblock wup_mem_138254, struct memblock wval_mem_138255, struct memblock wvoc_mem_138256, struct memblock masks_mem_138257, struct memblock dls_mem_138258, struct memblock seqs_mem_138259);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_141250, struct memblock *mem_out_p_141251, struct memblock *mem_out_p_141252, struct memblock *mem_out_p_141253, struct memblock *mem_out_p_141254, struct memblock *mem_out_p_141255, struct memblock *mem_out_p_141256, struct memblock *mem_out_p_141257, struct memblock *mem_out_p_141258);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_138221 (ctx->constants->mem_138221)
    #define mem_138222 (ctx->constants->mem_138222)
    #define mem_138223 (ctx->constants->mem_138223)
    #define mem_138224 (ctx->constants->mem_138224)
    #define mem_138225 (ctx->constants->mem_138225)
    #define mem_138226 (ctx->constants->mem_138226)
    #define mem_138227 (ctx->constants->mem_138227)
    #define mem_138228 (ctx->constants->mem_138228)
    #define mem_138229 (ctx->constants->mem_138229)
    mem_138221.references = NULL;
    mem_138222.references = NULL;
    mem_138223.references = NULL;
    mem_138224.references = NULL;
    mem_138225.references = NULL;
    mem_138226.references = NULL;
    mem_138227.references = NULL;
    mem_138228.references = NULL;
    mem_138229.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138221, (int64_t) 3456, "mem_138221")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140850 = 0; nest_i_140850 < (int64_t) 27; nest_i_140850++) {
        for (int64_t nest_i_140851 = 0; nest_i_140851 < (int64_t) 16; nest_i_140851++) {
            ((double *) mem_138221.mem)[nest_i_140850 * (int64_t) 16 + nest_i_140851] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138222, (int64_t) 2048, "mem_138222")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140852 = 0; nest_i_140852 < (int64_t) 16; nest_i_140852++) {
        for (int64_t nest_i_140853 = 0; nest_i_140853 < (int64_t) 16; nest_i_140853++) {
            ((double *) mem_138222.mem)[nest_i_140852 * (int64_t) 16 + nest_i_140853] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138223, (int64_t) 2048, "mem_138223")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140854 = 0; nest_i_140854 < (int64_t) 16; nest_i_140854++) {
        for (int64_t nest_i_140855 = 0; nest_i_140855 < (int64_t) 16; nest_i_140855++) {
            ((double *) mem_138223.mem)[nest_i_140854 * (int64_t) 16 + nest_i_140855] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138224, (int64_t) 2048, "mem_138224")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140856 = 0; nest_i_140856 < (int64_t) 16; nest_i_140856++) {
        for (int64_t nest_i_140857 = 0; nest_i_140857 < (int64_t) 16; nest_i_140857++) {
            ((double *) mem_138224.mem)[nest_i_140856 * (int64_t) 16 + nest_i_140857] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138225, (int64_t) 2048, "mem_138225")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140858 = 0; nest_i_140858 < (int64_t) 16; nest_i_140858++) {
        for (int64_t nest_i_140859 = 0; nest_i_140859 < (int64_t) 16; nest_i_140859++) {
            ((double *) mem_138225.mem)[nest_i_140858 * (int64_t) 16 + nest_i_140859] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138226, (int64_t) 2048, "mem_138226")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140860 = 0; nest_i_140860 < (int64_t) 16; nest_i_140860++) {
        for (int64_t nest_i_140861 = 0; nest_i_140861 < (int64_t) 16; nest_i_140861++) {
            ((double *) mem_138226.mem)[nest_i_140860 * (int64_t) 16 + nest_i_140861] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138227, (int64_t) 8192, "mem_138227")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140862 = 0; nest_i_140862 < (int64_t) 64; nest_i_140862++) {
        for (int64_t nest_i_140863 = 0; nest_i_140863 < (int64_t) 16; nest_i_140863++) {
            ((double *) mem_138227.mem)[nest_i_140862 * (int64_t) 16 + nest_i_140863] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138228, (int64_t) 8192, "mem_138228")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140864 = 0; nest_i_140864 < (int64_t) 16; nest_i_140864++) {
        for (int64_t nest_i_140865 = 0; nest_i_140865 < (int64_t) 64; nest_i_140865++) {
            ((double *) mem_138228.mem)[nest_i_140864 * (int64_t) 64 + nest_i_140865] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138229, (int64_t) 3456, "mem_138229")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_140866 = 0; nest_i_140866 < (int64_t) 27; nest_i_140866++) {
        for (int64_t nest_i_140867 = 0; nest_i_140867 < (int64_t) 16; nest_i_140867++) {
            ((double *) mem_138229.mem)[nest_i_140866 * (int64_t) 16 + nest_i_140867] = 0.0;
        }
    }
    #undef mem_138221
    #undef mem_138222
    #undef mem_138223
    #undef mem_138224
    #undef mem_138225
    #undef mem_138226
    #undef mem_138227
    #undef mem_138228
    #undef mem_138229
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_138221, "ctx->constants->mem_138221") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138222, "ctx->constants->mem_138222") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138223, "ctx->constants->mem_138223") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138224, "ctx->constants->mem_138224") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138225, "ctx->constants->mem_138225") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138226, "ctx->constants->mem_138226") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138227, "ctx->constants->mem_138227") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138228, "ctx->constants->mem_138228") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_138229, "ctx->constants->mem_138229") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12499(struct futhark_context *ctx, struct memblock *mem_out_p_140868, struct memblock *mem_out_p_140869, struct memblock *mem_out_p_140870, struct memblock w_mem_138230, struct memblock mw_mem_138231, struct memblock vw_mem_138232, struct memblock dw_mem_138233, int64_t n_101673, int64_t m_101674, int64_t step_101679, double lt_r_101680)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_138274_cached_sizze_140871 = 0;
    unsigned char *mem_138274 = NULL;
    int64_t mem_138277_cached_sizze_140872 = 0;
    unsigned char *mem_138277 = NULL;
    struct memblock mem_138312;
    
    mem_138312.references = NULL;
    
    struct memblock mem_138239;
    
    mem_138239.references = NULL;
    
    struct memblock mem_138236;
    
    mem_138236.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_138234 = (int64_t) 8 * n_101673;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_138235 = m_101674 * binop_x_138234;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138236, bytes_138235, "mem_138236")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138239, bytes_138235, "mem_138239")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137178 = 0; i_137178 < n_101673; i_137178++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137171 = 0; i_137171 < m_101674; i_137171++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127388 = ((double *) mw_mem_138231.mem)[i_137178 * m_101674 + i_137171];
            
            // futhark/microgpt.fut:472:10-20
            
            double zp_lhs_127389 = 0.85 * zt_rhs_127388;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127390 = ((double *) dw_mem_138233.mem)[i_137178 * m_101674 + i_137171];
            
            // futhark/microgpt.fut:472:35-45
            
            double zp_rhs_127391 = 0.15000000000000002 * zt_rhs_127390;
            
            // futhark/microgpt.fut:472:21-45
            
            double lifted_lambda_res_127392 = zp_lhs_127389 + zp_rhs_127391;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127399 = ((double *) vw_mem_138232.mem)[i_137178 * m_101674 + i_137171];
            
            // futhark/microgpt.fut:474:10-20
            
            double zp_lhs_127400 = 0.99 * zt_rhs_127399;
            
            // futhark/microgpt.fut:474:35-45
            
            double zt_lhs_127402 = 1.0000000000000009e-2 * zt_rhs_127390;
            
            // futhark/microgpt.fut:474:46-56
            
            double zp_rhs_127403 = zt_rhs_127390 * zt_lhs_127402;
            
            // futhark/microgpt.fut:474:21-56
            
            double lifted_lambda_res_127404 = zp_lhs_127400 + zp_rhs_127403;
            
            ((double *) mem_138236.mem)[i_137178 * m_101674 + i_137171] = lifted_lambda_res_127404;
            ((double *) mem_138239.mem)[i_137178 * m_101674 + i_137171] = lifted_lambda_res_127392;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_106692 = sitofp_i64_f64(step_101679);
    
    // futhark/microgpt.fut:476:54-57
    
    double ztzt_rhs_106693 = 1.0 + i64_res_106692;
    
    // futhark/microgpt.fut:476:30-57
    
    double zm_rhs_106694 = fpow64(0.85, ztzt_rhs_106693);
    
    // futhark/microgpt.fut:476:23-57
    
    double zs_rhs_106695 = 1.0 - zm_rhs_106694;
    
    // futhark/microgpt.fut:478:31-58
    
    double zm_rhs_106733 = fpow64(0.99, ztzt_rhs_106693);
    
    // futhark/microgpt.fut:478:23-58
    
    double zs_rhs_106734 = 1.0 - zm_rhs_106733;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_138274_cached_sizze_140871 < bytes_138235) {
        err = lexical_realloc(ctx, &mem_138274, &mem_138274_cached_sizze_140871, bytes_138235);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138277_cached_sizze_140872 < bytes_138235) {
        err = lexical_realloc(ctx, &mem_138277, &mem_138277_cached_sizze_140872, bytes_138235);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137192 = 0; i_137192 < n_101673; i_137192++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137185 = 0; i_137185 < m_101674; i_137185++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_127424 = ((double *) mem_138239.mem)[i_137192 * m_101674 + i_137185];
            
            // futhark/microgpt.fut:476:18-57
            
            double lifted_lambda_res_127425 = zs_lhs_127424 / zs_rhs_106695;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_127432 = ((double *) mem_138236.mem)[i_137192 * m_101674 + i_137185];
            
            // futhark/microgpt.fut:478:18-58
            
            double lifted_lambda_res_127433 = zs_lhs_127432 / zs_rhs_106734;
            
            ((double *) mem_138274)[i_137192 * m_101674 + i_137185] = lifted_lambda_res_127433;
            ((double *) mem_138277)[i_137192 * m_101674 + i_137185] = lifted_lambda_res_127425;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138312, bytes_138235, "mem_138312")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137201 = 0; i_137201 < n_101673; i_137201++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137197 = 0; i_137197 < m_101674; i_137197++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_106015 = ((double *) w_mem_138230.mem)[i_137201 * m_101674 + i_137197];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_106016 = ((double *) mem_138277)[i_137201 * m_101674 + i_137197];
            
            // futhark/microgpt.fut:480:21-34
            
            double zs_lhs_106017 = lt_r_101680 * zt_rhs_106016;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_106018 = ((double *) mem_138274)[i_137201 * m_101674 + i_137197];
            
            // futhark/microgpt.fut:480:51-57
            
            double zp_lhs_106019 = fpow64(ztzt_lhs_106018, 0.5);
            
            // futhark/microgpt.fut:480:59-71
            
            double zs_rhs_106020 = 1.0e-8 + zp_lhs_106019;
            
            // futhark/microgpt.fut:480:35-71
            
            double zm_rhs_106021 = zs_lhs_106017 / zs_rhs_106020;
            
            // futhark/microgpt.fut:480:13-71
            
            double lifted_lambda_res_106022 = zm_lhs_106015 - zm_rhs_106021;
            
            ((double *) mem_138312.mem)[i_137201 * m_101674 + i_137197] = lifted_lambda_res_106022;
        }
    }
    if (memblock_set(ctx, &mem_out_140478, &mem_138312, "mem_138312") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140479, &mem_138239, "mem_138239") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140480, &mem_138236, "mem_138236") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140868, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140869, &mem_out_140479, "mem_out_140479") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140870, &mem_out_140480, "mem_out_140480") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_138274);
        free(mem_138277);
        if (memblock_unref(ctx, &mem_138312, "mem_138312") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_138239, "mem_138239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_138236, "mem_138236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140480, "mem_out_140480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140479, "mem_out_140479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12500(struct futhark_context *ctx, struct memblock *mem_out_p_140873, struct memblock *mem_out_p_140874, struct memblock *mem_out_p_140875, struct memblock w_mem_138230, struct memblock mw_mem_138231, struct memblock vw_mem_138232, struct memblock dw_mem_138233, int64_t n_102706, int64_t m_102707, int64_t step_102712, double lt_r_102713)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_138274_cached_sizze_140876 = 0;
    unsigned char *mem_138274 = NULL;
    int64_t mem_138277_cached_sizze_140877 = 0;
    unsigned char *mem_138277 = NULL;
    struct memblock mem_138312;
    
    mem_138312.references = NULL;
    
    struct memblock mem_138239;
    
    mem_138239.references = NULL;
    
    struct memblock mem_138236;
    
    mem_138236.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_138234 = (int64_t) 8 * n_102706;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_138235 = m_102707 * binop_x_138234;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138236, bytes_138235, "mem_138236")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138239, bytes_138235, "mem_138239")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137178 = 0; i_137178 < n_102706; i_137178++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137171 = 0; i_137171 < m_102707; i_137171++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127388 = ((double *) mw_mem_138231.mem)[i_137178 * m_102707 + i_137171];
            
            // futhark/microgpt.fut:472:10-20
            
            double zp_lhs_127389 = 0.85 * zt_rhs_127388;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127390 = ((double *) dw_mem_138233.mem)[i_137178 * m_102707 + i_137171];
            
            // futhark/microgpt.fut:472:35-45
            
            double zp_rhs_127391 = 0.15000000000000002 * zt_rhs_127390;
            
            // futhark/microgpt.fut:472:21-45
            
            double lifted_lambda_res_127392 = zp_lhs_127389 + zp_rhs_127391;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_127399 = ((double *) vw_mem_138232.mem)[i_137178 * m_102707 + i_137171];
            
            // futhark/microgpt.fut:474:10-20
            
            double zp_lhs_127400 = 0.99 * zt_rhs_127399;
            
            // futhark/microgpt.fut:474:35-45
            
            double zt_lhs_127402 = 1.0000000000000009e-2 * zt_rhs_127390;
            
            // futhark/microgpt.fut:474:46-56
            
            double zp_rhs_127403 = zt_rhs_127390 * zt_lhs_127402;
            
            // futhark/microgpt.fut:474:21-56
            
            double lifted_lambda_res_127404 = zp_lhs_127400 + zp_rhs_127403;
            
            ((double *) mem_138236.mem)[i_137178 * m_102707 + i_137171] = lifted_lambda_res_127404;
            ((double *) mem_138239.mem)[i_137178 * m_102707 + i_137171] = lifted_lambda_res_127392;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_106692 = sitofp_i64_f64(step_102712);
    
    // futhark/microgpt.fut:476:54-57
    
    double ztzt_rhs_106693 = 1.0 + i64_res_106692;
    
    // futhark/microgpt.fut:476:30-57
    
    double zm_rhs_106694 = fpow64(0.85, ztzt_rhs_106693);
    
    // futhark/microgpt.fut:476:23-57
    
    double zs_rhs_106695 = 1.0 - zm_rhs_106694;
    
    // futhark/microgpt.fut:478:31-58
    
    double zm_rhs_106733 = fpow64(0.99, ztzt_rhs_106693);
    
    // futhark/microgpt.fut:478:23-58
    
    double zs_rhs_106734 = 1.0 - zm_rhs_106733;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_138274_cached_sizze_140876 < bytes_138235) {
        err = lexical_realloc(ctx, &mem_138274, &mem_138274_cached_sizze_140876, bytes_138235);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138277_cached_sizze_140877 < bytes_138235) {
        err = lexical_realloc(ctx, &mem_138277, &mem_138277_cached_sizze_140877, bytes_138235);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137192 = 0; i_137192 < n_102706; i_137192++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137185 = 0; i_137185 < m_102707; i_137185++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_127424 = ((double *) mem_138239.mem)[i_137192 * m_102707 + i_137185];
            
            // futhark/microgpt.fut:476:18-57
            
            double lifted_lambda_res_127425 = zs_lhs_127424 / zs_rhs_106695;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_127432 = ((double *) mem_138236.mem)[i_137192 * m_102707 + i_137185];
            
            // futhark/microgpt.fut:478:18-58
            
            double lifted_lambda_res_127433 = zs_lhs_127432 / zs_rhs_106734;
            
            ((double *) mem_138274)[i_137192 * m_102707 + i_137185] = lifted_lambda_res_127433;
            ((double *) mem_138277)[i_137192 * m_102707 + i_137185] = lifted_lambda_res_127425;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138312, bytes_138235, "mem_138312")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137201 = 0; i_137201 < n_102706; i_137201++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137197 = 0; i_137197 < m_102707; i_137197++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_106015 = ((double *) w_mem_138230.mem)[i_137201 * m_102707 + i_137197];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_106016 = ((double *) mem_138277)[i_137201 * m_102707 + i_137197];
            
            // futhark/microgpt.fut:480:21-34
            
            double zs_lhs_106017 = lt_r_102713 * zt_rhs_106016;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_106018 = ((double *) mem_138274)[i_137201 * m_102707 + i_137197];
            
            // futhark/microgpt.fut:480:51-57
            
            double zp_lhs_106019 = fpow64(ztzt_lhs_106018, 0.5);
            
            // futhark/microgpt.fut:480:59-71
            
            double zs_rhs_106020 = 1.0e-8 + zp_lhs_106019;
            
            // futhark/microgpt.fut:480:35-71
            
            double zm_rhs_106021 = zs_lhs_106017 / zs_rhs_106020;
            
            // futhark/microgpt.fut:480:13-71
            
            double lifted_lambda_res_106022 = zm_lhs_106015 - zm_rhs_106021;
            
            ((double *) mem_138312.mem)[i_137201 * m_102707 + i_137197] = lifted_lambda_res_106022;
        }
    }
    if (memblock_set(ctx, &mem_out_140478, &mem_138312, "mem_138312") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140479, &mem_138239, "mem_138239") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140480, &mem_138236, "mem_138236") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140873, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140874, &mem_out_140479, "mem_out_140479") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140875, &mem_out_140480, "mem_out_140480") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_138274);
        free(mem_138277);
        if (memblock_unref(ctx, &mem_138312, "mem_138312") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_138239, "mem_138239") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_138236, "mem_138236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140480, "mem_out_140480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140479, "mem_out_140479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_cal_loss(struct futhark_context *ctx, struct memblock *mem_out_p_140878, double *out_prim_out_140879, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock tokens_mem_138239, struct memblock target_mem_138240, struct memblock mask_mem_138241)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_138242_cached_sizze_140880 = 0;
    unsigned char *mem_138242 = NULL;
    int64_t mem_138247_cached_sizze_140881 = 0;
    unsigned char *mem_138247 = NULL;
    int64_t mem_138258_cached_sizze_140882 = 0;
    unsigned char *mem_138258 = NULL;
    int64_t mem_138263_cached_sizze_140883 = 0;
    unsigned char *mem_138263 = NULL;
    int64_t mem_138270_cached_sizze_140884 = 0;
    unsigned char *mem_138270 = NULL;
    int64_t mem_138281_cached_sizze_140885 = 0;
    unsigned char *mem_138281 = NULL;
    int64_t mem_138286_cached_sizze_140886 = 0;
    unsigned char *mem_138286 = NULL;
    int64_t mem_138293_cached_sizze_140887 = 0;
    unsigned char *mem_138293 = NULL;
    int64_t mem_138304_cached_sizze_140888 = 0;
    unsigned char *mem_138304 = NULL;
    int64_t mem_138305_cached_sizze_140889 = 0;
    unsigned char *mem_138305 = NULL;
    int64_t mem_138306_cached_sizze_140890 = 0;
    unsigned char *mem_138306 = NULL;
    int64_t mem_138319_cached_sizze_140891 = 0;
    unsigned char *mem_138319 = NULL;
    int64_t mem_138320_cached_sizze_140892 = 0;
    unsigned char *mem_138320 = NULL;
    int64_t mem_138321_cached_sizze_140893 = 0;
    unsigned char *mem_138321 = NULL;
    int64_t mem_138352_cached_sizze_140894 = 0;
    unsigned char *mem_138352 = NULL;
    int64_t mem_138353_cached_sizze_140895 = 0;
    unsigned char *mem_138353 = NULL;
    int64_t mem_138354_cached_sizze_140896 = 0;
    unsigned char *mem_138354 = NULL;
    int64_t mem_138370_cached_sizze_140897 = 0;
    unsigned char *mem_138370 = NULL;
    int64_t mem_138371_cached_sizze_140898 = 0;
    unsigned char *mem_138371 = NULL;
    int64_t mem_138372_cached_sizze_140899 = 0;
    unsigned char *mem_138372 = NULL;
    int64_t mem_138385_cached_sizze_140900 = 0;
    unsigned char *mem_138385 = NULL;
    int64_t mem_138386_cached_sizze_140901 = 0;
    unsigned char *mem_138386 = NULL;
    int64_t mem_138387_cached_sizze_140902 = 0;
    unsigned char *mem_138387 = NULL;
    int64_t mem_138433_cached_sizze_140903 = 0;
    unsigned char *mem_138433 = NULL;
    int64_t mem_138439_cached_sizze_140904 = 0;
    unsigned char *mem_138439 = NULL;
    int64_t mem_138444_cached_sizze_140905 = 0;
    unsigned char *mem_138444 = NULL;
    int64_t mem_138455_cached_sizze_140906 = 0;
    unsigned char *mem_138455 = NULL;
    int64_t mem_138460_cached_sizze_140907 = 0;
    unsigned char *mem_138460 = NULL;
    int64_t mem_138471_cached_sizze_140908 = 0;
    unsigned char *mem_138471 = NULL;
    int64_t mem_138476_cached_sizze_140909 = 0;
    unsigned char *mem_138476 = NULL;
    int64_t mem_138483_cached_sizze_140910 = 0;
    unsigned char *mem_138483 = NULL;
    int64_t mem_138490_cached_sizze_140911 = 0;
    unsigned char *mem_138490 = NULL;
    int64_t mem_138501_cached_sizze_140912 = 0;
    unsigned char *mem_138501 = NULL;
    int64_t mem_138506_cached_sizze_140913 = 0;
    unsigned char *mem_138506 = NULL;
    int64_t mem_138517_cached_sizze_140914 = 0;
    unsigned char *mem_138517 = NULL;
    int64_t mem_138522_cached_sizze_140915 = 0;
    unsigned char *mem_138522 = NULL;
    int64_t mem_138538_cached_sizze_140916 = 0;
    unsigned char *mem_138538 = NULL;
    int64_t mem_138543_cached_sizze_140917 = 0;
    unsigned char *mem_138543 = NULL;
    int64_t mem_138554_cached_sizze_140918 = 0;
    unsigned char *mem_138554 = NULL;
    int64_t mem_138559_cached_sizze_140919 = 0;
    unsigned char *mem_138559 = NULL;
    int64_t mem_138570_cached_sizze_140920 = 0;
    unsigned char *mem_138570 = NULL;
    int64_t mem_138575_cached_sizze_140921 = 0;
    unsigned char *mem_138575 = NULL;
    int64_t mem_138586_cached_sizze_140922 = 0;
    unsigned char *mem_138586 = NULL;
    int64_t mem_138591_cached_sizze_140923 = 0;
    unsigned char *mem_138591 = NULL;
    int64_t mem_138598_cached_sizze_140924 = 0;
    unsigned char *mem_138598 = NULL;
    int64_t mem_138609_cached_sizze_140925 = 0;
    unsigned char *mem_138609 = NULL;
    int64_t mem_138614_cached_sizze_140926 = 0;
    unsigned char *mem_138614 = NULL;
    int64_t mem_138625_cached_sizze_140927 = 0;
    unsigned char *mem_138625 = NULL;
    int64_t mem_138630_cached_sizze_140928 = 0;
    unsigned char *mem_138630 = NULL;
    int64_t mem_138641_cached_sizze_140929 = 0;
    unsigned char *mem_138641 = NULL;
    int64_t mem_138646_cached_sizze_140930 = 0;
    unsigned char *mem_138646 = NULL;
    int64_t mem_138657_cached_sizze_140931 = 0;
    unsigned char *mem_138657 = NULL;
    int64_t mem_138662_cached_sizze_140932 = 0;
    unsigned char *mem_138662 = NULL;
    int64_t mem_138673_cached_sizze_140933 = 0;
    unsigned char *mem_138673 = NULL;
    int64_t mem_138678_cached_sizze_140934 = 0;
    unsigned char *mem_138678 = NULL;
    int64_t mem_138693_cached_sizze_140935 = 0;
    unsigned char *mem_138693 = NULL;
    int64_t mem_138700_cached_sizze_140936 = 0;
    unsigned char *mem_138700 = NULL;
    struct memblock mem_138689;
    
    mem_138689.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    double prim_out_140479;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_138242_cached_sizze_140880 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138242, &mem_138242_cached_sizze_140880, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138247_cached_sizze_140881 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138247, &mem_138247_cached_sizze_140881, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137173 = 0; i_137173 < (int64_t) 16; i_137173++) {
        // futhark/microgpt.fut:462:41-50
        
        int64_t tmp_119785 = ((int64_t *) tokens_mem_138239.mem)[i_137173];
        
        // futhark/microgpt.fut:462:37-51
        
        bool x_119786 = sle64((int64_t) 0, tmp_119785);
        
        // futhark/microgpt.fut:462:37-51
        
        bool y_119787 = slt64(tmp_119785, (int64_t) 27);
        
        // futhark/microgpt.fut:462:37-51
        
        bool bounds_check_119788 = x_119786 && y_119787;
        
        // futhark/microgpt.fut:462:37-51
        
        bool index_certs_119789;
        
        if (!bounds_check_119788) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_119785, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:462:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:462:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137169 = 0; i_137169 < (int64_t) 16; i_137169++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_119796 = ((double *) wte_mem_138235.mem)[tmp_119785 * (int64_t) 16 + i_137169];
            
            ((double *) mem_138247)[i_137169] = lifted_lambda_res_119796;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138242, i_137173 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138258_cached_sizze_140882 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138258, &mem_138258_cached_sizze_140882, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138263_cached_sizze_140883 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138263, &mem_138263_cached_sizze_140883, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138270_cached_sizze_140884 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138270, &mem_138270_cached_sizze_140884, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137185 = 0; i_137185 < (int64_t) 16; i_137185++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_119822;
        double r_119824 = 0.0;
        
        for (int64_t i_119823 = 0; i_119823 < (int64_t) 16; i_119823++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_119825 = ((double *) wpe_mem_138233.mem)[i_137185 * (int64_t) 16 + i_119823];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_119826 = ((double *) mem_138242)[i_137185 * (int64_t) 16 + i_119823];
            
            // futhark/microgpt.fut:203:76-116
            
            double zp_res_119827 = zp_lhs_119825 + zp_rhs_119826;
            
            // futhark/microgpt.fut:203:94-163
            
            double zt_res_119828 = zp_res_119827 * zp_res_119827;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_119829 = r_119824 + zt_res_119828;
            double r_tmp_140483 = zp_res_119829;
            
            r_119824 = r_tmp_140483;
        }
        defunc_0_lifted_lambda_res_119822 = r_119824;
        // futhark/microgpt.fut:203:54-182
        
        double zs_res_119830 = defunc_0_lifted_lambda_res_119822 / 16.0;
        
        // futhark/microgpt.fut:204:24-55
        
        double zp_res_119831 = 1.0e-5 + zs_res_119830;
        
        // futhark/microgpt.fut:204:16-55
        
        double sqrt_res_119832 = futrts_sqrt64(zp_res_119831);
        
        // futhark/microgpt.fut:205:85-96
        
        double zs_res_119833 = 1.0 / sqrt_res_119832;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137177 = 0; i_137177 < (int64_t) 16; i_137177++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_119840 = ((double *) wpe_mem_138233.mem)[i_137185 * (int64_t) 16 + i_137177];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_119841 = ((double *) mem_138242)[i_137185 * (int64_t) 16 + i_137177];
            
            // futhark/microgpt.fut:205:38-78
            
            double zp_res_119842 = zp_lhs_119840 + zp_rhs_119841;
            
            // futhark/microgpt.fut:205:56-96
            
            double zt_res_119843 = zs_res_119833 * zp_res_119842;
            
            ((double *) mem_138263)[i_137177] = zt_res_119843;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137181 = 0; i_137181 < (int64_t) 16; i_137181++) {
            // futhark/microgpt.fut:206:4-14
            
            double lifted_lambda_res_119851 = ((double *) mem_138263)[i_137181];
            
            ((double *) mem_138270)[i_137181] = lifted_lambda_res_119851;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138258, i_137185 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138270, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138281_cached_sizze_140885 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138281, &mem_138281_cached_sizze_140885, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138286_cached_sizze_140886 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138286, &mem_138286_cached_sizze_140886, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138293_cached_sizze_140887 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138293, &mem_138293_cached_sizze_140887, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137197 = 0; i_137197 < (int64_t) 16; i_137197++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_119860;
        double r_119862 = 0.0;
        
        for (int64_t i_119861 = 0; i_119861 < (int64_t) 16; i_119861++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_119863 = ((double *) mem_138258)[i_137197 * (int64_t) 16 + i_119861];
            
            // futhark/microgpt.fut:207:78-115
            
            double zt_res_119864 = zt_lhs_119863 * zt_lhs_119863;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_119865 = r_119862 + zt_res_119864;
            double r_tmp_140487 = zp_res_119865;
            
            r_119862 = r_tmp_140487;
        }
        defunc_0_lifted_lambda_res_119860 = r_119862;
        // futhark/microgpt.fut:207:57-133
        
        double zs_res_119866 = defunc_0_lifted_lambda_res_119860 / 16.0;
        
        // futhark/microgpt.fut:208:24-55
        
        double zp_res_119867 = 1.0e-5 + zs_res_119866;
        
        // futhark/microgpt.fut:208:16-55
        
        double sqrt_res_119868 = futrts_sqrt64(zp_res_119867);
        
        // futhark/microgpt.fut:209:59-70
        
        double zs_res_119869 = 1.0 / sqrt_res_119868;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137189 = 0; i_137189 < (int64_t) 16; i_137189++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_119876 = ((double *) mem_138258)[i_137197 * (int64_t) 16 + i_137189];
            
            // futhark/microgpt.fut:209:37-70
            
            double zt_res_119877 = zs_res_119869 * zt_lhs_119876;
            
            ((double *) mem_138286)[i_137189] = zt_res_119877;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137193 = 0; i_137193 < (int64_t) 16; i_137193++) {
            // futhark/microgpt.fut:210:4-14
            
            double lifted_lambda_res_119885 = ((double *) mem_138286)[i_137193];
            
            ((double *) mem_138293)[i_137193] = lifted_lambda_res_119885;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138281, i_137197 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138293, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138304_cached_sizze_140888 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138304, &mem_138304_cached_sizze_140888, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138305_cached_sizze_140889 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138305, &mem_138305_cached_sizze_140889, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138306_cached_sizze_140890 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138306, &mem_138306_cached_sizze_140890, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138319_cached_sizze_140891 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138319, &mem_138319_cached_sizze_140891, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138320_cached_sizze_140892 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138320, &mem_138320_cached_sizze_140892, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138321_cached_sizze_140893 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138321, &mem_138321_cached_sizze_140893, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137215 = 0; i_137215 < (int64_t) 16; i_137215++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137205 = 0; i_137205 < (int64_t) 16; i_137205++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127606;
            double r_127608 = 0.0;
            
            for (int64_t i_127607 = 0; i_127607 < (int64_t) 16; i_127607++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127609 = ((double *) wqry_mem_138234.mem)[i_137205 * (int64_t) 16 + i_127607];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127610 = ((double *) mem_138281)[i_137215 * (int64_t) 16 + i_127607];
                
                // futhark/microgpt.fut:211:66-105
                
                double zt_res_127611 = zt_lhs_127609 * zt_rhs_127610;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127612 = r_127608 + zt_res_127611;
                double r_tmp_140496 = zp_res_127612;
                
                r_127608 = r_tmp_140496;
            }
            defunc_0_lifted_lambda_res_127606 = r_127608;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127619;
            double r_127621 = 0.0;
            
            for (int64_t i_127620 = 0; i_127620 < (int64_t) 16; i_127620++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127622 = ((double *) wkey_mem_138231.mem)[i_137205 * (int64_t) 16 + i_127620];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127623 = ((double *) mem_138281)[i_137215 * (int64_t) 16 + i_127620];
                
                // futhark/microgpt.fut:212:66-105
                
                double zt_res_127624 = zt_lhs_127622 * zt_rhs_127623;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127625 = r_127621 + zt_res_127624;
                double r_tmp_140497 = zp_res_127625;
                
                r_127621 = r_tmp_140497;
            }
            defunc_0_lifted_lambda_res_127619 = r_127621;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127635;
            double r_127637 = 0.0;
            
            for (int64_t i_127636 = 0; i_127636 < (int64_t) 16; i_127636++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127638 = ((double *) wval_mem_138237.mem)[i_137205 * (int64_t) 16 + i_127636];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127639 = ((double *) mem_138281)[i_137215 * (int64_t) 16 + i_127636];
                
                // futhark/microgpt.fut:213:66-105
                
                double zt_res_127640 = zt_lhs_127638 * zt_rhs_127639;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127641 = r_127637 + zt_res_127640;
                double r_tmp_140498 = zp_res_127641;
                
                r_127637 = r_tmp_140498;
            }
            defunc_0_lifted_lambda_res_127635 = r_127637;
            ((double *) mem_138319)[i_137205] = defunc_0_lifted_lambda_res_127635;
            ((double *) mem_138320)[i_137205] = defunc_0_lifted_lambda_res_127619;
            ((double *) mem_138321)[i_137205] = defunc_0_lifted_lambda_res_127606;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138304, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138305, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138306, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138321, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138352_cached_sizze_140894 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138352, &mem_138352_cached_sizze_140894, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138353_cached_sizze_140895 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138353, &mem_138353_cached_sizze_140895, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138354_cached_sizze_140896 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138354, &mem_138354_cached_sizze_140896, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138370_cached_sizze_140897 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138370, &mem_138370_cached_sizze_140897, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138371_cached_sizze_140898 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138371, &mem_138371_cached_sizze_140898, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138372_cached_sizze_140899 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138372, &mem_138372_cached_sizze_140899, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138385_cached_sizze_140900 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138385, &mem_138385_cached_sizze_140900, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138386_cached_sizze_140901 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138386, &mem_138386_cached_sizze_140901, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138387_cached_sizze_140902 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138387, &mem_138387_cached_sizze_140902, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137245 = 0; i_137245 < (int64_t) 4; i_137245++) {
        // futhark/microgpt.fut:214:69-72
        
        int64_t zp_lhs_127482 = mul64((int64_t) 4, i_137245);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137235 = 0; i_137235 < (int64_t) 16; i_137235++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137225 = 0; i_137225 < (int64_t) 4; i_137225++) {
                // futhark/microgpt.fut:214:74-81
                
                int64_t tmp_127799 = add64(zp_lhs_127482, i_137225);
                
                // futhark/microgpt.fut:214:51-83
                
                bool x_127800 = sle64((int64_t) 0, tmp_127799);
                
                // futhark/microgpt.fut:214:51-83
                
                bool y_127801 = slt64(tmp_127799, (int64_t) 16);
                
                // futhark/microgpt.fut:214:51-83
                
                bool bounds_check_127802 = x_127800 && y_127801;
                
                // futhark/microgpt.fut:214:51-83
                
                bool index_certs_127803;
                
                if (!bounds_check_127802) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_127799, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:214:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:214:15-84\n   #9  futhark/microgpt.fut:463:7-76\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127804 = ((double *) mem_138306)[i_137235 * (int64_t) 16 + tmp_127799];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127812 = ((double *) mem_138305)[i_137235 * (int64_t) 16 + tmp_127799];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127823 = ((double *) mem_138304)[i_137235 * (int64_t) 16 + tmp_127799];
                
                ((double *) mem_138385)[i_137225] = lifted_lambda_res_127823;
                ((double *) mem_138386)[i_137225] = lifted_lambda_res_127812;
                ((double *) mem_138387)[i_137225] = lifted_lambda_res_127804;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138370, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138385, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138371, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138386, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138372, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138387, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138352, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138370, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138353, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138371, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138354, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138372, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138433_cached_sizze_140903 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138433, &mem_138433_cached_sizze_140903, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138439_cached_sizze_140904 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138439, &mem_138439_cached_sizze_140904, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138444_cached_sizze_140905 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138444, &mem_138444_cached_sizze_140905, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138455_cached_sizze_140906 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138455, &mem_138455_cached_sizze_140906, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138460_cached_sizze_140907 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138460, &mem_138460_cached_sizze_140907, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138471_cached_sizze_140908 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138471, &mem_138471_cached_sizze_140908, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138476_cached_sizze_140909 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138476, &mem_138476_cached_sizze_140909, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138483_cached_sizze_140910 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138483, &mem_138483_cached_sizze_140910, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138490_cached_sizze_140911 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138490, &mem_138490_cached_sizze_140911, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138501_cached_sizze_140912 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138501, &mem_138501_cached_sizze_140912, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138506_cached_sizze_140913 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138506, &mem_138506_cached_sizze_140913, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138517_cached_sizze_140914 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138517, &mem_138517_cached_sizze_140914, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138522_cached_sizze_140915 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138522, &mem_138522_cached_sizze_140915, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137301 = 0; i_137301 < (int64_t) 4; i_137301++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137255 = 0; i_137255 < (int64_t) 16; i_137255++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137251 = 0; i_137251 < (int64_t) 16; i_137251++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_120030;
                double r_120032 = 0.0;
                
                for (int64_t i_120031 = 0; i_120031 < (int64_t) 4; i_120031++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_120033 = ((double *) mem_138354)[i_137301 * (int64_t) 64 + i_137255 * (int64_t) 4 + i_120031];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_120034 = ((double *) mem_138353)[i_137301 * (int64_t) 64 + i_137251 * (int64_t) 4 + i_120031];
                    
                    // futhark/microgpt.fut:217:113-164
                    
                    double zt_res_120035 = zt_lhs_120033 * zt_rhs_120034;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_120036 = r_120032 + zt_res_120035;
                    double r_tmp_140511 = zp_res_120036;
                    
                    r_120032 = r_tmp_140511;
                }
                defunc_0_lifted_lambda_res_120030 = r_120032;
                ((double *) mem_138444)[i_137251] = defunc_0_lifted_lambda_res_120030;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138439, i_137255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137263 = 0; i_137263 < (int64_t) 16; i_137263++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137259 = 0; i_137259 < (int64_t) 16; i_137259++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_120051 = ((double *) mem_138439)[i_137263 * (int64_t) 16 + i_137259];
                
                // futhark/microgpt.fut:218:47-78
                
                double zs_res_120052 = zs_lhs_120051 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_120053 = ((double *) mask_mem_138241.mem)[i_137263 * (int64_t) 16 + i_137259];
                
                // futhark/microgpt.fut:218:65-102
                
                double zp_res_120054 = zs_res_120052 + zp_rhs_120053;
                
                ((double *) mem_138460)[i_137259] = zp_res_120054;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138455, i_137263 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138460, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137281 = 0; i_137281 < (int64_t) 16; i_137281++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_127926;
            double redout_137265 = -INFINITY;
            
            for (int64_t i_137266 = 0; i_137266 < (int64_t) 16; i_137266++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127850 = ((double *) mem_138455)[i_137281 * (int64_t) 16 + i_137266];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_120075 = fmax64(lifted_lambda_res_127850, redout_137265);
                double redout_tmp_140515 = max_res_120075;
                
                redout_137265 = redout_tmp_140515;
            }
            defunc_0_reduce_res_127926 = redout_137265;
            // futhark/microgpt.fut:220:67-76
            
            double neg_res_120076 = -defunc_0_reduce_res_127926;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137269 = 0; i_137269 < (int64_t) 16; i_137269++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_120083 = ((double *) mem_138455)[i_137281 * (int64_t) 16 + i_137269];
                
                // futhark/microgpt.fut:220:44-76
                
                double zp_res_120084 = neg_res_120076 + zp_lhs_120083;
                
                // futhark/microgpt.fut:220:37-76
                
                double exp_res_120085 = futrts_exp64(zp_res_120084);
                
                ((double *) mem_138476)[i_137269] = exp_res_120085;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120087;
            double r_120089 = 0.0;
            
            for (int64_t i_120088 = 0; i_120088 < (int64_t) 16; i_120088++) {
                // futhark/microgpt.fut:221:36-46
                
                double lifted_lambda_res_120090 = ((double *) mem_138476)[i_120088];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120091 = r_120089 + lifted_lambda_res_120090;
                double r_tmp_140517 = zp_res_120091;
                
                r_120089 = r_tmp_140517;
            }
            defunc_0_lifted_lambda_res_120087 = r_120089;
            // futhark/microgpt.fut:222:53-64
            
            double zs_res_120092 = 1.0 / defunc_0_lifted_lambda_res_120087;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137273 = 0; i_137273 < (int64_t) 16; i_137273++) {
                // futhark/microgpt.fut:222:37-47
                
                double zt_lhs_120099 = ((double *) mem_138476)[i_137273];
                
                // futhark/microgpt.fut:222:37-64
                
                double zt_res_120100 = zs_res_120092 * zt_lhs_120099;
                
                ((double *) mem_138483)[i_137273] = zt_res_120100;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137277 = 0; i_137277 < (int64_t) 16; i_137277++) {
                // futhark/microgpt.fut:223:4-14
                
                double lifted_lambda_res_120108 = ((double *) mem_138483)[i_137277];
                
                ((double *) mem_138490)[i_137277] = lifted_lambda_res_120108;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138471, i_137281 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138490, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137289 = 0; i_137289 < (int64_t) 16; i_137289++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137285 = 0; i_137285 < (int64_t) 4; i_137285++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_120123;
                double r_120125 = 0.0;
                
                for (int64_t i_120124 = 0; i_120124 < (int64_t) 16; i_120124++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_120126 = ((double *) mem_138471)[i_137289 * (int64_t) 16 + i_120124];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_120127 = ((double *) mem_138352)[i_137301 * (int64_t) 64 + i_120124 * (int64_t) 4 + i_137285];
                    
                    // futhark/microgpt.fut:224:66-111
                    
                    double zt_res_120128 = zt_lhs_120126 * zt_rhs_120127;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_120129 = r_120125 + zt_res_120128;
                    double r_tmp_140522 = zp_res_120129;
                    
                    r_120125 = r_tmp_140522;
                }
                defunc_0_lifted_lambda_res_120123 = r_120125;
                ((double *) mem_138506)[i_137285] = defunc_0_lifted_lambda_res_120123;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138501, i_137289 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138506, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137297 = 0; i_137297 < (int64_t) 16; i_137297++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137293 = 0; i_137293 < (int64_t) 4; i_137293++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_120144 = ((double *) mem_138501)[i_137297 * (int64_t) 4 + i_137293];
                
                ((double *) mem_138522)[i_137293] = lifted_lambda_res_120144;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138517, i_137297 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138522, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138433, i_137301 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138517, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138538_cached_sizze_140916 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138538, &mem_138538_cached_sizze_140916, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138543_cached_sizze_140917 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138543, &mem_138543_cached_sizze_140917, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137309 = 0; i_137309 < (int64_t) 16; i_137309++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137305 = 0; i_137305 < (int64_t) 16; i_137305++) {
            // futhark/microgpt.fut:226:54-57
            
            int64_t tmp_120156 = sdiv64(i_137305, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool x_120157 = sle64((int64_t) 0, tmp_120156);
            
            // futhark/microgpt.fut:226:44-59
            
            bool y_120158 = slt64(tmp_120156, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-59
            
            bool bounds_check_120159 = x_120157 && y_120158;
            
            // futhark/microgpt.fut:226:44-59
            
            bool index_certs_120160;
            
            if (!bounds_check_120159) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_120156, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:463:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:226:74-77
            
            int64_t tmp_120161 = smod64(i_137305, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool x_120162 = sle64((int64_t) 0, tmp_120161);
            
            // futhark/microgpt.fut:226:44-79
            
            bool y_120163 = slt64(tmp_120161, (int64_t) 4);
            
            // futhark/microgpt.fut:226:44-79
            
            bool bounds_check_120164 = x_120162 && y_120163;
            
            // futhark/microgpt.fut:226:44-79
            
            bool index_certs_120165;
            
            if (!bounds_check_120164) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_120161, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:226:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:226:15-80\n   #6  futhark/microgpt.fut:463:7-76\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_120166 = ((double *) mem_138433)[tmp_120156 * (int64_t) 64 + i_137309 * (int64_t) 4 + tmp_120161];
            
            ((double *) mem_138543)[i_137305] = lifted_lambda_res_120166;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138538, i_137309 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138543, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138554_cached_sizze_140918 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138554, &mem_138554_cached_sizze_140918, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138559_cached_sizze_140919 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138559, &mem_138559_cached_sizze_140919, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137317 = 0; i_137317 < (int64_t) 16; i_137317++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137313 = 0; i_137313 < (int64_t) 16; i_137313++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120181;
            double r_120183 = 0.0;
            
            for (int64_t i_120182 = 0; i_120182 < (int64_t) 16; i_120182++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120184 = ((double *) wout_mem_138232.mem)[i_137313 * (int64_t) 16 + i_120182];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120185 = ((double *) mem_138538)[i_137317 * (int64_t) 16 + i_120182];
                
                // futhark/microgpt.fut:227:67-106
                
                double zt_res_120186 = zt_lhs_120184 * zt_rhs_120185;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120187 = r_120183 + zt_res_120186;
                double r_tmp_140529 = zp_res_120187;
                
                r_120183 = r_tmp_140529;
            }
            defunc_0_lifted_lambda_res_120181 = r_120183;
            ((double *) mem_138559)[i_137313] = defunc_0_lifted_lambda_res_120181;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138554, i_137317 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138559, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138570_cached_sizze_140920 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138570, &mem_138570_cached_sizze_140920, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138575_cached_sizze_140921 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138575, &mem_138575_cached_sizze_140921, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137325 = 0; i_137325 < (int64_t) 16; i_137325++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137321 = 0; i_137321 < (int64_t) 16; i_137321++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_120202 = ((double *) mem_138554)[i_137325 * (int64_t) 16 + i_137321];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_120203 = ((double *) mem_138258)[i_137325 * (int64_t) 16 + i_137321];
            
            // futhark/microgpt.fut:228:46-84
            
            double zp_res_120204 = zp_lhs_120202 + zp_rhs_120203;
            
            ((double *) mem_138575)[i_137321] = zp_res_120204;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138570, i_137325 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138575, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138586_cached_sizze_140922 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138586, &mem_138586_cached_sizze_140922, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138591_cached_sizze_140923 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138591, &mem_138591_cached_sizze_140923, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138598_cached_sizze_140924 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138598, &mem_138598_cached_sizze_140924, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137337 = 0; i_137337 < (int64_t) 16; i_137337++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_120213;
        double r_120215 = 0.0;
        
        for (int64_t i_120214 = 0; i_120214 < (int64_t) 16; i_120214++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_120216 = ((double *) mem_138570)[i_137337 * (int64_t) 16 + i_120214];
            
            // futhark/microgpt.fut:229:79-118
            
            double zt_res_120217 = zt_lhs_120216 * zt_lhs_120216;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_120218 = r_120215 + zt_res_120217;
            double r_tmp_140533 = zp_res_120218;
            
            r_120215 = r_tmp_140533;
        }
        defunc_0_lifted_lambda_res_120213 = r_120215;
        // futhark/microgpt.fut:229:58-136
        
        double zs_res_120219 = defunc_0_lifted_lambda_res_120213 / 16.0;
        
        // futhark/microgpt.fut:230:24-55
        
        double zp_res_120220 = 1.0e-5 + zs_res_120219;
        
        // futhark/microgpt.fut:230:16-55
        
        double sqrt_res_120221 = futrts_sqrt64(zp_res_120220);
        
        // futhark/microgpt.fut:231:60-71
        
        double zs_res_120222 = 1.0 / sqrt_res_120221;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137329 = 0; i_137329 < (int64_t) 16; i_137329++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_120229 = ((double *) mem_138570)[i_137337 * (int64_t) 16 + i_137329];
            
            // futhark/microgpt.fut:231:37-71
            
            double zt_res_120230 = zs_res_120222 * zt_lhs_120229;
            
            ((double *) mem_138591)[i_137329] = zt_res_120230;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137333 = 0; i_137333 < (int64_t) 16; i_137333++) {
            // futhark/microgpt.fut:232:4-14
            
            double lifted_lambda_res_120238 = ((double *) mem_138591)[i_137333];
            
            ((double *) mem_138598)[i_137333] = lifted_lambda_res_120238;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138586, i_137337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138598, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138609_cached_sizze_140925 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138609, &mem_138609_cached_sizze_140925, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138614_cached_sizze_140926 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138614, &mem_138614_cached_sizze_140926, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137345 = 0; i_137345 < (int64_t) 16; i_137345++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137341 = 0; i_137341 < (int64_t) 64; i_137341++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120254;
            double r_120256 = 0.0;
            
            for (int64_t i_120255 = 0; i_120255 < (int64_t) 16; i_120255++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120257 = ((double *) wup_mem_138236.mem)[i_137341 * (int64_t) 16 + i_120255];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120258 = ((double *) mem_138586)[i_137345 * (int64_t) 16 + i_120255];
                
                // futhark/microgpt.fut:233:67-106
                
                double zt_res_120259 = zt_lhs_120257 * zt_rhs_120258;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120260 = r_120256 + zt_res_120259;
                double r_tmp_140538 = zp_res_120260;
                
                r_120256 = r_tmp_140538;
            }
            defunc_0_lifted_lambda_res_120254 = r_120256;
            ((double *) mem_138614)[i_137341] = defunc_0_lifted_lambda_res_120254;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138609, i_137345 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138614, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138625_cached_sizze_140927 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138625, &mem_138625_cached_sizze_140927, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138630_cached_sizze_140928 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138630, &mem_138630_cached_sizze_140928, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137353 = 0; i_137353 < (int64_t) 16; i_137353++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137349 = 0; i_137349 < (int64_t) 64; i_137349++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_120275 = ((double *) mem_138609)[i_137353 * (int64_t) 64 + i_137349];
            
            // futhark/microgpt.fut:234:45-73
            
            double max_res_120276 = fmax64(0.0, max_arg0_120275);
            
            ((double *) mem_138630)[i_137349] = max_res_120276;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138625, i_137353 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138630, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138641_cached_sizze_140929 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138641, &mem_138641_cached_sizze_140929, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138646_cached_sizze_140930 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138646, &mem_138646_cached_sizze_140930, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137361 = 0; i_137361 < (int64_t) 16; i_137361++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137357 = 0; i_137357 < (int64_t) 16; i_137357++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120291;
            double r_120293 = 0.0;
            
            for (int64_t i_120292 = 0; i_120292 < (int64_t) 64; i_120292++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120294 = ((double *) wdown_mem_138230.mem)[i_137357 * (int64_t) 64 + i_120292];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120295 = ((double *) mem_138625)[i_137361 * (int64_t) 64 + i_120292];
                
                // futhark/microgpt.fut:235:67-108
                
                double zt_res_120296 = zt_lhs_120294 * zt_rhs_120295;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120297 = r_120293 + zt_res_120296;
                double r_tmp_140543 = zp_res_120297;
                
                r_120293 = r_tmp_140543;
            }
            defunc_0_lifted_lambda_res_120291 = r_120293;
            ((double *) mem_138646)[i_137357] = defunc_0_lifted_lambda_res_120291;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138641, i_137361 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138646, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138657_cached_sizze_140931 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138657, &mem_138657_cached_sizze_140931, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138662_cached_sizze_140932 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138662, &mem_138662_cached_sizze_140932, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137369 = 0; i_137369 < (int64_t) 16; i_137369++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137365 = 0; i_137365 < (int64_t) 16; i_137365++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_120312 = ((double *) mem_138641)[i_137369 * (int64_t) 16 + i_137365];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_120313 = ((double *) mem_138570)[i_137369 * (int64_t) 16 + i_137365];
            
            // futhark/microgpt.fut:236:46-85
            
            double zp_res_120314 = zp_lhs_120312 + zp_rhs_120313;
            
            ((double *) mem_138662)[i_137365] = zp_res_120314;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138657, i_137369 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138662, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138673_cached_sizze_140933 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138673, &mem_138673_cached_sizze_140933, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138678_cached_sizze_140934 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138678, &mem_138678_cached_sizze_140934, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137377 = 0; i_137377 < (int64_t) 16; i_137377++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137373 = 0; i_137373 < (int64_t) 27; i_137373++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120330;
            double r_120332 = 0.0;
            
            for (int64_t i_120331 = 0; i_120331 < (int64_t) 16; i_120331++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120333 = ((double *) wvoc_mem_138238.mem)[i_137373 * (int64_t) 16 + i_120331];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120334 = ((double *) mem_138657)[i_137377 * (int64_t) 16 + i_120331];
                
                // futhark/microgpt.fut:237:67-107
                
                double zt_res_120335 = zt_lhs_120333 * zt_rhs_120334;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120336 = r_120332 + zt_res_120335;
                double r_tmp_140548 = zp_res_120336;
                
                r_120332 = r_tmp_140548;
            }
            defunc_0_lifted_lambda_res_120330 = r_120332;
            ((double *) mem_138678)[i_137373] = defunc_0_lifted_lambda_res_120330;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138673, i_137377 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138678, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138689, (int64_t) 128, "mem_138689")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138693_cached_sizze_140935 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138693, &mem_138693_cached_sizze_140935, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138700_cached_sizze_140936 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138700, &mem_138700_cached_sizze_140936, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137391 = 0; i_137391 < (int64_t) 16; i_137391++) {
        double x_127949;
        double redout_137379 = -INFINITY;
        
        for (int64_t i_137380 = 0; i_137380 < (int64_t) 27; i_137380++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_127896 = ((double *) mem_138673)[i_137391 * (int64_t) 27 + i_137380];
            
            // futhark/microgpt.fut:115:13-33
            
            double max_res_120360 = fmax64(lifted_lambda_res_127896, redout_137379);
            double redout_tmp_140550 = max_res_120360;
            
            redout_137379 = redout_tmp_140550;
        }
        x_127949 = redout_137379;
        // futhark/microgpt.fut:239:67-76
        
        double neg_res_120361 = -x_127949;
        
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_120345;
        double r_120347 = 0.0;
        
        for (int64_t i_120346 = 0; i_120346 < (int64_t) 27; i_120346++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137383 = 0; i_137383 < (int64_t) 27; i_137383++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_120368 = ((double *) mem_138673)[i_137391 * (int64_t) 27 + i_137383];
                
                // futhark/microgpt.fut:239:44-76
                
                double zp_res_120369 = neg_res_120361 + zp_lhs_120368;
                
                // futhark/microgpt.fut:239:37-76
                
                double exp_res_120370 = futrts_exp64(zp_res_120369);
                
                ((double *) mem_138693)[i_137383] = exp_res_120370;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120372;
            double r_120374 = 0.0;
            
            for (int64_t i_120373 = 0; i_120373 < (int64_t) 27; i_120373++) {
                // futhark/microgpt.fut:240:36-46
                
                double lifted_lambda_res_120375 = ((double *) mem_138693)[i_120373];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120376 = r_120374 + lifted_lambda_res_120375;
                double r_tmp_140553 = zp_res_120376;
                
                r_120374 = r_tmp_140553;
            }
            defunc_0_lifted_lambda_res_120372 = r_120374;
            // futhark/microgpt.fut:241:53-64
            
            double zs_res_120377 = 1.0 / defunc_0_lifted_lambda_res_120372;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137387 = 0; i_137387 < (int64_t) 27; i_137387++) {
                // futhark/microgpt.fut:241:37-47
                
                double zt_lhs_120384 = ((double *) mem_138693)[i_137387];
                
                // futhark/microgpt.fut:241:37-64
                
                double zt_res_120385 = zs_res_120377 * zt_lhs_120384;
                
                ((double *) mem_138700)[i_137387] = zt_res_120385;
            }
            // futhark/microgpt.fut:242:12-22
            
            double log_arg0_120387 = ((double *) mem_138700)[i_120346];
            
            // futhark/microgpt.fut:242:6-22
            
            double log_res_120388 = futrts_log64(log_arg0_120387);
            
            // futhark/microgpt.fut:71:46-49
            
            double zt_rhs_120389 = ((double *) target_mem_138240.mem)[i_137391 * (int64_t) 27 + i_120346];
            
            // futhark/microgpt.fut:242:6-48
            
            double zt_res_120390 = log_res_120388 * zt_rhs_120389;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_120391 = r_120347 + zt_res_120390;
            double r_tmp_140551 = zp_res_120391;
            
            r_120347 = r_tmp_140551;
        }
        defunc_0_lifted_lambda_res_120345 = r_120347;
        // futhark/microgpt.fut:238:37-242:54
        
        double neg_res_120392 = -defunc_0_lifted_lambda_res_120345;
        
        ((double *) mem_138689.mem)[i_137391] = neg_res_120392;
    }
    // futhark/microgpt.fut:71:13-49
    
    double defunc_0_lifted_lambda_res_120394;
    double r_120396 = 0.0;
    
    for (int64_t i_120395 = 0; i_120395 < (int64_t) 16; i_120395++) {
        // futhark/microgpt.fut:243:37-47
        
        double lifted_lambda_res_120397 = ((double *) mem_138689.mem)[i_120395];
        
        // futhark/microgpt.fut:71:40-49
        
        double zp_res_120398 = r_120396 + lifted_lambda_res_120397;
        double r_tmp_140555 = zp_res_120398;
        
        r_120396 = r_tmp_140555;
    }
    defunc_0_lifted_lambda_res_120394 = r_120396;
    // futhark/microgpt.fut:243:17-64
    
    double zs_res_120399 = defunc_0_lifted_lambda_res_120394 / 16.0;
    
    if (memblock_set(ctx, &mem_out_140478, &mem_138689, "mem_138689") != 0)
        return 1;
    prim_out_140479 = zs_res_120399;
    if (memblock_set(ctx, &*mem_out_p_140878, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    *out_prim_out_140879 = prim_out_140479;
    
  cleanup:
    {
        free(mem_138242);
        free(mem_138247);
        free(mem_138258);
        free(mem_138263);
        free(mem_138270);
        free(mem_138281);
        free(mem_138286);
        free(mem_138293);
        free(mem_138304);
        free(mem_138305);
        free(mem_138306);
        free(mem_138319);
        free(mem_138320);
        free(mem_138321);
        free(mem_138352);
        free(mem_138353);
        free(mem_138354);
        free(mem_138370);
        free(mem_138371);
        free(mem_138372);
        free(mem_138385);
        free(mem_138386);
        free(mem_138387);
        free(mem_138433);
        free(mem_138439);
        free(mem_138444);
        free(mem_138455);
        free(mem_138460);
        free(mem_138471);
        free(mem_138476);
        free(mem_138483);
        free(mem_138490);
        free(mem_138501);
        free(mem_138506);
        free(mem_138517);
        free(mem_138522);
        free(mem_138538);
        free(mem_138543);
        free(mem_138554);
        free(mem_138559);
        free(mem_138570);
        free(mem_138575);
        free(mem_138586);
        free(mem_138591);
        free(mem_138598);
        free(mem_138609);
        free(mem_138614);
        free(mem_138625);
        free(mem_138630);
        free(mem_138641);
        free(mem_138646);
        free(mem_138657);
        free(mem_138662);
        free(mem_138673);
        free(mem_138678);
        free(mem_138693);
        free(mem_138700);
        if (memblock_unref(ctx, &mem_138689, "mem_138689") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_140937, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock tokens_mem_138239, struct memblock mask_mem_138240)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_138241_cached_sizze_140938 = 0;
    unsigned char *mem_138241 = NULL;
    int64_t mem_138246_cached_sizze_140939 = 0;
    unsigned char *mem_138246 = NULL;
    int64_t mem_138257_cached_sizze_140940 = 0;
    unsigned char *mem_138257 = NULL;
    int64_t mem_138262_cached_sizze_140941 = 0;
    unsigned char *mem_138262 = NULL;
    int64_t mem_138269_cached_sizze_140942 = 0;
    unsigned char *mem_138269 = NULL;
    int64_t mem_138280_cached_sizze_140943 = 0;
    unsigned char *mem_138280 = NULL;
    int64_t mem_138285_cached_sizze_140944 = 0;
    unsigned char *mem_138285 = NULL;
    int64_t mem_138292_cached_sizze_140945 = 0;
    unsigned char *mem_138292 = NULL;
    int64_t mem_138303_cached_sizze_140946 = 0;
    unsigned char *mem_138303 = NULL;
    int64_t mem_138304_cached_sizze_140947 = 0;
    unsigned char *mem_138304 = NULL;
    int64_t mem_138305_cached_sizze_140948 = 0;
    unsigned char *mem_138305 = NULL;
    int64_t mem_138318_cached_sizze_140949 = 0;
    unsigned char *mem_138318 = NULL;
    int64_t mem_138319_cached_sizze_140950 = 0;
    unsigned char *mem_138319 = NULL;
    int64_t mem_138320_cached_sizze_140951 = 0;
    unsigned char *mem_138320 = NULL;
    int64_t mem_138351_cached_sizze_140952 = 0;
    unsigned char *mem_138351 = NULL;
    int64_t mem_138352_cached_sizze_140953 = 0;
    unsigned char *mem_138352 = NULL;
    int64_t mem_138353_cached_sizze_140954 = 0;
    unsigned char *mem_138353 = NULL;
    int64_t mem_138369_cached_sizze_140955 = 0;
    unsigned char *mem_138369 = NULL;
    int64_t mem_138370_cached_sizze_140956 = 0;
    unsigned char *mem_138370 = NULL;
    int64_t mem_138371_cached_sizze_140957 = 0;
    unsigned char *mem_138371 = NULL;
    int64_t mem_138384_cached_sizze_140958 = 0;
    unsigned char *mem_138384 = NULL;
    int64_t mem_138385_cached_sizze_140959 = 0;
    unsigned char *mem_138385 = NULL;
    int64_t mem_138386_cached_sizze_140960 = 0;
    unsigned char *mem_138386 = NULL;
    int64_t mem_138432_cached_sizze_140961 = 0;
    unsigned char *mem_138432 = NULL;
    int64_t mem_138438_cached_sizze_140962 = 0;
    unsigned char *mem_138438 = NULL;
    int64_t mem_138443_cached_sizze_140963 = 0;
    unsigned char *mem_138443 = NULL;
    int64_t mem_138454_cached_sizze_140964 = 0;
    unsigned char *mem_138454 = NULL;
    int64_t mem_138459_cached_sizze_140965 = 0;
    unsigned char *mem_138459 = NULL;
    int64_t mem_138470_cached_sizze_140966 = 0;
    unsigned char *mem_138470 = NULL;
    int64_t mem_138475_cached_sizze_140967 = 0;
    unsigned char *mem_138475 = NULL;
    int64_t mem_138482_cached_sizze_140968 = 0;
    unsigned char *mem_138482 = NULL;
    int64_t mem_138489_cached_sizze_140969 = 0;
    unsigned char *mem_138489 = NULL;
    int64_t mem_138500_cached_sizze_140970 = 0;
    unsigned char *mem_138500 = NULL;
    int64_t mem_138505_cached_sizze_140971 = 0;
    unsigned char *mem_138505 = NULL;
    int64_t mem_138516_cached_sizze_140972 = 0;
    unsigned char *mem_138516 = NULL;
    int64_t mem_138521_cached_sizze_140973 = 0;
    unsigned char *mem_138521 = NULL;
    int64_t mem_138537_cached_sizze_140974 = 0;
    unsigned char *mem_138537 = NULL;
    int64_t mem_138542_cached_sizze_140975 = 0;
    unsigned char *mem_138542 = NULL;
    int64_t mem_138553_cached_sizze_140976 = 0;
    unsigned char *mem_138553 = NULL;
    int64_t mem_138558_cached_sizze_140977 = 0;
    unsigned char *mem_138558 = NULL;
    int64_t mem_138569_cached_sizze_140978 = 0;
    unsigned char *mem_138569 = NULL;
    int64_t mem_138574_cached_sizze_140979 = 0;
    unsigned char *mem_138574 = NULL;
    int64_t mem_138585_cached_sizze_140980 = 0;
    unsigned char *mem_138585 = NULL;
    int64_t mem_138590_cached_sizze_140981 = 0;
    unsigned char *mem_138590 = NULL;
    int64_t mem_138597_cached_sizze_140982 = 0;
    unsigned char *mem_138597 = NULL;
    int64_t mem_138608_cached_sizze_140983 = 0;
    unsigned char *mem_138608 = NULL;
    int64_t mem_138613_cached_sizze_140984 = 0;
    unsigned char *mem_138613 = NULL;
    int64_t mem_138624_cached_sizze_140985 = 0;
    unsigned char *mem_138624 = NULL;
    int64_t mem_138629_cached_sizze_140986 = 0;
    unsigned char *mem_138629 = NULL;
    int64_t mem_138640_cached_sizze_140987 = 0;
    unsigned char *mem_138640 = NULL;
    int64_t mem_138645_cached_sizze_140988 = 0;
    unsigned char *mem_138645 = NULL;
    int64_t mem_138656_cached_sizze_140989 = 0;
    unsigned char *mem_138656 = NULL;
    int64_t mem_138661_cached_sizze_140990 = 0;
    unsigned char *mem_138661 = NULL;
    int64_t mem_138672_cached_sizze_140991 = 0;
    unsigned char *mem_138672 = NULL;
    int64_t mem_138677_cached_sizze_140992 = 0;
    unsigned char *mem_138677 = NULL;
    int64_t mem_138693_cached_sizze_140993 = 0;
    unsigned char *mem_138693 = NULL;
    struct memblock mem_138688;
    
    mem_138688.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_138241_cached_sizze_140938 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138241, &mem_138241_cached_sizze_140938, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138246_cached_sizze_140939 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138246, &mem_138246_cached_sizze_140939, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137173 = 0; i_137173 < (int64_t) 16; i_137173++) {
        // futhark/microgpt.fut:457:41-50
        
        int64_t tmp_119784 = ((int64_t *) tokens_mem_138239.mem)[i_137173];
        
        // futhark/microgpt.fut:457:37-51
        
        bool x_119785 = sle64((int64_t) 0, tmp_119784);
        
        // futhark/microgpt.fut:457:37-51
        
        bool y_119786 = slt64(tmp_119784, (int64_t) 27);
        
        // futhark/microgpt.fut:457:37-51
        
        bool bounds_check_119787 = x_119785 && y_119786;
        
        // futhark/microgpt.fut:457:37-51
        
        bool index_certs_119788;
        
        if (!bounds_check_119787) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_119784, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:457:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:457:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137169 = 0; i_137169 < (int64_t) 16; i_137169++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_119795 = ((double *) wte_mem_138235.mem)[tmp_119784 * (int64_t) 16 + i_137169];
            
            ((double *) mem_138246)[i_137169] = lifted_lambda_res_119795;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138241, i_137173 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138246, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138257_cached_sizze_140940 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138257, &mem_138257_cached_sizze_140940, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138262_cached_sizze_140941 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138262, &mem_138262_cached_sizze_140941, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138269_cached_sizze_140942 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138269, &mem_138269_cached_sizze_140942, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137185 = 0; i_137185 < (int64_t) 16; i_137185++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_119821;
        double r_119823 = 0.0;
        
        for (int64_t i_119822 = 0; i_119822 < (int64_t) 16; i_119822++) {
            // futhark/microgpt.fut:71:46-49
            
            double zp_lhs_119824 = ((double *) wpe_mem_138233.mem)[i_137185 * (int64_t) 16 + i_119822];
            
            // futhark/microgpt.fut:71:46-49
            
            double zp_rhs_119825 = ((double *) mem_138241)[i_137185 * (int64_t) 16 + i_119822];
            
            // futhark/microgpt.fut:148:76-116
            
            double zp_res_119826 = zp_lhs_119824 + zp_rhs_119825;
            
            // futhark/microgpt.fut:148:94-163
            
            double zt_res_119827 = zp_res_119826 * zp_res_119826;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_119828 = r_119823 + zt_res_119827;
            double r_tmp_140482 = zp_res_119828;
            
            r_119823 = r_tmp_140482;
        }
        defunc_0_lifted_lambda_res_119821 = r_119823;
        // futhark/microgpt.fut:148:54-182
        
        double zs_res_119829 = defunc_0_lifted_lambda_res_119821 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_119830 = 1.0e-5 + zs_res_119829;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_119831 = futrts_sqrt64(zp_res_119830);
        
        // futhark/microgpt.fut:150:85-96
        
        double zs_res_119832 = 1.0 / sqrt_res_119831;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137177 = 0; i_137177 < (int64_t) 16; i_137177++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_119839 = ((double *) wpe_mem_138233.mem)[i_137185 * (int64_t) 16 + i_137177];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_119840 = ((double *) mem_138241)[i_137185 * (int64_t) 16 + i_137177];
            
            // futhark/microgpt.fut:150:38-78
            
            double zp_res_119841 = zp_lhs_119839 + zp_rhs_119840;
            
            // futhark/microgpt.fut:150:56-96
            
            double zt_res_119842 = zs_res_119832 * zp_res_119841;
            
            ((double *) mem_138262)[i_137177] = zt_res_119842;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137181 = 0; i_137181 < (int64_t) 16; i_137181++) {
            // futhark/microgpt.fut:151:4-14
            
            double lifted_lambda_res_119850 = ((double *) mem_138262)[i_137181];
            
            ((double *) mem_138269)[i_137181] = lifted_lambda_res_119850;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138257, i_137185 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138269, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138280_cached_sizze_140943 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138280, &mem_138280_cached_sizze_140943, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138285_cached_sizze_140944 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138285, &mem_138285_cached_sizze_140944, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138292_cached_sizze_140945 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138292, &mem_138292_cached_sizze_140945, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137197 = 0; i_137197 < (int64_t) 16; i_137197++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_119859;
        double r_119861 = 0.0;
        
        for (int64_t i_119860 = 0; i_119860 < (int64_t) 16; i_119860++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_119862 = ((double *) mem_138257)[i_137197 * (int64_t) 16 + i_119860];
            
            // futhark/microgpt.fut:152:78-115
            
            double zt_res_119863 = zt_lhs_119862 * zt_lhs_119862;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_119864 = r_119861 + zt_res_119863;
            double r_tmp_140486 = zp_res_119864;
            
            r_119861 = r_tmp_140486;
        }
        defunc_0_lifted_lambda_res_119859 = r_119861;
        // futhark/microgpt.fut:152:57-133
        
        double zs_res_119865 = defunc_0_lifted_lambda_res_119859 / 16.0;
        
        // futhark/microgpt.fut:153:24-55
        
        double zp_res_119866 = 1.0e-5 + zs_res_119865;
        
        // futhark/microgpt.fut:153:16-55
        
        double sqrt_res_119867 = futrts_sqrt64(zp_res_119866);
        
        // futhark/microgpt.fut:154:59-70
        
        double zs_res_119868 = 1.0 / sqrt_res_119867;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137189 = 0; i_137189 < (int64_t) 16; i_137189++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_119875 = ((double *) mem_138257)[i_137197 * (int64_t) 16 + i_137189];
            
            // futhark/microgpt.fut:154:37-70
            
            double zt_res_119876 = zs_res_119868 * zt_lhs_119875;
            
            ((double *) mem_138285)[i_137189] = zt_res_119876;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137193 = 0; i_137193 < (int64_t) 16; i_137193++) {
            // futhark/microgpt.fut:155:4-14
            
            double lifted_lambda_res_119884 = ((double *) mem_138285)[i_137193];
            
            ((double *) mem_138292)[i_137193] = lifted_lambda_res_119884;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138280, i_137197 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138292, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138303_cached_sizze_140946 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138303, &mem_138303_cached_sizze_140946, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138304_cached_sizze_140947 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138304, &mem_138304_cached_sizze_140947, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138305_cached_sizze_140948 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138305, &mem_138305_cached_sizze_140948, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138318_cached_sizze_140949 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138318, &mem_138318_cached_sizze_140949, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138319_cached_sizze_140950 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138319, &mem_138319_cached_sizze_140950, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138320_cached_sizze_140951 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138320, &mem_138320_cached_sizze_140951, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137215 = 0; i_137215 < (int64_t) 16; i_137215++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137205 = 0; i_137205 < (int64_t) 16; i_137205++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127606;
            double r_127608 = 0.0;
            
            for (int64_t i_127607 = 0; i_127607 < (int64_t) 16; i_127607++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127609 = ((double *) wqry_mem_138234.mem)[i_137205 * (int64_t) 16 + i_127607];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127610 = ((double *) mem_138280)[i_137215 * (int64_t) 16 + i_127607];
                
                // futhark/microgpt.fut:156:66-105
                
                double zt_res_127611 = zt_lhs_127609 * zt_rhs_127610;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127612 = r_127608 + zt_res_127611;
                double r_tmp_140495 = zp_res_127612;
                
                r_127608 = r_tmp_140495;
            }
            defunc_0_lifted_lambda_res_127606 = r_127608;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127619;
            double r_127621 = 0.0;
            
            for (int64_t i_127620 = 0; i_127620 < (int64_t) 16; i_127620++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127622 = ((double *) wkey_mem_138231.mem)[i_137205 * (int64_t) 16 + i_127620];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127623 = ((double *) mem_138280)[i_137215 * (int64_t) 16 + i_127620];
                
                // futhark/microgpt.fut:157:66-105
                
                double zt_res_127624 = zt_lhs_127622 * zt_rhs_127623;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127625 = r_127621 + zt_res_127624;
                double r_tmp_140496 = zp_res_127625;
                
                r_127621 = r_tmp_140496;
            }
            defunc_0_lifted_lambda_res_127619 = r_127621;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127635;
            double r_127637 = 0.0;
            
            for (int64_t i_127636 = 0; i_127636 < (int64_t) 16; i_127636++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127638 = ((double *) wval_mem_138237.mem)[i_137205 * (int64_t) 16 + i_127636];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127639 = ((double *) mem_138280)[i_137215 * (int64_t) 16 + i_127636];
                
                // futhark/microgpt.fut:158:66-105
                
                double zt_res_127640 = zt_lhs_127638 * zt_rhs_127639;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127641 = r_127637 + zt_res_127640;
                double r_tmp_140497 = zp_res_127641;
                
                r_127637 = r_tmp_140497;
            }
            defunc_0_lifted_lambda_res_127635 = r_127637;
            ((double *) mem_138318)[i_137205] = defunc_0_lifted_lambda_res_127635;
            ((double *) mem_138319)[i_137205] = defunc_0_lifted_lambda_res_127619;
            ((double *) mem_138320)[i_137205] = defunc_0_lifted_lambda_res_127606;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138303, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138318, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138304, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138305, i_137215 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138351_cached_sizze_140952 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138351, &mem_138351_cached_sizze_140952, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138352_cached_sizze_140953 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138352, &mem_138352_cached_sizze_140953, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138353_cached_sizze_140954 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138353, &mem_138353_cached_sizze_140954, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138369_cached_sizze_140955 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138369, &mem_138369_cached_sizze_140955, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138370_cached_sizze_140956 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138370, &mem_138370_cached_sizze_140956, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138371_cached_sizze_140957 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138371, &mem_138371_cached_sizze_140957, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138384_cached_sizze_140958 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138384, &mem_138384_cached_sizze_140958, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138385_cached_sizze_140959 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138385, &mem_138385_cached_sizze_140959, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138386_cached_sizze_140960 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138386, &mem_138386_cached_sizze_140960, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137245 = 0; i_137245 < (int64_t) 4; i_137245++) {
        // futhark/microgpt.fut:159:69-72
        
        int64_t zp_lhs_127482 = mul64((int64_t) 4, i_137245);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137235 = 0; i_137235 < (int64_t) 16; i_137235++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137225 = 0; i_137225 < (int64_t) 4; i_137225++) {
                // futhark/microgpt.fut:159:74-81
                
                int64_t tmp_127799 = add64(zp_lhs_127482, i_137225);
                
                // futhark/microgpt.fut:159:51-83
                
                bool x_127800 = sle64((int64_t) 0, tmp_127799);
                
                // futhark/microgpt.fut:159:51-83
                
                bool y_127801 = slt64(tmp_127799, (int64_t) 16);
                
                // futhark/microgpt.fut:159:51-83
                
                bool bounds_check_127802 = x_127800 && y_127801;
                
                // futhark/microgpt.fut:159:51-83
                
                bool index_certs_127803;
                
                if (!bounds_check_127802) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_127799, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:159:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:159:15-84\n   #9  futhark/microgpt.fut:458:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127804 = ((double *) mem_138305)[i_137235 * (int64_t) 16 + tmp_127799];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127812 = ((double *) mem_138304)[i_137235 * (int64_t) 16 + tmp_127799];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127823 = ((double *) mem_138303)[i_137235 * (int64_t) 16 + tmp_127799];
                
                ((double *) mem_138384)[i_137225] = lifted_lambda_res_127823;
                ((double *) mem_138385)[i_137225] = lifted_lambda_res_127812;
                ((double *) mem_138386)[i_137225] = lifted_lambda_res_127804;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138369, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138384, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138370, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138385, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138371, i_137235 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138386, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138351, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138369, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138352, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138370, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138353, i_137245 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138371, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138432_cached_sizze_140961 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138432, &mem_138432_cached_sizze_140961, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138438_cached_sizze_140962 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138438, &mem_138438_cached_sizze_140962, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138443_cached_sizze_140963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138443, &mem_138443_cached_sizze_140963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138454_cached_sizze_140964 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138454, &mem_138454_cached_sizze_140964, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138459_cached_sizze_140965 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138459, &mem_138459_cached_sizze_140965, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138470_cached_sizze_140966 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138470, &mem_138470_cached_sizze_140966, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138475_cached_sizze_140967 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138475, &mem_138475_cached_sizze_140967, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138482_cached_sizze_140968 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138482, &mem_138482_cached_sizze_140968, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138489_cached_sizze_140969 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138489, &mem_138489_cached_sizze_140969, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138500_cached_sizze_140970 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138500, &mem_138500_cached_sizze_140970, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138505_cached_sizze_140971 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138505, &mem_138505_cached_sizze_140971, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138516_cached_sizze_140972 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138516, &mem_138516_cached_sizze_140972, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138521_cached_sizze_140973 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138521, &mem_138521_cached_sizze_140973, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137301 = 0; i_137301 < (int64_t) 4; i_137301++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137255 = 0; i_137255 < (int64_t) 16; i_137255++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137251 = 0; i_137251 < (int64_t) 16; i_137251++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_120029;
                double r_120031 = 0.0;
                
                for (int64_t i_120030 = 0; i_120030 < (int64_t) 4; i_120030++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_120032 = ((double *) mem_138353)[i_137301 * (int64_t) 64 + i_137255 * (int64_t) 4 + i_120030];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_120033 = ((double *) mem_138352)[i_137301 * (int64_t) 64 + i_137251 * (int64_t) 4 + i_120030];
                    
                    // futhark/microgpt.fut:162:113-164
                    
                    double zt_res_120034 = zt_lhs_120032 * zt_rhs_120033;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_120035 = r_120031 + zt_res_120034;
                    double r_tmp_140510 = zp_res_120035;
                    
                    r_120031 = r_tmp_140510;
                }
                defunc_0_lifted_lambda_res_120029 = r_120031;
                ((double *) mem_138443)[i_137251] = defunc_0_lifted_lambda_res_120029;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138438, i_137255 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138443, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137263 = 0; i_137263 < (int64_t) 16; i_137263++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137259 = 0; i_137259 < (int64_t) 16; i_137259++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_120050 = ((double *) mem_138438)[i_137263 * (int64_t) 16 + i_137259];
                
                // futhark/microgpt.fut:163:47-78
                
                double zs_res_120051 = zs_lhs_120050 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_120052 = ((double *) mask_mem_138240.mem)[i_137263 * (int64_t) 16 + i_137259];
                
                // futhark/microgpt.fut:163:65-102
                
                double zp_res_120053 = zs_res_120051 + zp_rhs_120052;
                
                ((double *) mem_138459)[i_137259] = zp_res_120053;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138454, i_137263 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138459, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137281 = 0; i_137281 < (int64_t) 16; i_137281++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_127901;
            double redout_137265 = -INFINITY;
            
            for (int64_t i_137266 = 0; i_137266 < (int64_t) 16; i_137266++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127850 = ((double *) mem_138454)[i_137281 * (int64_t) 16 + i_137266];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_120074 = fmax64(lifted_lambda_res_127850, redout_137265);
                double redout_tmp_140514 = max_res_120074;
                
                redout_137265 = redout_tmp_140514;
            }
            defunc_0_reduce_res_127901 = redout_137265;
            // futhark/microgpt.fut:165:67-76
            
            double neg_res_120075 = -defunc_0_reduce_res_127901;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137269 = 0; i_137269 < (int64_t) 16; i_137269++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_120082 = ((double *) mem_138454)[i_137281 * (int64_t) 16 + i_137269];
                
                // futhark/microgpt.fut:165:44-76
                
                double zp_res_120083 = neg_res_120075 + zp_lhs_120082;
                
                // futhark/microgpt.fut:165:37-76
                
                double exp_res_120084 = futrts_exp64(zp_res_120083);
                
                ((double *) mem_138475)[i_137269] = exp_res_120084;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120086;
            double r_120088 = 0.0;
            
            for (int64_t i_120087 = 0; i_120087 < (int64_t) 16; i_120087++) {
                // futhark/microgpt.fut:166:36-46
                
                double lifted_lambda_res_120089 = ((double *) mem_138475)[i_120087];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120090 = r_120088 + lifted_lambda_res_120089;
                double r_tmp_140516 = zp_res_120090;
                
                r_120088 = r_tmp_140516;
            }
            defunc_0_lifted_lambda_res_120086 = r_120088;
            // futhark/microgpt.fut:167:53-64
            
            double zs_res_120091 = 1.0 / defunc_0_lifted_lambda_res_120086;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137273 = 0; i_137273 < (int64_t) 16; i_137273++) {
                // futhark/microgpt.fut:167:37-47
                
                double zt_lhs_120098 = ((double *) mem_138475)[i_137273];
                
                // futhark/microgpt.fut:167:37-64
                
                double zt_res_120099 = zs_res_120091 * zt_lhs_120098;
                
                ((double *) mem_138482)[i_137273] = zt_res_120099;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137277 = 0; i_137277 < (int64_t) 16; i_137277++) {
                // futhark/microgpt.fut:168:4-14
                
                double lifted_lambda_res_120107 = ((double *) mem_138482)[i_137277];
                
                ((double *) mem_138489)[i_137277] = lifted_lambda_res_120107;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138470, i_137281 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138489, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137289 = 0; i_137289 < (int64_t) 16; i_137289++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137285 = 0; i_137285 < (int64_t) 4; i_137285++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_120122;
                double r_120124 = 0.0;
                
                for (int64_t i_120123 = 0; i_120123 < (int64_t) 16; i_120123++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_120125 = ((double *) mem_138470)[i_137289 * (int64_t) 16 + i_120123];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_120126 = ((double *) mem_138351)[i_137301 * (int64_t) 64 + i_120123 * (int64_t) 4 + i_137285];
                    
                    // futhark/microgpt.fut:169:66-111
                    
                    double zt_res_120127 = zt_lhs_120125 * zt_rhs_120126;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_120128 = r_120124 + zt_res_120127;
                    double r_tmp_140521 = zp_res_120128;
                    
                    r_120124 = r_tmp_140521;
                }
                defunc_0_lifted_lambda_res_120122 = r_120124;
                ((double *) mem_138505)[i_137285] = defunc_0_lifted_lambda_res_120122;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138500, i_137289 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138505, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137297 = 0; i_137297 < (int64_t) 16; i_137297++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137293 = 0; i_137293 < (int64_t) 4; i_137293++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_120143 = ((double *) mem_138500)[i_137297 * (int64_t) 4 + i_137293];
                
                ((double *) mem_138521)[i_137293] = lifted_lambda_res_120143;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138516, i_137297 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138521, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_138432, i_137301 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138516, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138537_cached_sizze_140974 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138537, &mem_138537_cached_sizze_140974, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138542_cached_sizze_140975 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138542, &mem_138542_cached_sizze_140975, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137309 = 0; i_137309 < (int64_t) 16; i_137309++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137305 = 0; i_137305 < (int64_t) 16; i_137305++) {
            // futhark/microgpt.fut:171:54-57
            
            int64_t tmp_120155 = sdiv64(i_137305, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool x_120156 = sle64((int64_t) 0, tmp_120155);
            
            // futhark/microgpt.fut:171:44-59
            
            bool y_120157 = slt64(tmp_120155, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-59
            
            bool bounds_check_120158 = x_120156 && y_120157;
            
            // futhark/microgpt.fut:171:44-59
            
            bool index_certs_120159;
            
            if (!bounds_check_120158) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_120155, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-59\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:458:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:171:74-77
            
            int64_t tmp_120160 = smod64(i_137305, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool x_120161 = sle64((int64_t) 0, tmp_120160);
            
            // futhark/microgpt.fut:171:44-79
            
            bool y_120162 = slt64(tmp_120160, (int64_t) 4);
            
            // futhark/microgpt.fut:171:44-79
            
            bool bounds_check_120163 = x_120161 && y_120162;
            
            // futhark/microgpt.fut:171:44-79
            
            bool index_certs_120164;
            
            if (!bounds_check_120163) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_120160, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:171:44-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:171:15-80\n   #6  futhark/microgpt.fut:458:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_120165 = ((double *) mem_138432)[tmp_120155 * (int64_t) 64 + i_137309 * (int64_t) 4 + tmp_120160];
            
            ((double *) mem_138542)[i_137305] = lifted_lambda_res_120165;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138537, i_137309 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138542, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138553_cached_sizze_140976 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138553, &mem_138553_cached_sizze_140976, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138558_cached_sizze_140977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138558, &mem_138558_cached_sizze_140977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137317 = 0; i_137317 < (int64_t) 16; i_137317++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137313 = 0; i_137313 < (int64_t) 16; i_137313++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120180;
            double r_120182 = 0.0;
            
            for (int64_t i_120181 = 0; i_120181 < (int64_t) 16; i_120181++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120183 = ((double *) wout_mem_138232.mem)[i_137313 * (int64_t) 16 + i_120181];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120184 = ((double *) mem_138537)[i_137317 * (int64_t) 16 + i_120181];
                
                // futhark/microgpt.fut:172:67-106
                
                double zt_res_120185 = zt_lhs_120183 * zt_rhs_120184;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120186 = r_120182 + zt_res_120185;
                double r_tmp_140528 = zp_res_120186;
                
                r_120182 = r_tmp_140528;
            }
            defunc_0_lifted_lambda_res_120180 = r_120182;
            ((double *) mem_138558)[i_137313] = defunc_0_lifted_lambda_res_120180;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138553, i_137317 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138558, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138569_cached_sizze_140978 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138569, &mem_138569_cached_sizze_140978, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138574_cached_sizze_140979 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138574, &mem_138574_cached_sizze_140979, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137325 = 0; i_137325 < (int64_t) 16; i_137325++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137321 = 0; i_137321 < (int64_t) 16; i_137321++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_120201 = ((double *) mem_138553)[i_137325 * (int64_t) 16 + i_137321];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_120202 = ((double *) mem_138257)[i_137325 * (int64_t) 16 + i_137321];
            
            // futhark/microgpt.fut:173:46-84
            
            double zp_res_120203 = zp_lhs_120201 + zp_rhs_120202;
            
            ((double *) mem_138574)[i_137321] = zp_res_120203;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138569, i_137325 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138574, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138585_cached_sizze_140980 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138585, &mem_138585_cached_sizze_140980, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138590_cached_sizze_140981 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138590, &mem_138590_cached_sizze_140981, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138597_cached_sizze_140982 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138597, &mem_138597_cached_sizze_140982, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137337 = 0; i_137337 < (int64_t) 16; i_137337++) {
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_120212;
        double r_120214 = 0.0;
        
        for (int64_t i_120213 = 0; i_120213 < (int64_t) 16; i_120213++) {
            // futhark/microgpt.fut:71:46-49
            
            double zt_lhs_120215 = ((double *) mem_138569)[i_137337 * (int64_t) 16 + i_120213];
            
            // futhark/microgpt.fut:174:79-118
            
            double zt_res_120216 = zt_lhs_120215 * zt_lhs_120215;
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_120217 = r_120214 + zt_res_120216;
            double r_tmp_140532 = zp_res_120217;
            
            r_120214 = r_tmp_140532;
        }
        defunc_0_lifted_lambda_res_120212 = r_120214;
        // futhark/microgpt.fut:174:58-136
        
        double zs_res_120218 = defunc_0_lifted_lambda_res_120212 / 16.0;
        
        // futhark/microgpt.fut:175:24-55
        
        double zp_res_120219 = 1.0e-5 + zs_res_120218;
        
        // futhark/microgpt.fut:175:16-55
        
        double sqrt_res_120220 = futrts_sqrt64(zp_res_120219);
        
        // futhark/microgpt.fut:176:60-71
        
        double zs_res_120221 = 1.0 / sqrt_res_120220;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137329 = 0; i_137329 < (int64_t) 16; i_137329++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_120228 = ((double *) mem_138569)[i_137337 * (int64_t) 16 + i_137329];
            
            // futhark/microgpt.fut:176:37-71
            
            double zt_res_120229 = zs_res_120221 * zt_lhs_120228;
            
            ((double *) mem_138590)[i_137329] = zt_res_120229;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137333 = 0; i_137333 < (int64_t) 16; i_137333++) {
            // futhark/microgpt.fut:177:4-14
            
            double lifted_lambda_res_120237 = ((double *) mem_138590)[i_137333];
            
            ((double *) mem_138597)[i_137333] = lifted_lambda_res_120237;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138585, i_137337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138597, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138608_cached_sizze_140983 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138608, &mem_138608_cached_sizze_140983, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138613_cached_sizze_140984 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138613, &mem_138613_cached_sizze_140984, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137345 = 0; i_137345 < (int64_t) 16; i_137345++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137341 = 0; i_137341 < (int64_t) 64; i_137341++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120253;
            double r_120255 = 0.0;
            
            for (int64_t i_120254 = 0; i_120254 < (int64_t) 16; i_120254++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120256 = ((double *) wup_mem_138236.mem)[i_137341 * (int64_t) 16 + i_120254];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120257 = ((double *) mem_138585)[i_137345 * (int64_t) 16 + i_120254];
                
                // futhark/microgpt.fut:178:67-106
                
                double zt_res_120258 = zt_lhs_120256 * zt_rhs_120257;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120259 = r_120255 + zt_res_120258;
                double r_tmp_140537 = zp_res_120259;
                
                r_120255 = r_tmp_140537;
            }
            defunc_0_lifted_lambda_res_120253 = r_120255;
            ((double *) mem_138613)[i_137341] = defunc_0_lifted_lambda_res_120253;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138608, i_137345 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138613, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138624_cached_sizze_140985 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138624, &mem_138624_cached_sizze_140985, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138629_cached_sizze_140986 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138629, &mem_138629_cached_sizze_140986, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137353 = 0; i_137353 < (int64_t) 16; i_137353++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137349 = 0; i_137349 < (int64_t) 64; i_137349++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_120274 = ((double *) mem_138608)[i_137353 * (int64_t) 64 + i_137349];
            
            // futhark/microgpt.fut:179:45-73
            
            double max_res_120275 = fmax64(0.0, max_arg0_120274);
            
            ((double *) mem_138629)[i_137349] = max_res_120275;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138624, i_137353 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138629, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138640_cached_sizze_140987 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138640, &mem_138640_cached_sizze_140987, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138645_cached_sizze_140988 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138645, &mem_138645_cached_sizze_140988, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137361 = 0; i_137361 < (int64_t) 16; i_137361++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137357 = 0; i_137357 < (int64_t) 16; i_137357++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120290;
            double r_120292 = 0.0;
            
            for (int64_t i_120291 = 0; i_120291 < (int64_t) 64; i_120291++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120293 = ((double *) wdown_mem_138230.mem)[i_137357 * (int64_t) 64 + i_120291];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120294 = ((double *) mem_138624)[i_137361 * (int64_t) 64 + i_120291];
                
                // futhark/microgpt.fut:180:67-108
                
                double zt_res_120295 = zt_lhs_120293 * zt_rhs_120294;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120296 = r_120292 + zt_res_120295;
                double r_tmp_140542 = zp_res_120296;
                
                r_120292 = r_tmp_140542;
            }
            defunc_0_lifted_lambda_res_120290 = r_120292;
            ((double *) mem_138645)[i_137357] = defunc_0_lifted_lambda_res_120290;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138640, i_137361 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138645, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138656_cached_sizze_140989 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138656, &mem_138656_cached_sizze_140989, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138661_cached_sizze_140990 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138661, &mem_138661_cached_sizze_140990, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137369 = 0; i_137369 < (int64_t) 16; i_137369++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137365 = 0; i_137365 < (int64_t) 16; i_137365++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_120311 = ((double *) mem_138640)[i_137369 * (int64_t) 16 + i_137365];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_120312 = ((double *) mem_138569)[i_137369 * (int64_t) 16 + i_137365];
            
            // futhark/microgpt.fut:181:46-85
            
            double zp_res_120313 = zp_lhs_120311 + zp_rhs_120312;
            
            ((double *) mem_138661)[i_137365] = zp_res_120313;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138656, i_137369 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138661, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138672_cached_sizze_140991 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138672, &mem_138672_cached_sizze_140991, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138677_cached_sizze_140992 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138677, &mem_138677_cached_sizze_140992, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137377 = 0; i_137377 < (int64_t) 16; i_137377++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137373 = 0; i_137373 < (int64_t) 27; i_137373++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_120329;
            double r_120331 = 0.0;
            
            for (int64_t i_120330 = 0; i_120330 < (int64_t) 16; i_120330++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_120332 = ((double *) wvoc_mem_138238.mem)[i_137373 * (int64_t) 16 + i_120330];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_120333 = ((double *) mem_138656)[i_137377 * (int64_t) 16 + i_120330];
                
                // futhark/microgpt.fut:182:67-107
                
                double zt_res_120334 = zt_lhs_120332 * zt_rhs_120333;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_120335 = r_120331 + zt_res_120334;
                double r_tmp_140547 = zp_res_120335;
                
                r_120331 = r_tmp_140547;
            }
            defunc_0_lifted_lambda_res_120329 = r_120331;
            ((double *) mem_138677)[i_137373] = defunc_0_lifted_lambda_res_120329;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138672, i_137377 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138677, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_138688, (int64_t) 3456, "mem_138688")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138693_cached_sizze_140993 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138693, &mem_138693_cached_sizze_140993, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_137385 = 0; i_137385 < (int64_t) 16; i_137385++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137381 = 0; i_137381 < (int64_t) 27; i_137381++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_120350 = ((double *) mem_138672)[i_137385 * (int64_t) 27 + i_137381];
            
            ((double *) mem_138693)[i_137381] = lifted_lambda_res_120350;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_138688.mem, i_137385 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138693, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_140478, &mem_138688, "mem_138688") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140937, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_138241);
        free(mem_138246);
        free(mem_138257);
        free(mem_138262);
        free(mem_138269);
        free(mem_138280);
        free(mem_138285);
        free(mem_138292);
        free(mem_138303);
        free(mem_138304);
        free(mem_138305);
        free(mem_138318);
        free(mem_138319);
        free(mem_138320);
        free(mem_138351);
        free(mem_138352);
        free(mem_138353);
        free(mem_138369);
        free(mem_138370);
        free(mem_138371);
        free(mem_138384);
        free(mem_138385);
        free(mem_138386);
        free(mem_138432);
        free(mem_138438);
        free(mem_138443);
        free(mem_138454);
        free(mem_138459);
        free(mem_138470);
        free(mem_138475);
        free(mem_138482);
        free(mem_138489);
        free(mem_138500);
        free(mem_138505);
        free(mem_138516);
        free(mem_138521);
        free(mem_138537);
        free(mem_138542);
        free(mem_138553);
        free(mem_138558);
        free(mem_138569);
        free(mem_138574);
        free(mem_138585);
        free(mem_138590);
        free(mem_138597);
        free(mem_138608);
        free(mem_138613);
        free(mem_138624);
        free(mem_138629);
        free(mem_138640);
        free(mem_138645);
        free(mem_138656);
        free(mem_138661);
        free(mem_138672);
        free(mem_138677);
        free(mem_138693);
        if (memblock_unref(ctx, &mem_138688, "mem_138688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_140994, struct memblock *mem_out_p_140995, struct memblock *mem_out_p_140996, struct memblock *mem_out_p_140997, struct memblock *mem_out_p_140998, struct memblock *mem_out_p_140999, struct memblock *mem_out_p_141000, struct memblock *mem_out_p_141001, struct memblock *mem_out_p_141002, struct memblock wte_mem_138230, struct memblock wpe_mem_138231, struct memblock wqry_mem_138232, struct memblock wkey_mem_138233, struct memblock wval_mem_138234, struct memblock wout_mem_138235, struct memblock wup_mem_138236, struct memblock wdown_mem_138237, struct memblock wvoc_mem_138238)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    if (memblock_set(ctx, &mem_out_140478, &wdown_mem_138237, "wdown_mem_138237") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140479, &wkey_mem_138233, "wkey_mem_138233") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140480, &wout_mem_138235, "wout_mem_138235") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140481, &wpe_mem_138231, "wpe_mem_138231") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140482, &wqry_mem_138232, "wqry_mem_138232") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140483, &wte_mem_138230, "wte_mem_138230") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140484, &wup_mem_138236, "wup_mem_138236") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140485, &wval_mem_138234, "wval_mem_138234") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140486, &wvoc_mem_138238, "wvoc_mem_138238") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140994, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140995, &mem_out_140479, "mem_out_140479") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140996, &mem_out_140480, "mem_out_140480") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140997, &mem_out_140481, "mem_out_140481") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140998, &mem_out_140482, "mem_out_140482") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_140999, &mem_out_140483, "mem_out_140483") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141000, &mem_out_140484, "mem_out_140484") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141001, &mem_out_140485, "mem_out_140485") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141002, &mem_out_140486, "mem_out_140486") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_140486, "mem_out_140486") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140485, "mem_out_140485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140484, "mem_out_140484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140483, "mem_out_140483") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140482, "mem_out_140482") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140481, "mem_out_140481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140480, "mem_out_140480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140479, "mem_out_140479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_141003, struct memblock *mem_out_p_141004, struct memblock *mem_out_p_141005, struct memblock *mem_out_p_141006, struct memblock *mem_out_p_141007, struct memblock *mem_out_p_141008, struct memblock *mem_out_p_141009, struct memblock *mem_out_p_141010, struct memblock *mem_out_p_141011, struct memblock *mem_out_p_141012, struct memblock *mem_out_p_141013, struct memblock *mem_out_p_141014, struct memblock *mem_out_p_141015, struct memblock *mem_out_p_141016, struct memblock *mem_out_p_141017, struct memblock *mem_out_p_141018, struct memblock *mem_out_p_141019, struct memblock *mem_out_p_141020, struct memblock *mem_out_p_141021, struct memblock *mem_out_p_141022, struct memblock *mem_out_p_141023, struct memblock *mem_out_p_141024, struct memblock *mem_out_p_141025, struct memblock *mem_out_p_141026, struct memblock *mem_out_p_141027, struct memblock *mem_out_p_141028, struct memblock *mem_out_p_141029, struct memblock wdown_mem_138230, struct memblock wkey_mem_138231, struct memblock wout_mem_138232, struct memblock wpe_mem_138233, struct memblock wqry_mem_138234, struct memblock wte_mem_138235, struct memblock wup_mem_138236, struct memblock wval_mem_138237, struct memblock wvoc_mem_138238, struct memblock wdown_mem_138239, struct memblock wkey_mem_138240, struct memblock wout_mem_138241, struct memblock wpe_mem_138242, struct memblock wqry_mem_138243, struct memblock wte_mem_138244, struct memblock wup_mem_138245, struct memblock wval_mem_138246, struct memblock wvoc_mem_138247, struct memblock wdown_mem_138248, struct memblock wkey_mem_138249, struct memblock wout_mem_138250, struct memblock wpe_mem_138251, struct memblock wqry_mem_138252, struct memblock wte_mem_138253, struct memblock wup_mem_138254, struct memblock wval_mem_138255, struct memblock wvoc_mem_138256, struct memblock masks_mem_138257, struct memblock dls_mem_138258, struct memblock seqs_mem_138259)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_138368_cached_sizze_141030 = 0;
    unsigned char *mem_138368 = NULL;
    int64_t mem_138369_cached_sizze_141031 = 0;
    unsigned char *mem_138369 = NULL;
    int64_t mem_138378_cached_sizze_141032 = 0;
    unsigned char *mem_138378 = NULL;
    int64_t mem_138385_cached_sizze_141033 = 0;
    unsigned char *mem_138385 = NULL;
    int64_t mem_138400_cached_sizze_141034 = 0;
    unsigned char *mem_138400 = NULL;
    int64_t mem_138405_cached_sizze_141035 = 0;
    unsigned char *mem_138405 = NULL;
    int64_t mem_138416_cached_sizze_141036 = 0;
    unsigned char *mem_138416 = NULL;
    int64_t mem_138417_cached_sizze_141037 = 0;
    unsigned char *mem_138417 = NULL;
    int64_t mem_138425_cached_sizze_141038 = 0;
    unsigned char *mem_138425 = NULL;
    int64_t mem_138439_cached_sizze_141039 = 0;
    unsigned char *mem_138439 = NULL;
    int64_t mem_138440_cached_sizze_141040 = 0;
    unsigned char *mem_138440 = NULL;
    int64_t mem_138448_cached_sizze_141041 = 0;
    unsigned char *mem_138448 = NULL;
    int64_t mem_138462_cached_sizze_141042 = 0;
    unsigned char *mem_138462 = NULL;
    int64_t mem_138463_cached_sizze_141043 = 0;
    unsigned char *mem_138463 = NULL;
    int64_t mem_138464_cached_sizze_141044 = 0;
    unsigned char *mem_138464 = NULL;
    int64_t mem_138477_cached_sizze_141045 = 0;
    unsigned char *mem_138477 = NULL;
    int64_t mem_138478_cached_sizze_141046 = 0;
    unsigned char *mem_138478 = NULL;
    int64_t mem_138479_cached_sizze_141047 = 0;
    unsigned char *mem_138479 = NULL;
    int64_t mem_138510_cached_sizze_141048 = 0;
    unsigned char *mem_138510 = NULL;
    int64_t mem_138511_cached_sizze_141049 = 0;
    unsigned char *mem_138511 = NULL;
    int64_t mem_138512_cached_sizze_141050 = 0;
    unsigned char *mem_138512 = NULL;
    int64_t mem_138528_cached_sizze_141051 = 0;
    unsigned char *mem_138528 = NULL;
    int64_t mem_138529_cached_sizze_141052 = 0;
    unsigned char *mem_138529 = NULL;
    int64_t mem_138530_cached_sizze_141053 = 0;
    unsigned char *mem_138530 = NULL;
    int64_t mem_138543_cached_sizze_141054 = 0;
    unsigned char *mem_138543 = NULL;
    int64_t mem_138544_cached_sizze_141055 = 0;
    unsigned char *mem_138544 = NULL;
    int64_t mem_138545_cached_sizze_141056 = 0;
    unsigned char *mem_138545 = NULL;
    int64_t mem_138591_cached_sizze_141057 = 0;
    unsigned char *mem_138591 = NULL;
    int64_t mem_138592_cached_sizze_141058 = 0;
    unsigned char *mem_138592 = NULL;
    int64_t mem_138593_cached_sizze_141059 = 0;
    unsigned char *mem_138593 = NULL;
    int64_t mem_138594_cached_sizze_141060 = 0;
    unsigned char *mem_138594 = NULL;
    int64_t mem_138615_cached_sizze_141061 = 0;
    unsigned char *mem_138615 = NULL;
    int64_t mem_138616_cached_sizze_141062 = 0;
    unsigned char *mem_138616 = NULL;
    int64_t mem_138617_cached_sizze_141063 = 0;
    unsigned char *mem_138617 = NULL;
    int64_t mem_138618_cached_sizze_141064 = 0;
    unsigned char *mem_138618 = NULL;
    int64_t mem_138635_cached_sizze_141065 = 0;
    unsigned char *mem_138635 = NULL;
    int64_t mem_138636_cached_sizze_141066 = 0;
    unsigned char *mem_138636 = NULL;
    int64_t mem_138637_cached_sizze_141067 = 0;
    unsigned char *mem_138637 = NULL;
    int64_t mem_138638_cached_sizze_141068 = 0;
    unsigned char *mem_138638 = NULL;
    int64_t mem_138679_cached_sizze_141069 = 0;
    unsigned char *mem_138679 = NULL;
    int64_t mem_138684_cached_sizze_141070 = 0;
    unsigned char *mem_138684 = NULL;
    int64_t mem_138695_cached_sizze_141071 = 0;
    unsigned char *mem_138695 = NULL;
    int64_t mem_138700_cached_sizze_141072 = 0;
    unsigned char *mem_138700 = NULL;
    int64_t mem_138707_cached_sizze_141073 = 0;
    unsigned char *mem_138707 = NULL;
    int64_t mem_138718_cached_sizze_141074 = 0;
    unsigned char *mem_138718 = NULL;
    int64_t mem_138723_cached_sizze_141075 = 0;
    unsigned char *mem_138723 = NULL;
    int64_t mem_138754_cached_sizze_141076 = 0;
    unsigned char *mem_138754 = NULL;
    int64_t mem_138759_cached_sizze_141077 = 0;
    unsigned char *mem_138759 = NULL;
    int64_t mem_138770_cached_sizze_141078 = 0;
    unsigned char *mem_138770 = NULL;
    int64_t mem_138775_cached_sizze_141079 = 0;
    unsigned char *mem_138775 = NULL;
    int64_t mem_138786_cached_sizze_141080 = 0;
    unsigned char *mem_138786 = NULL;
    int64_t mem_138791_cached_sizze_141081 = 0;
    unsigned char *mem_138791 = NULL;
    int64_t mem_138802_cached_sizze_141082 = 0;
    unsigned char *mem_138802 = NULL;
    int64_t mem_138803_cached_sizze_141083 = 0;
    unsigned char *mem_138803 = NULL;
    int64_t mem_138811_cached_sizze_141084 = 0;
    unsigned char *mem_138811 = NULL;
    int64_t mem_138825_cached_sizze_141085 = 0;
    unsigned char *mem_138825 = NULL;
    int64_t mem_138830_cached_sizze_141086 = 0;
    unsigned char *mem_138830 = NULL;
    int64_t mem_138841_cached_sizze_141087 = 0;
    unsigned char *mem_138841 = NULL;
    int64_t mem_138846_cached_sizze_141088 = 0;
    unsigned char *mem_138846 = NULL;
    int64_t mem_138857_cached_sizze_141089 = 0;
    unsigned char *mem_138857 = NULL;
    int64_t mem_138862_cached_sizze_141090 = 0;
    unsigned char *mem_138862 = NULL;
    int64_t mem_138873_cached_sizze_141091 = 0;
    unsigned char *mem_138873 = NULL;
    int64_t mem_138878_cached_sizze_141092 = 0;
    unsigned char *mem_138878 = NULL;
    int64_t mem_138889_cached_sizze_141093 = 0;
    unsigned char *mem_138889 = NULL;
    int64_t mem_138894_cached_sizze_141094 = 0;
    unsigned char *mem_138894 = NULL;
    int64_t mem_138905_cached_sizze_141095 = 0;
    unsigned char *mem_138905 = NULL;
    int64_t mem_138906_cached_sizze_141096 = 0;
    unsigned char *mem_138906 = NULL;
    int64_t mem_138907_cached_sizze_141097 = 0;
    unsigned char *mem_138907 = NULL;
    int64_t mem_138935_cached_sizze_141098 = 0;
    unsigned char *mem_138935 = NULL;
    int64_t mem_138941_cached_sizze_141099 = 0;
    unsigned char *mem_138941 = NULL;
    int64_t mem_138946_cached_sizze_141100 = 0;
    unsigned char *mem_138946 = NULL;
    int64_t mem_138962_cached_sizze_141101 = 0;
    unsigned char *mem_138962 = NULL;
    int64_t mem_138967_cached_sizze_141102 = 0;
    unsigned char *mem_138967 = NULL;
    int64_t mem_138978_cached_sizze_141103 = 0;
    unsigned char *mem_138978 = NULL;
    int64_t mem_138983_cached_sizze_141104 = 0;
    unsigned char *mem_138983 = NULL;
    int64_t mem_138987_cached_sizze_141105 = 0;
    unsigned char *mem_138987 = NULL;
    int64_t mem_139001_cached_sizze_141106 = 0;
    unsigned char *mem_139001 = NULL;
    int64_t mem_139007_cached_sizze_141107 = 0;
    unsigned char *mem_139007 = NULL;
    int64_t mem_139012_cached_sizze_141108 = 0;
    unsigned char *mem_139012 = NULL;
    int64_t mem_139016_cached_sizze_141109 = 0;
    unsigned char *mem_139016 = NULL;
    int64_t mem_139035_cached_sizze_141110 = 0;
    unsigned char *mem_139035 = NULL;
    int64_t mem_139040_cached_sizze_141111 = 0;
    unsigned char *mem_139040 = NULL;
    int64_t mem_139051_cached_sizze_141112 = 0;
    unsigned char *mem_139051 = NULL;
    int64_t mem_139056_cached_sizze_141113 = 0;
    unsigned char *mem_139056 = NULL;
    int64_t mem_139067_cached_sizze_141114 = 0;
    unsigned char *mem_139067 = NULL;
    int64_t mem_139072_cached_sizze_141115 = 0;
    unsigned char *mem_139072 = NULL;
    int64_t mem_139083_cached_sizze_141116 = 0;
    unsigned char *mem_139083 = NULL;
    int64_t mem_139088_cached_sizze_141117 = 0;
    unsigned char *mem_139088 = NULL;
    int64_t mem_139099_cached_sizze_141118 = 0;
    unsigned char *mem_139099 = NULL;
    int64_t mem_139100_cached_sizze_141119 = 0;
    unsigned char *mem_139100 = NULL;
    int64_t mem_139109_cached_sizze_141120 = 0;
    unsigned char *mem_139109 = NULL;
    int64_t mem_139110_cached_sizze_141121 = 0;
    unsigned char *mem_139110 = NULL;
    int64_t mem_139131_cached_sizze_141122 = 0;
    unsigned char *mem_139131 = NULL;
    int64_t mem_139136_cached_sizze_141123 = 0;
    unsigned char *mem_139136 = NULL;
    int64_t mem_139147_cached_sizze_141124 = 0;
    unsigned char *mem_139147 = NULL;
    int64_t mem_139152_cached_sizze_141125 = 0;
    unsigned char *mem_139152 = NULL;
    int64_t mem_139163_cached_sizze_141126 = 0;
    unsigned char *mem_139163 = NULL;
    int64_t mem_139164_cached_sizze_141127 = 0;
    unsigned char *mem_139164 = NULL;
    int64_t mem_139177_cached_sizze_141128 = 0;
    unsigned char *mem_139177 = NULL;
    int64_t mem_139184_cached_sizze_141129 = 0;
    unsigned char *mem_139184 = NULL;
    int64_t mem_139189_cached_sizze_141130 = 0;
    unsigned char *mem_139189 = NULL;
    int64_t mem_139200_cached_sizze_141131 = 0;
    unsigned char *mem_139200 = NULL;
    int64_t mem_139205_cached_sizze_141132 = 0;
    unsigned char *mem_139205 = NULL;
    int64_t mem_139216_cached_sizze_141133 = 0;
    unsigned char *mem_139216 = NULL;
    int64_t mem_139217_cached_sizze_141134 = 0;
    unsigned char *mem_139217 = NULL;
    int64_t mem_139226_cached_sizze_141135 = 0;
    unsigned char *mem_139226 = NULL;
    int64_t mem_139227_cached_sizze_141136 = 0;
    unsigned char *mem_139227 = NULL;
    int64_t mem_139248_cached_sizze_141137 = 0;
    unsigned char *mem_139248 = NULL;
    int64_t mem_139249_cached_sizze_141138 = 0;
    unsigned char *mem_139249 = NULL;
    int64_t mem_139250_cached_sizze_141139 = 0;
    unsigned char *mem_139250 = NULL;
    int64_t mem_139251_cached_sizze_141140 = 0;
    unsigned char *mem_139251 = NULL;
    int64_t mem_139272_cached_sizze_141141 = 0;
    unsigned char *mem_139272 = NULL;
    int64_t mem_139273_cached_sizze_141142 = 0;
    unsigned char *mem_139273 = NULL;
    int64_t mem_139274_cached_sizze_141143 = 0;
    unsigned char *mem_139274 = NULL;
    int64_t mem_139275_cached_sizze_141144 = 0;
    unsigned char *mem_139275 = NULL;
    int64_t mem_139292_cached_sizze_141145 = 0;
    unsigned char *mem_139292 = NULL;
    int64_t mem_139299_cached_sizze_141146 = 0;
    unsigned char *mem_139299 = NULL;
    int64_t mem_139300_cached_sizze_141147 = 0;
    unsigned char *mem_139300 = NULL;
    int64_t mem_139301_cached_sizze_141148 = 0;
    unsigned char *mem_139301 = NULL;
    int64_t mem_139356_cached_sizze_141149 = 0;
    unsigned char *mem_139356 = NULL;
    int64_t mem_139357_cached_sizze_141150 = 0;
    unsigned char *mem_139357 = NULL;
    int64_t mem_139358_cached_sizze_141151 = 0;
    unsigned char *mem_139358 = NULL;
    int64_t mem_139359_cached_sizze_141152 = 0;
    unsigned char *mem_139359 = NULL;
    int64_t mem_139360_cached_sizze_141153 = 0;
    unsigned char *mem_139360 = NULL;
    int64_t mem_139361_cached_sizze_141154 = 0;
    unsigned char *mem_139361 = NULL;
    int64_t mem_139362_cached_sizze_141155 = 0;
    unsigned char *mem_139362 = NULL;
    int64_t mem_139363_cached_sizze_141156 = 0;
    unsigned char *mem_139363 = NULL;
    int64_t mem_139364_cached_sizze_141157 = 0;
    unsigned char *mem_139364 = NULL;
    int64_t mem_139404_cached_sizze_141158 = 0;
    unsigned char *mem_139404 = NULL;
    int64_t mem_139405_cached_sizze_141159 = 0;
    unsigned char *mem_139405 = NULL;
    int64_t mem_139406_cached_sizze_141160 = 0;
    unsigned char *mem_139406 = NULL;
    int64_t mem_139407_cached_sizze_141161 = 0;
    unsigned char *mem_139407 = NULL;
    int64_t mem_139408_cached_sizze_141162 = 0;
    unsigned char *mem_139408 = NULL;
    int64_t mem_139409_cached_sizze_141163 = 0;
    unsigned char *mem_139409 = NULL;
    int64_t mem_139410_cached_sizze_141164 = 0;
    unsigned char *mem_139410 = NULL;
    int64_t mem_139411_cached_sizze_141165 = 0;
    unsigned char *mem_139411 = NULL;
    int64_t mem_139412_cached_sizze_141166 = 0;
    unsigned char *mem_139412 = NULL;
    int64_t mem_139443_cached_sizze_141167 = 0;
    unsigned char *mem_139443 = NULL;
    int64_t mem_139444_cached_sizze_141168 = 0;
    unsigned char *mem_139444 = NULL;
    int64_t mem_139457_cached_sizze_141169 = 0;
    unsigned char *mem_139457 = NULL;
    int64_t mem_139464_cached_sizze_141170 = 0;
    unsigned char *mem_139464 = NULL;
    int64_t mem_139540_cached_sizze_141171 = 0;
    unsigned char *mem_139540 = NULL;
    int64_t mem_139541_cached_sizze_141172 = 0;
    unsigned char *mem_139541 = NULL;
    int64_t mem_139542_cached_sizze_141173 = 0;
    unsigned char *mem_139542 = NULL;
    int64_t mem_139558_cached_sizze_141174 = 0;
    unsigned char *mem_139558 = NULL;
    int64_t mem_139559_cached_sizze_141175 = 0;
    unsigned char *mem_139559 = NULL;
    int64_t mem_139560_cached_sizze_141176 = 0;
    unsigned char *mem_139560 = NULL;
    int64_t mem_139573_cached_sizze_141177 = 0;
    unsigned char *mem_139573 = NULL;
    int64_t mem_139580_cached_sizze_141178 = 0;
    unsigned char *mem_139580 = NULL;
    int64_t mem_139581_cached_sizze_141179 = 0;
    unsigned char *mem_139581 = NULL;
    int64_t mem_139621_cached_sizze_141180 = 0;
    unsigned char *mem_139621 = NULL;
    int64_t mem_139622_cached_sizze_141181 = 0;
    unsigned char *mem_139622 = NULL;
    int64_t mem_139623_cached_sizze_141182 = 0;
    unsigned char *mem_139623 = NULL;
    int64_t mem_139624_cached_sizze_141183 = 0;
    unsigned char *mem_139624 = NULL;
    int64_t mem_139641_cached_sizze_141184 = 0;
    unsigned char *mem_139641 = NULL;
    int64_t mem_139642_cached_sizze_141185 = 0;
    unsigned char *mem_139642 = NULL;
    int64_t mem_139643_cached_sizze_141186 = 0;
    unsigned char *mem_139643 = NULL;
    int64_t mem_139644_cached_sizze_141187 = 0;
    unsigned char *mem_139644 = NULL;
    int64_t mem_139685_cached_sizze_141188 = 0;
    unsigned char *mem_139685 = NULL;
    int64_t mem_139686_cached_sizze_141189 = 0;
    unsigned char *mem_139686 = NULL;
    int64_t mem_139697_cached_sizze_141190 = 0;
    unsigned char *mem_139697 = NULL;
    int64_t mem_139698_cached_sizze_141191 = 0;
    unsigned char *mem_139698 = NULL;
    int64_t mem_139707_cached_sizze_141192 = 0;
    unsigned char *mem_139707 = NULL;
    int64_t mem_139708_cached_sizze_141193 = 0;
    unsigned char *mem_139708 = NULL;
    int64_t mem_139739_cached_sizze_141194 = 0;
    unsigned char *mem_139739 = NULL;
    int64_t mem_139740_cached_sizze_141195 = 0;
    unsigned char *mem_139740 = NULL;
    int64_t mem_139749_cached_sizze_141196 = 0;
    unsigned char *mem_139749 = NULL;
    int64_t mem_139750_cached_sizze_141197 = 0;
    unsigned char *mem_139750 = NULL;
    int64_t mem_139771_cached_sizze_141198 = 0;
    unsigned char *mem_139771 = NULL;
    int64_t mem_139772_cached_sizze_141199 = 0;
    unsigned char *mem_139772 = NULL;
    int64_t mem_139783_cached_sizze_141200 = 0;
    unsigned char *mem_139783 = NULL;
    int64_t mem_139784_cached_sizze_141201 = 0;
    unsigned char *mem_139784 = NULL;
    int64_t mem_139793_cached_sizze_141202 = 0;
    unsigned char *mem_139793 = NULL;
    int64_t mem_139794_cached_sizze_141203 = 0;
    unsigned char *mem_139794 = NULL;
    int64_t mem_139825_cached_sizze_141204 = 0;
    unsigned char *mem_139825 = NULL;
    int64_t mem_139826_cached_sizze_141205 = 0;
    unsigned char *mem_139826 = NULL;
    int64_t mem_139837_cached_sizze_141206 = 0;
    unsigned char *mem_139837 = NULL;
    int64_t mem_139838_cached_sizze_141207 = 0;
    unsigned char *mem_139838 = NULL;
    int64_t mem_139847_cached_sizze_141208 = 0;
    unsigned char *mem_139847 = NULL;
    int64_t mem_139848_cached_sizze_141209 = 0;
    unsigned char *mem_139848 = NULL;
    int64_t mem_139879_cached_sizze_141210 = 0;
    unsigned char *mem_139879 = NULL;
    int64_t mem_139880_cached_sizze_141211 = 0;
    unsigned char *mem_139880 = NULL;
    int64_t mem_139891_cached_sizze_141212 = 0;
    unsigned char *mem_139891 = NULL;
    int64_t mem_139892_cached_sizze_141213 = 0;
    unsigned char *mem_139892 = NULL;
    int64_t mem_139901_cached_sizze_141214 = 0;
    unsigned char *mem_139901 = NULL;
    int64_t mem_139902_cached_sizze_141215 = 0;
    unsigned char *mem_139902 = NULL;
    int64_t mem_139933_cached_sizze_141216 = 0;
    unsigned char *mem_139933 = NULL;
    int64_t mem_139934_cached_sizze_141217 = 0;
    unsigned char *mem_139934 = NULL;
    int64_t mem_139935_cached_sizze_141218 = 0;
    unsigned char *mem_139935 = NULL;
    int64_t mem_139948_cached_sizze_141219 = 0;
    unsigned char *mem_139948 = NULL;
    int64_t mem_139949_cached_sizze_141220 = 0;
    unsigned char *mem_139949 = NULL;
    int64_t mem_139950_cached_sizze_141221 = 0;
    unsigned char *mem_139950 = NULL;
    int64_t mem_139981_cached_sizze_141222 = 0;
    unsigned char *mem_139981 = NULL;
    int64_t mem_139982_cached_sizze_141223 = 0;
    unsigned char *mem_139982 = NULL;
    int64_t mem_139983_cached_sizze_141224 = 0;
    unsigned char *mem_139983 = NULL;
    int64_t mem_139984_cached_sizze_141225 = 0;
    unsigned char *mem_139984 = NULL;
    int64_t mem_140001_cached_sizze_141226 = 0;
    unsigned char *mem_140001 = NULL;
    int64_t mem_140002_cached_sizze_141227 = 0;
    unsigned char *mem_140002 = NULL;
    int64_t mem_140003_cached_sizze_141228 = 0;
    unsigned char *mem_140003 = NULL;
    int64_t mem_140004_cached_sizze_141229 = 0;
    unsigned char *mem_140004 = NULL;
    int64_t mem_140045_cached_sizze_141230 = 0;
    unsigned char *mem_140045 = NULL;
    int64_t mem_140046_cached_sizze_141231 = 0;
    unsigned char *mem_140046 = NULL;
    int64_t mem_140059_cached_sizze_141232 = 0;
    unsigned char *mem_140059 = NULL;
    int64_t mem_140066_cached_sizze_141233 = 0;
    unsigned char *mem_140066 = NULL;
    int64_t mem_140071_cached_sizze_141234 = 0;
    unsigned char *mem_140071 = NULL;
    int64_t mem_140082_cached_sizze_141235 = 0;
    unsigned char *mem_140082 = NULL;
    int64_t mem_140083_cached_sizze_141236 = 0;
    unsigned char *mem_140083 = NULL;
    int64_t mem_140096_cached_sizze_141237 = 0;
    unsigned char *mem_140096 = NULL;
    int64_t mem_140103_cached_sizze_141238 = 0;
    unsigned char *mem_140103 = NULL;
    int64_t mem_140108_cached_sizze_141239 = 0;
    unsigned char *mem_140108 = NULL;
    int64_t mem_140119_cached_sizze_141240 = 0;
    unsigned char *mem_140119 = NULL;
    int64_t mem_140120_cached_sizze_141241 = 0;
    unsigned char *mem_140120 = NULL;
    int64_t mem_140129_cached_sizze_141242 = 0;
    unsigned char *mem_140129 = NULL;
    int64_t mem_140130_cached_sizze_141243 = 0;
    unsigned char *mem_140130 = NULL;
    int64_t mem_140151_cached_sizze_141244 = 0;
    unsigned char *mem_140151 = NULL;
    int64_t mem_140156_cached_sizze_141245 = 0;
    unsigned char *mem_140156 = NULL;
    int64_t mem_140167_cached_sizze_141246 = 0;
    unsigned char *mem_140167 = NULL;
    int64_t mem_140168_cached_sizze_141247 = 0;
    unsigned char *mem_140168 = NULL;
    int64_t mem_140177_cached_sizze_141248 = 0;
    unsigned char *mem_140177 = NULL;
    int64_t mem_140178_cached_sizze_141249 = 0;
    unsigned char *mem_140178 = NULL;
    struct memblock mem_param_tmp_140531;
    
    mem_param_tmp_140531.references = NULL;
    
    struct memblock mem_param_tmp_140530;
    
    mem_param_tmp_140530.references = NULL;
    
    struct memblock mem_param_tmp_140529;
    
    mem_param_tmp_140529.references = NULL;
    
    struct memblock mem_param_tmp_140528;
    
    mem_param_tmp_140528.references = NULL;
    
    struct memblock mem_param_tmp_140527;
    
    mem_param_tmp_140527.references = NULL;
    
    struct memblock mem_param_tmp_140526;
    
    mem_param_tmp_140526.references = NULL;
    
    struct memblock mem_param_tmp_140525;
    
    mem_param_tmp_140525.references = NULL;
    
    struct memblock mem_param_tmp_140524;
    
    mem_param_tmp_140524.references = NULL;
    
    struct memblock mem_param_tmp_140523;
    
    mem_param_tmp_140523.references = NULL;
    
    struct memblock mem_param_tmp_140522;
    
    mem_param_tmp_140522.references = NULL;
    
    struct memblock mem_param_tmp_140521;
    
    mem_param_tmp_140521.references = NULL;
    
    struct memblock mem_param_tmp_140520;
    
    mem_param_tmp_140520.references = NULL;
    
    struct memblock mem_param_tmp_140519;
    
    mem_param_tmp_140519.references = NULL;
    
    struct memblock mem_param_tmp_140518;
    
    mem_param_tmp_140518.references = NULL;
    
    struct memblock mem_param_tmp_140517;
    
    mem_param_tmp_140517.references = NULL;
    
    struct memblock mem_param_tmp_140516;
    
    mem_param_tmp_140516.references = NULL;
    
    struct memblock mem_param_tmp_140515;
    
    mem_param_tmp_140515.references = NULL;
    
    struct memblock mem_param_tmp_140514;
    
    mem_param_tmp_140514.references = NULL;
    
    struct memblock mem_param_tmp_140513;
    
    mem_param_tmp_140513.references = NULL;
    
    struct memblock mem_param_tmp_140512;
    
    mem_param_tmp_140512.references = NULL;
    
    struct memblock mem_param_tmp_140511;
    
    mem_param_tmp_140511.references = NULL;
    
    struct memblock mem_param_tmp_140510;
    
    mem_param_tmp_140510.references = NULL;
    
    struct memblock mem_param_tmp_140509;
    
    mem_param_tmp_140509.references = NULL;
    
    struct memblock mem_param_tmp_140508;
    
    mem_param_tmp_140508.references = NULL;
    
    struct memblock mem_param_tmp_140507;
    
    mem_param_tmp_140507.references = NULL;
    
    struct memblock mem_param_tmp_140506;
    
    mem_param_tmp_140506.references = NULL;
    
    struct memblock mem_param_tmp_140505;
    
    mem_param_tmp_140505.references = NULL;
    
    struct memblock ext_mem_140295;
    
    ext_mem_140295.references = NULL;
    
    struct memblock ext_mem_140296;
    
    ext_mem_140296.references = NULL;
    
    struct memblock ext_mem_140297;
    
    ext_mem_140297.references = NULL;
    
    struct memblock mem_140293;
    
    mem_140293.references = NULL;
    
    struct memblock mem_140291;
    
    mem_140291.references = NULL;
    
    struct memblock mem_140289;
    
    mem_140289.references = NULL;
    
    struct memblock mem_140287;
    
    mem_140287.references = NULL;
    
    struct memblock ext_mem_140284;
    
    ext_mem_140284.references = NULL;
    
    struct memblock ext_mem_140285;
    
    ext_mem_140285.references = NULL;
    
    struct memblock ext_mem_140286;
    
    ext_mem_140286.references = NULL;
    
    struct memblock mem_140282;
    
    mem_140282.references = NULL;
    
    struct memblock mem_140280;
    
    mem_140280.references = NULL;
    
    struct memblock mem_140278;
    
    mem_140278.references = NULL;
    
    struct memblock mem_140276;
    
    mem_140276.references = NULL;
    
    struct memblock ext_mem_140273;
    
    ext_mem_140273.references = NULL;
    
    struct memblock ext_mem_140274;
    
    ext_mem_140274.references = NULL;
    
    struct memblock ext_mem_140275;
    
    ext_mem_140275.references = NULL;
    
    struct memblock mem_140271;
    
    mem_140271.references = NULL;
    
    struct memblock mem_140269;
    
    mem_140269.references = NULL;
    
    struct memblock mem_140267;
    
    mem_140267.references = NULL;
    
    struct memblock mem_140265;
    
    mem_140265.references = NULL;
    
    struct memblock ext_mem_140262;
    
    ext_mem_140262.references = NULL;
    
    struct memblock ext_mem_140263;
    
    ext_mem_140263.references = NULL;
    
    struct memblock ext_mem_140264;
    
    ext_mem_140264.references = NULL;
    
    struct memblock mem_140260;
    
    mem_140260.references = NULL;
    
    struct memblock mem_140258;
    
    mem_140258.references = NULL;
    
    struct memblock mem_140256;
    
    mem_140256.references = NULL;
    
    struct memblock mem_140254;
    
    mem_140254.references = NULL;
    
    struct memblock ext_mem_140251;
    
    ext_mem_140251.references = NULL;
    
    struct memblock ext_mem_140252;
    
    ext_mem_140252.references = NULL;
    
    struct memblock ext_mem_140253;
    
    ext_mem_140253.references = NULL;
    
    struct memblock mem_140249;
    
    mem_140249.references = NULL;
    
    struct memblock mem_140247;
    
    mem_140247.references = NULL;
    
    struct memblock mem_140245;
    
    mem_140245.references = NULL;
    
    struct memblock mem_140243;
    
    mem_140243.references = NULL;
    
    struct memblock ext_mem_140240;
    
    ext_mem_140240.references = NULL;
    
    struct memblock ext_mem_140241;
    
    ext_mem_140241.references = NULL;
    
    struct memblock ext_mem_140242;
    
    ext_mem_140242.references = NULL;
    
    struct memblock mem_140238;
    
    mem_140238.references = NULL;
    
    struct memblock mem_140236;
    
    mem_140236.references = NULL;
    
    struct memblock mem_140234;
    
    mem_140234.references = NULL;
    
    struct memblock mem_140232;
    
    mem_140232.references = NULL;
    
    struct memblock ext_mem_140229;
    
    ext_mem_140229.references = NULL;
    
    struct memblock ext_mem_140230;
    
    ext_mem_140230.references = NULL;
    
    struct memblock ext_mem_140231;
    
    ext_mem_140231.references = NULL;
    
    struct memblock mem_140227;
    
    mem_140227.references = NULL;
    
    struct memblock mem_140225;
    
    mem_140225.references = NULL;
    
    struct memblock mem_140223;
    
    mem_140223.references = NULL;
    
    struct memblock mem_140221;
    
    mem_140221.references = NULL;
    
    struct memblock ext_mem_140218;
    
    ext_mem_140218.references = NULL;
    
    struct memblock ext_mem_140219;
    
    ext_mem_140219.references = NULL;
    
    struct memblock ext_mem_140220;
    
    ext_mem_140220.references = NULL;
    
    struct memblock mem_140216;
    
    mem_140216.references = NULL;
    
    struct memblock mem_140214;
    
    mem_140214.references = NULL;
    
    struct memblock mem_140212;
    
    mem_140212.references = NULL;
    
    struct memblock mem_140210;
    
    mem_140210.references = NULL;
    
    struct memblock ext_mem_140207;
    
    ext_mem_140207.references = NULL;
    
    struct memblock ext_mem_140208;
    
    ext_mem_140208.references = NULL;
    
    struct memblock ext_mem_140209;
    
    ext_mem_140209.references = NULL;
    
    struct memblock mem_140205;
    
    mem_140205.references = NULL;
    
    struct memblock mem_140203;
    
    mem_140203.references = NULL;
    
    struct memblock mem_140201;
    
    mem_140201.references = NULL;
    
    struct memblock mem_140199;
    
    mem_140199.references = NULL;
    
    struct memblock mem_param_138367;
    
    mem_param_138367.references = NULL;
    
    struct memblock mem_param_138363;
    
    mem_param_138363.references = NULL;
    
    struct memblock mem_param_138359;
    
    mem_param_138359.references = NULL;
    
    struct memblock mem_param_138355;
    
    mem_param_138355.references = NULL;
    
    struct memblock mem_param_138351;
    
    mem_param_138351.references = NULL;
    
    struct memblock mem_param_138347;
    
    mem_param_138347.references = NULL;
    
    struct memblock mem_param_138343;
    
    mem_param_138343.references = NULL;
    
    struct memblock mem_param_138339;
    
    mem_param_138339.references = NULL;
    
    struct memblock mem_param_138335;
    
    mem_param_138335.references = NULL;
    
    struct memblock mem_param_138331;
    
    mem_param_138331.references = NULL;
    
    struct memblock mem_param_138327;
    
    mem_param_138327.references = NULL;
    
    struct memblock mem_param_138323;
    
    mem_param_138323.references = NULL;
    
    struct memblock mem_param_138319;
    
    mem_param_138319.references = NULL;
    
    struct memblock mem_param_138315;
    
    mem_param_138315.references = NULL;
    
    struct memblock mem_param_138311;
    
    mem_param_138311.references = NULL;
    
    struct memblock mem_param_138307;
    
    mem_param_138307.references = NULL;
    
    struct memblock mem_param_138303;
    
    mem_param_138303.references = NULL;
    
    struct memblock mem_param_138299;
    
    mem_param_138299.references = NULL;
    
    struct memblock mem_param_138295;
    
    mem_param_138295.references = NULL;
    
    struct memblock mem_param_138291;
    
    mem_param_138291.references = NULL;
    
    struct memblock mem_param_138287;
    
    mem_param_138287.references = NULL;
    
    struct memblock mem_param_138283;
    
    mem_param_138283.references = NULL;
    
    struct memblock mem_param_138279;
    
    mem_param_138279.references = NULL;
    
    struct memblock mem_param_138275;
    
    mem_param_138275.references = NULL;
    
    struct memblock mem_param_138271;
    
    mem_param_138271.references = NULL;
    
    struct memblock mem_param_138267;
    
    mem_param_138267.references = NULL;
    
    struct memblock mem_param_138263;
    
    mem_param_138263.references = NULL;
    
    struct memblock ext_mem_140379;
    
    ext_mem_140379.references = NULL;
    
    struct memblock ext_mem_140380;
    
    ext_mem_140380.references = NULL;
    
    struct memblock ext_mem_140381;
    
    ext_mem_140381.references = NULL;
    
    struct memblock ext_mem_140382;
    
    ext_mem_140382.references = NULL;
    
    struct memblock ext_mem_140383;
    
    ext_mem_140383.references = NULL;
    
    struct memblock ext_mem_140384;
    
    ext_mem_140384.references = NULL;
    
    struct memblock ext_mem_140385;
    
    ext_mem_140385.references = NULL;
    
    struct memblock ext_mem_140386;
    
    ext_mem_140386.references = NULL;
    
    struct memblock ext_mem_140387;
    
    ext_mem_140387.references = NULL;
    
    struct memblock ext_mem_140388;
    
    ext_mem_140388.references = NULL;
    
    struct memblock ext_mem_140389;
    
    ext_mem_140389.references = NULL;
    
    struct memblock ext_mem_140390;
    
    ext_mem_140390.references = NULL;
    
    struct memblock ext_mem_140391;
    
    ext_mem_140391.references = NULL;
    
    struct memblock ext_mem_140392;
    
    ext_mem_140392.references = NULL;
    
    struct memblock ext_mem_140393;
    
    ext_mem_140393.references = NULL;
    
    struct memblock ext_mem_140394;
    
    ext_mem_140394.references = NULL;
    
    struct memblock ext_mem_140395;
    
    ext_mem_140395.references = NULL;
    
    struct memblock ext_mem_140396;
    
    ext_mem_140396.references = NULL;
    
    struct memblock ext_mem_140397;
    
    ext_mem_140397.references = NULL;
    
    struct memblock ext_mem_140398;
    
    ext_mem_140398.references = NULL;
    
    struct memblock ext_mem_140399;
    
    ext_mem_140399.references = NULL;
    
    struct memblock ext_mem_140400;
    
    ext_mem_140400.references = NULL;
    
    struct memblock ext_mem_140401;
    
    ext_mem_140401.references = NULL;
    
    struct memblock ext_mem_140402;
    
    ext_mem_140402.references = NULL;
    
    struct memblock ext_mem_140403;
    
    ext_mem_140403.references = NULL;
    
    struct memblock ext_mem_140404;
    
    ext_mem_140404.references = NULL;
    
    struct memblock ext_mem_140405;
    
    ext_mem_140405.references = NULL;
    
    struct memblock mem_out_140504;
    
    mem_out_140504.references = NULL;
    
    struct memblock mem_out_140503;
    
    mem_out_140503.references = NULL;
    
    struct memblock mem_out_140502;
    
    mem_out_140502.references = NULL;
    
    struct memblock mem_out_140501;
    
    mem_out_140501.references = NULL;
    
    struct memblock mem_out_140500;
    
    mem_out_140500.references = NULL;
    
    struct memblock mem_out_140499;
    
    mem_out_140499.references = NULL;
    
    struct memblock mem_out_140498;
    
    mem_out_140498.references = NULL;
    
    struct memblock mem_out_140497;
    
    mem_out_140497.references = NULL;
    
    struct memblock mem_out_140496;
    
    mem_out_140496.references = NULL;
    
    struct memblock mem_out_140495;
    
    mem_out_140495.references = NULL;
    
    struct memblock mem_out_140494;
    
    mem_out_140494.references = NULL;
    
    struct memblock mem_out_140493;
    
    mem_out_140493.references = NULL;
    
    struct memblock mem_out_140492;
    
    mem_out_140492.references = NULL;
    
    struct memblock mem_out_140491;
    
    mem_out_140491.references = NULL;
    
    struct memblock mem_out_140490;
    
    mem_out_140490.references = NULL;
    
    struct memblock mem_out_140489;
    
    mem_out_140489.references = NULL;
    
    struct memblock mem_out_140488;
    
    mem_out_140488.references = NULL;
    
    struct memblock mem_out_140487;
    
    mem_out_140487.references = NULL;
    
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_138368_cached_sizze_141030 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138368, &mem_138368_cached_sizze_141030, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138369_cached_sizze_141031 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138369, &mem_138369_cached_sizze_141031, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138378_cached_sizze_141032 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138378, &mem_138378_cached_sizze_141032, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138385_cached_sizze_141033 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138385, &mem_138385_cached_sizze_141033, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138400_cached_sizze_141034 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138400, &mem_138400_cached_sizze_141034, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138405_cached_sizze_141035 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138405, &mem_138405_cached_sizze_141035, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138416_cached_sizze_141036 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138416, &mem_138416_cached_sizze_141036, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138417_cached_sizze_141037 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138417, &mem_138417_cached_sizze_141037, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138425_cached_sizze_141038 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138425, &mem_138425_cached_sizze_141038, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138439_cached_sizze_141039 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138439, &mem_138439_cached_sizze_141039, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138440_cached_sizze_141040 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138440, &mem_138440_cached_sizze_141040, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138448_cached_sizze_141041 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138448, &mem_138448_cached_sizze_141041, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138462_cached_sizze_141042 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138462, &mem_138462_cached_sizze_141042, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138463_cached_sizze_141043 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138463, &mem_138463_cached_sizze_141043, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138464_cached_sizze_141044 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138464, &mem_138464_cached_sizze_141044, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138477_cached_sizze_141045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138477, &mem_138477_cached_sizze_141045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138478_cached_sizze_141046 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138478, &mem_138478_cached_sizze_141046, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138479_cached_sizze_141047 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138479, &mem_138479_cached_sizze_141047, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138510_cached_sizze_141048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138510, &mem_138510_cached_sizze_141048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138511_cached_sizze_141049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138511, &mem_138511_cached_sizze_141049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138512_cached_sizze_141050 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138512, &mem_138512_cached_sizze_141050, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138528_cached_sizze_141051 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138528, &mem_138528_cached_sizze_141051, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138529_cached_sizze_141052 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138529, &mem_138529_cached_sizze_141052, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138530_cached_sizze_141053 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138530, &mem_138530_cached_sizze_141053, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138543_cached_sizze_141054 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138543, &mem_138543_cached_sizze_141054, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138544_cached_sizze_141055 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138544, &mem_138544_cached_sizze_141055, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138545_cached_sizze_141056 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138545, &mem_138545_cached_sizze_141056, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138591_cached_sizze_141057 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138591, &mem_138591_cached_sizze_141057, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138592_cached_sizze_141058 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138592, &mem_138592_cached_sizze_141058, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138593_cached_sizze_141059 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138593, &mem_138593_cached_sizze_141059, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138594_cached_sizze_141060 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138594, &mem_138594_cached_sizze_141060, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138615_cached_sizze_141061 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138615, &mem_138615_cached_sizze_141061, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138616_cached_sizze_141062 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138616, &mem_138616_cached_sizze_141062, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138617_cached_sizze_141063 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138617, &mem_138617_cached_sizze_141063, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138618_cached_sizze_141064 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138618, &mem_138618_cached_sizze_141064, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138635_cached_sizze_141065 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138635, &mem_138635_cached_sizze_141065, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138636_cached_sizze_141066 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138636, &mem_138636_cached_sizze_141066, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138637_cached_sizze_141067 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138637, &mem_138637_cached_sizze_141067, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138638_cached_sizze_141068 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138638, &mem_138638_cached_sizze_141068, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138679_cached_sizze_141069 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138679, &mem_138679_cached_sizze_141069, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138684_cached_sizze_141070 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138684, &mem_138684_cached_sizze_141070, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138695_cached_sizze_141071 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138695, &mem_138695_cached_sizze_141071, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138700_cached_sizze_141072 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138700, &mem_138700_cached_sizze_141072, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138707_cached_sizze_141073 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138707, &mem_138707_cached_sizze_141073, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138718_cached_sizze_141074 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138718, &mem_138718_cached_sizze_141074, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138723_cached_sizze_141075 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_138723, &mem_138723_cached_sizze_141075, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138754_cached_sizze_141076 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138754, &mem_138754_cached_sizze_141076, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138759_cached_sizze_141077 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138759, &mem_138759_cached_sizze_141077, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138770_cached_sizze_141078 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138770, &mem_138770_cached_sizze_141078, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138775_cached_sizze_141079 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138775, &mem_138775_cached_sizze_141079, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138786_cached_sizze_141080 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138786, &mem_138786_cached_sizze_141080, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138791_cached_sizze_141081 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138791, &mem_138791_cached_sizze_141081, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138802_cached_sizze_141082 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138802, &mem_138802_cached_sizze_141082, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138803_cached_sizze_141083 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138803, &mem_138803_cached_sizze_141083, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138811_cached_sizze_141084 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138811, &mem_138811_cached_sizze_141084, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138825_cached_sizze_141085 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138825, &mem_138825_cached_sizze_141085, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138830_cached_sizze_141086 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138830, &mem_138830_cached_sizze_141086, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138841_cached_sizze_141087 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_138841, &mem_138841_cached_sizze_141087, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138846_cached_sizze_141088 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_138846, &mem_138846_cached_sizze_141088, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138857_cached_sizze_141089 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138857, &mem_138857_cached_sizze_141089, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138862_cached_sizze_141090 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138862, &mem_138862_cached_sizze_141090, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138873_cached_sizze_141091 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_138873, &mem_138873_cached_sizze_141091, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138878_cached_sizze_141092 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_138878, &mem_138878_cached_sizze_141092, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138889_cached_sizze_141093 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138889, &mem_138889_cached_sizze_141093, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138894_cached_sizze_141094 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138894, &mem_138894_cached_sizze_141094, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138905_cached_sizze_141095 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138905, &mem_138905_cached_sizze_141095, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138906_cached_sizze_141096 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138906, &mem_138906_cached_sizze_141096, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138907_cached_sizze_141097 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138907, &mem_138907_cached_sizze_141097, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138935_cached_sizze_141098 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_138935, &mem_138935_cached_sizze_141098, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138941_cached_sizze_141099 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_138941, &mem_138941_cached_sizze_141099, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138946_cached_sizze_141100 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138946, &mem_138946_cached_sizze_141100, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138962_cached_sizze_141101 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138962, &mem_138962_cached_sizze_141101, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138967_cached_sizze_141102 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138967, &mem_138967_cached_sizze_141102, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138978_cached_sizze_141103 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_138978, &mem_138978_cached_sizze_141103, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_138983_cached_sizze_141104 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_138983, &mem_138983_cached_sizze_141104, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139001_cached_sizze_141106 < (int64_t) 93312) {
        err = lexical_realloc(ctx, &mem_139001, &mem_139001_cached_sizze_141106, (int64_t) 93312);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139007_cached_sizze_141107 < (int64_t) 5832) {
        err = lexical_realloc(ctx, &mem_139007, &mem_139007_cached_sizze_141107, (int64_t) 5832);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139012_cached_sizze_141108 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139012, &mem_139012_cached_sizze_141108, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139035_cached_sizze_141110 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_139035, &mem_139035_cached_sizze_141110, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139040_cached_sizze_141111 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139040, &mem_139040_cached_sizze_141111, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139051_cached_sizze_141112 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_139051, &mem_139051_cached_sizze_141112, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139056_cached_sizze_141113 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_139056, &mem_139056_cached_sizze_141113, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139067_cached_sizze_141114 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139067, &mem_139067_cached_sizze_141114, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139072_cached_sizze_141115 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139072, &mem_139072_cached_sizze_141115, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139083_cached_sizze_141116 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139083, &mem_139083_cached_sizze_141116, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139088_cached_sizze_141117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139088, &mem_139088_cached_sizze_141117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139099_cached_sizze_141118 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139099, &mem_139099_cached_sizze_141118, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139100_cached_sizze_141119 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139100, &mem_139100_cached_sizze_141119, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139109_cached_sizze_141120 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139109, &mem_139109_cached_sizze_141120, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139110_cached_sizze_141121 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139110, &mem_139110_cached_sizze_141121, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139131_cached_sizze_141122 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139131, &mem_139131_cached_sizze_141122, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139136_cached_sizze_141123 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139136, &mem_139136_cached_sizze_141123, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139147_cached_sizze_141124 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139147, &mem_139147_cached_sizze_141124, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139152_cached_sizze_141125 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139152, &mem_139152_cached_sizze_141125, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139163_cached_sizze_141126 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139163, &mem_139163_cached_sizze_141126, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139164_cached_sizze_141127 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139164, &mem_139164_cached_sizze_141127, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139177_cached_sizze_141128 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139177, &mem_139177_cached_sizze_141128, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139184_cached_sizze_141129 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139184, &mem_139184_cached_sizze_141129, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139189_cached_sizze_141130 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139189, &mem_139189_cached_sizze_141130, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139200_cached_sizze_141131 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139200, &mem_139200_cached_sizze_141131, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139205_cached_sizze_141132 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139205, &mem_139205_cached_sizze_141132, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139216_cached_sizze_141133 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139216, &mem_139216_cached_sizze_141133, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139217_cached_sizze_141134 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139217, &mem_139217_cached_sizze_141134, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139226_cached_sizze_141135 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139226, &mem_139226_cached_sizze_141135, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139227_cached_sizze_141136 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139227, &mem_139227_cached_sizze_141136, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139248_cached_sizze_141137 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139248, &mem_139248_cached_sizze_141137, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139249_cached_sizze_141138 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139249, &mem_139249_cached_sizze_141138, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139250_cached_sizze_141139 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139250, &mem_139250_cached_sizze_141139, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139251_cached_sizze_141140 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139251, &mem_139251_cached_sizze_141140, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139272_cached_sizze_141141 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139272, &mem_139272_cached_sizze_141141, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139273_cached_sizze_141142 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139273, &mem_139273_cached_sizze_141142, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139274_cached_sizze_141143 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139274, &mem_139274_cached_sizze_141143, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139275_cached_sizze_141144 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139275, &mem_139275_cached_sizze_141144, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139292_cached_sizze_141145 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139292, &mem_139292_cached_sizze_141145, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139299_cached_sizze_141146 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139299, &mem_139299_cached_sizze_141146, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139300_cached_sizze_141147 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139300, &mem_139300_cached_sizze_141147, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139301_cached_sizze_141148 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139301, &mem_139301_cached_sizze_141148, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139356_cached_sizze_141149 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139356, &mem_139356_cached_sizze_141149, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139357_cached_sizze_141150 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139357, &mem_139357_cached_sizze_141150, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139358_cached_sizze_141151 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139358, &mem_139358_cached_sizze_141151, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139359_cached_sizze_141152 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139359, &mem_139359_cached_sizze_141152, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139360_cached_sizze_141153 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139360, &mem_139360_cached_sizze_141153, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139361_cached_sizze_141154 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139361, &mem_139361_cached_sizze_141154, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139362_cached_sizze_141155 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139362, &mem_139362_cached_sizze_141155, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139363_cached_sizze_141156 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139363, &mem_139363_cached_sizze_141156, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139364_cached_sizze_141157 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139364, &mem_139364_cached_sizze_141157, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139404_cached_sizze_141158 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139404, &mem_139404_cached_sizze_141158, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139405_cached_sizze_141159 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139405, &mem_139405_cached_sizze_141159, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139406_cached_sizze_141160 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139406, &mem_139406_cached_sizze_141160, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139407_cached_sizze_141161 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139407, &mem_139407_cached_sizze_141161, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139408_cached_sizze_141162 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139408, &mem_139408_cached_sizze_141162, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139409_cached_sizze_141163 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139409, &mem_139409_cached_sizze_141163, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139410_cached_sizze_141164 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139410, &mem_139410_cached_sizze_141164, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139411_cached_sizze_141165 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139411, &mem_139411_cached_sizze_141165, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139412_cached_sizze_141166 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139412, &mem_139412_cached_sizze_141166, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_139443_cached_sizze_141167 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139443, &mem_139443_cached_sizze_141167, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:13-33
    if (mem_139444_cached_sizze_141168 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139444, &mem_139444_cached_sizze_141168, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139457_cached_sizze_141169 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139457, &mem_139457_cached_sizze_141169, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139464_cached_sizze_141170 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139464, &mem_139464_cached_sizze_141170, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139540_cached_sizze_141171 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139540, &mem_139540_cached_sizze_141171, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139541_cached_sizze_141172 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139541, &mem_139541_cached_sizze_141172, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139542_cached_sizze_141173 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139542, &mem_139542_cached_sizze_141173, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139558_cached_sizze_141174 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139558, &mem_139558_cached_sizze_141174, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139559_cached_sizze_141175 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139559, &mem_139559_cached_sizze_141175, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139560_cached_sizze_141176 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139560, &mem_139560_cached_sizze_141176, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139573_cached_sizze_141177 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139573, &mem_139573_cached_sizze_141177, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139580_cached_sizze_141178 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139580, &mem_139580_cached_sizze_141178, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139581_cached_sizze_141179 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139581, &mem_139581_cached_sizze_141179, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139621_cached_sizze_141180 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139621, &mem_139621_cached_sizze_141180, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139622_cached_sizze_141181 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139622, &mem_139622_cached_sizze_141181, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139623_cached_sizze_141182 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139623, &mem_139623_cached_sizze_141182, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139624_cached_sizze_141183 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139624, &mem_139624_cached_sizze_141183, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139641_cached_sizze_141184 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139641, &mem_139641_cached_sizze_141184, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139642_cached_sizze_141185 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139642, &mem_139642_cached_sizze_141185, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139643_cached_sizze_141186 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139643, &mem_139643_cached_sizze_141186, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139644_cached_sizze_141187 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139644, &mem_139644_cached_sizze_141187, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139685_cached_sizze_141188 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139685, &mem_139685_cached_sizze_141188, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139686_cached_sizze_141189 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139686, &mem_139686_cached_sizze_141189, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139697_cached_sizze_141190 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139697, &mem_139697_cached_sizze_141190, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139698_cached_sizze_141191 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139698, &mem_139698_cached_sizze_141191, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139707_cached_sizze_141192 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139707, &mem_139707_cached_sizze_141192, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139708_cached_sizze_141193 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139708, &mem_139708_cached_sizze_141193, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139739_cached_sizze_141194 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139739, &mem_139739_cached_sizze_141194, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139740_cached_sizze_141195 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139740, &mem_139740_cached_sizze_141195, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139749_cached_sizze_141196 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139749, &mem_139749_cached_sizze_141196, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139750_cached_sizze_141197 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139750, &mem_139750_cached_sizze_141197, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139771_cached_sizze_141198 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139771, &mem_139771_cached_sizze_141198, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139772_cached_sizze_141199 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139772, &mem_139772_cached_sizze_141199, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139783_cached_sizze_141200 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139783, &mem_139783_cached_sizze_141200, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139784_cached_sizze_141201 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139784, &mem_139784_cached_sizze_141201, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139793_cached_sizze_141202 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139793, &mem_139793_cached_sizze_141202, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139794_cached_sizze_141203 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139794, &mem_139794_cached_sizze_141203, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139825_cached_sizze_141204 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139825, &mem_139825_cached_sizze_141204, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139826_cached_sizze_141205 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_139826, &mem_139826_cached_sizze_141205, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139837_cached_sizze_141206 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139837, &mem_139837_cached_sizze_141206, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139838_cached_sizze_141207 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139838, &mem_139838_cached_sizze_141207, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139847_cached_sizze_141208 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139847, &mem_139847_cached_sizze_141208, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139848_cached_sizze_141209 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139848, &mem_139848_cached_sizze_141209, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139879_cached_sizze_141210 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139879, &mem_139879_cached_sizze_141210, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139880_cached_sizze_141211 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139880, &mem_139880_cached_sizze_141211, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139891_cached_sizze_141212 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139891, &mem_139891_cached_sizze_141212, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139892_cached_sizze_141213 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_139892, &mem_139892_cached_sizze_141213, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139901_cached_sizze_141214 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139901, &mem_139901_cached_sizze_141214, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139902_cached_sizze_141215 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_139902, &mem_139902_cached_sizze_141215, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139933_cached_sizze_141216 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139933, &mem_139933_cached_sizze_141216, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139934_cached_sizze_141217 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139934, &mem_139934_cached_sizze_141217, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139935_cached_sizze_141218 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139935, &mem_139935_cached_sizze_141218, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139948_cached_sizze_141219 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139948, &mem_139948_cached_sizze_141219, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139949_cached_sizze_141220 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139949, &mem_139949_cached_sizze_141220, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139950_cached_sizze_141221 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_139950, &mem_139950_cached_sizze_141221, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139981_cached_sizze_141222 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139981, &mem_139981_cached_sizze_141222, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139982_cached_sizze_141223 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139982, &mem_139982_cached_sizze_141223, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139983_cached_sizze_141224 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139983, &mem_139983_cached_sizze_141224, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_139984_cached_sizze_141225 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_139984, &mem_139984_cached_sizze_141225, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140001_cached_sizze_141226 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140001, &mem_140001_cached_sizze_141226, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140002_cached_sizze_141227 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140002, &mem_140002_cached_sizze_141227, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140003_cached_sizze_141228 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140003, &mem_140003_cached_sizze_141228, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140004_cached_sizze_141229 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140004, &mem_140004_cached_sizze_141229, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140045_cached_sizze_141230 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140045, &mem_140045_cached_sizze_141230, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140046_cached_sizze_141231 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140046, &mem_140046_cached_sizze_141231, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140059_cached_sizze_141232 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140059, &mem_140059_cached_sizze_141232, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140066_cached_sizze_141233 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140066, &mem_140066_cached_sizze_141233, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140071_cached_sizze_141234 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140071, &mem_140071_cached_sizze_141234, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140082_cached_sizze_141235 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140082, &mem_140082_cached_sizze_141235, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140083_cached_sizze_141236 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140083, &mem_140083_cached_sizze_141236, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140096_cached_sizze_141237 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140096, &mem_140096_cached_sizze_141237, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140103_cached_sizze_141238 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140103, &mem_140103_cached_sizze_141238, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140108_cached_sizze_141239 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140108, &mem_140108_cached_sizze_141239, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140119_cached_sizze_141240 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140119, &mem_140119_cached_sizze_141240, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140120_cached_sizze_141241 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_140120, &mem_140120_cached_sizze_141241, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140129_cached_sizze_141242 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140129, &mem_140129_cached_sizze_141242, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140130_cached_sizze_141243 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140130, &mem_140130_cached_sizze_141243, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140151_cached_sizze_141244 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_140151, &mem_140151_cached_sizze_141244, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140156_cached_sizze_141245 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140156, &mem_140156_cached_sizze_141245, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140167_cached_sizze_141246 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140167, &mem_140167_cached_sizze_141246, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140168_cached_sizze_141247 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_140168, &mem_140168_cached_sizze_141247, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140177_cached_sizze_141248 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140177, &mem_140177_cached_sizze_141248, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_140178_cached_sizze_141249 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_140178, &mem_140178_cached_sizze_141249, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:624:5-629:51
    if (memblock_set(ctx, &mem_param_138263, &wdown_mem_138230, "wdown_mem_138230") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138267, &wkey_mem_138231, "wkey_mem_138231") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138271, &wout_mem_138232, "wout_mem_138232") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138275, &wpe_mem_138233, "wpe_mem_138233") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138279, &wqry_mem_138234, "wqry_mem_138234") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138283, &wte_mem_138235, "wte_mem_138235") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138287, &wup_mem_138236, "wup_mem_138236") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138291, &wval_mem_138237, "wval_mem_138237") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138295, &wvoc_mem_138238, "wvoc_mem_138238") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138299, &wdown_mem_138239, "wdown_mem_138239") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138303, &wkey_mem_138240, "wkey_mem_138240") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138307, &wout_mem_138241, "wout_mem_138241") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138311, &wpe_mem_138242, "wpe_mem_138242") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138315, &wqry_mem_138243, "wqry_mem_138243") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138319, &wte_mem_138244, "wte_mem_138244") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138323, &wup_mem_138245, "wup_mem_138245") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138327, &wval_mem_138246, "wval_mem_138246") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138331, &wvoc_mem_138247, "wvoc_mem_138247") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138335, &wdown_mem_138248, "wdown_mem_138248") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138339, &wkey_mem_138249, "wkey_mem_138249") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138343, &wout_mem_138250, "wout_mem_138250") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138347, &wpe_mem_138251, "wpe_mem_138251") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138351, &wqry_mem_138252, "wqry_mem_138252") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138355, &wte_mem_138253, "wte_mem_138253") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138359, &wup_mem_138254, "wup_mem_138254") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138363, &wval_mem_138255, "wval_mem_138255") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_138367, &wvoc_mem_138256, "wvoc_mem_138256") != 0)
        return 1;
    for (int64_t step_124785 = 0; step_124785 < (int64_t) 500; step_124785++) {
        // futhark/microgpt.fut:626:16-25
        
        int64_t dl_124813 = ((int64_t *) dls_mem_138258.mem)[step_124785];
        
        // futhark/microgpt.fut:466:37-40
        
        int64_t zl_rhs_124818 = sub64(dl_124813, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137179 = 0; i_137179 < (int64_t) 16; i_137179++) {
            // futhark/microgpt.fut:466:25-81
            
            bool cond_127703 = slt64(i_137179, zl_rhs_124818);
            
            // futhark/microgpt.fut:466:56-59
            
            int64_t zeze_lhs_127704 = add64((int64_t) 1, i_137179);
            
            // futhark/microgpt.fut:466:47-60
            
            bool x_127705 = sle64((int64_t) 0, zeze_lhs_127704);
            
            // futhark/microgpt.fut:466:47-60
            
            bool y_127706 = slt64(zeze_lhs_127704, (int64_t) 16);
            
            // futhark/microgpt.fut:466:47-60
            
            bool bounds_check_127707 = x_127705 && y_127706;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_127708 = !cond_127703;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_127709 = bounds_check_127707 || loop_not_taken_127708;
            
            // futhark/microgpt.fut:466:47-60
            
            bool index_certs_127710;
            
            if (!protect_assert_disj_127709) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_127704, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:466:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:466:3-83\n   #6  futhark/microgpt.fut:573:18-38\n   #7  futhark/microgpt.fut:595:26-601:31\n   #8  futhark/microgpt.fut:629:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_127725 = ((int64_t *) seqs_mem_138259.mem)[step_124785 * (int64_t) 16 + i_137179];
            
            // futhark/microgpt.fut:575:37-51
            
            bool x_127726 = sle64((int64_t) 0, tmp_127725);
            
            // futhark/microgpt.fut:575:37-51
            
            bool y_127727 = slt64(tmp_127725, (int64_t) 27);
            
            // futhark/microgpt.fut:575:37-51
            
            bool bounds_check_127728 = x_127726 && y_127727;
            
            // futhark/microgpt.fut:575:37-51
            
            bool index_certs_127729;
            
            if (!bounds_check_127728) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_127725, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:575:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:575:16-55\n   #6  futhark/microgpt.fut:595:26-601:31\n   #7  futhark/microgpt.fut:629:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:466:47-60
            
            int64_t zeze_lhs_127711;
            
            if (cond_127703) {
                int64_t x_136830 = ((int64_t *) seqs_mem_138259.mem)[step_124785 * (int64_t) 16 + zeze_lhs_127704];
                
                zeze_lhs_127711 = x_136830;
            } else {
                zeze_lhs_127711 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137169 = 0; i_137169 < (int64_t) 27; i_137169++) {
                // futhark/microgpt.fut:466:61-65
                
                bool cond_t_res_127715 = zeze_lhs_127711 == i_137169;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_127716 = cond_127703 && cond_t_res_127715;
                
                // futhark/microgpt.fut:466:25-81
                
                double lifted_lambda_res_127717;
                
                if (x_127716) {
                    lifted_lambda_res_127717 = 1.0;
                } else {
                    lifted_lambda_res_127717 = 0.0;
                }
                ((double *) mem_138378)[i_137169] = lifted_lambda_res_127717;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137173 = 0; i_137173 < (int64_t) 16; i_137173++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_127736 = ((double *) mem_param_138283.mem)[tmp_127725 * (int64_t) 16 + i_137173];
                
                ((double *) mem_138385)[i_137173] = lifted_lambda_res_127736;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138368, i_137179 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138385, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138369, i_137179 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138378, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137188 = 0; i_137188 < (int64_t) 16; i_137188++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137184 = 0; i_137184 < (int64_t) 16; i_137184++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_124870 = ((double *) mem_param_138275.mem)[i_137188 * (int64_t) 16 + i_137184];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_124871 = ((double *) mem_138368)[i_137188 * (int64_t) 16 + i_137184];
                
                // futhark/microgpt.fut:279:39-75
                
                double zp_res_124872 = zp_lhs_124870 + zp_rhs_124871;
                
                ((double *) mem_138405)[i_137184] = zp_res_124872;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138400, i_137188 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137198 = 0; i_137198 < (int64_t) 16; i_137198++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127755;
            double r_127757 = 0.0;
            
            for (int64_t i_127756 = 0; i_127756 < (int64_t) 16; i_127756++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127758 = ((double *) mem_138400)[i_137198 * (int64_t) 16 + i_127756];
                
                // futhark/microgpt.fut:280:70-103
                
                double zt_res_127759 = zt_lhs_127758 * zt_lhs_127758;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127760 = r_127757 + zt_res_127759;
                double r_tmp_140567 = zp_res_127760;
                
                r_127757 = r_tmp_140567;
            }
            defunc_0_lifted_lambda_res_127755 = r_127757;
            // futhark/microgpt.fut:280:50-121
            
            double zs_res_127761 = defunc_0_lifted_lambda_res_127755 / 16.0;
            
            // futhark/microgpt.fut:281:23-53
            
            double zp_res_127762 = 1.0e-5 + zs_res_127761;
            
            // futhark/microgpt.fut:281:15-53
            
            double sqrt_res_127763 = futrts_sqrt64(zp_res_127762);
            
            // futhark/microgpt.fut:282:25-35
            
            double zs_res_127764 = 1.0 / sqrt_res_127763;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137192 = 0; i_137192 < (int64_t) 16; i_137192++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_127771 = ((double *) mem_138400)[i_137198 * (int64_t) 16 + i_137192];
                
                // futhark/microgpt.fut:282:5-35
                
                double zt_res_127772 = zs_res_127764 * zt_lhs_127771;
                
                ((double *) mem_138425)[i_137192] = zt_res_127772;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127780;
            double r_127782 = 0.0;
            
            for (int64_t i_127781 = 0; i_127781 < (int64_t) 16; i_127781++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127783 = ((double *) mem_138400)[i_137198 * (int64_t) 16 + i_127781];
                
                // futhark/microgpt.fut:383:70-111
                
                double zt_res_127784 = zt_lhs_127783 * zt_lhs_127783;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127785 = r_127782 + zt_res_127784;
                double r_tmp_140569 = zp_res_127785;
                
                r_127782 = r_tmp_140569;
            }
            defunc_0_lifted_lambda_res_127780 = r_127782;
            // futhark/microgpt.fut:383:48-129
            
            double zs_res_127786 = defunc_0_lifted_lambda_res_127780 / 16.0;
            
            ((double *) mem_138416)[i_137198] = zs_res_127786;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138417, i_137198 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138425, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137209 = 0; i_137209 < (int64_t) 16; i_137209++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127804;
            double r_127806 = 0.0;
            
            for (int64_t i_127805 = 0; i_127805 < (int64_t) 16; i_127805++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127807 = ((double *) mem_138417)[i_137209 * (int64_t) 16 + i_127805];
                
                // futhark/microgpt.fut:283:71-106
                
                double zt_res_127808 = zt_lhs_127807 * zt_lhs_127807;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127809 = r_127806 + zt_res_127808;
                double r_tmp_140572 = zp_res_127809;
                
                r_127806 = r_tmp_140572;
            }
            defunc_0_lifted_lambda_res_127804 = r_127806;
            // futhark/microgpt.fut:283:50-124
            
            double zs_res_127810 = defunc_0_lifted_lambda_res_127804 / 16.0;
            
            // futhark/microgpt.fut:284:24-54
            
            double zp_res_127811 = 1.0e-5 + zs_res_127810;
            
            // futhark/microgpt.fut:284:16-54
            
            double sqrt_res_127812 = futrts_sqrt64(zp_res_127811);
            
            // futhark/microgpt.fut:285:25-36
            
            double zs_res_127813 = 1.0 / sqrt_res_127812;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137203 = 0; i_137203 < (int64_t) 16; i_137203++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_127820 = ((double *) mem_138417)[i_137209 * (int64_t) 16 + i_137203];
                
                // futhark/microgpt.fut:285:5-36
                
                double zt_res_127821 = zs_res_127813 * zt_lhs_127820;
                
                ((double *) mem_138448)[i_137203] = zt_res_127821;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127829;
            double r_127831 = 0.0;
            
            for (int64_t i_127830 = 0; i_127830 < (int64_t) 16; i_127830++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127832 = ((double *) mem_138417)[i_137209 * (int64_t) 16 + i_127830];
                
                // futhark/microgpt.fut:378:70-111
                
                double zt_res_127833 = zt_lhs_127832 * zt_lhs_127832;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127834 = r_127831 + zt_res_127833;
                double r_tmp_140574 = zp_res_127834;
                
                r_127831 = r_tmp_140574;
            }
            defunc_0_lifted_lambda_res_127829 = r_127831;
            // futhark/microgpt.fut:378:48-129
            
            double zs_res_127835 = defunc_0_lifted_lambda_res_127829 / 16.0;
            
            ((double *) mem_138439)[i_137209] = zs_res_127835;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138440, i_137209 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138448, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137228 = 0; i_137228 < (int64_t) 16; i_137228++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137218 = 0; i_137218 < (int64_t) 16; i_137218++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131666;
                double r_131668 = 0.0;
                
                for (int64_t i_131667 = 0; i_131667 < (int64_t) 16; i_131667++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131669 = ((double *) mem_param_138279.mem)[i_137218 * (int64_t) 16 + i_131667];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131670 = ((double *) mem_138440)[i_137228 * (int64_t) 16 + i_131667];
                    
                    // futhark/microgpt.fut:286:63-102
                    
                    double zt_res_131671 = zt_lhs_131669 * zt_rhs_131670;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131672 = r_131668 + zt_res_131671;
                    double r_tmp_140581 = zp_res_131672;
                    
                    r_131668 = r_tmp_140581;
                }
                defunc_0_lifted_lambda_res_131666 = r_131668;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131679;
                double r_131681 = 0.0;
                
                for (int64_t i_131680 = 0; i_131680 < (int64_t) 16; i_131680++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131682 = ((double *) mem_param_138267.mem)[i_137218 * (int64_t) 16 + i_131680];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131683 = ((double *) mem_138440)[i_137228 * (int64_t) 16 + i_131680];
                    
                    // futhark/microgpt.fut:287:63-102
                    
                    double zt_res_131684 = zt_lhs_131682 * zt_rhs_131683;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131685 = r_131681 + zt_res_131684;
                    double r_tmp_140582 = zp_res_131685;
                    
                    r_131681 = r_tmp_140582;
                }
                defunc_0_lifted_lambda_res_131679 = r_131681;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_131695;
                double r_131697 = 0.0;
                
                for (int64_t i_131696 = 0; i_131696 < (int64_t) 16; i_131696++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_131698 = ((double *) mem_param_138291.mem)[i_137218 * (int64_t) 16 + i_131696];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_131699 = ((double *) mem_138440)[i_137228 * (int64_t) 16 + i_131696];
                    
                    // futhark/microgpt.fut:288:63-102
                    
                    double zt_res_131700 = zt_lhs_131698 * zt_rhs_131699;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_131701 = r_131697 + zt_res_131700;
                    double r_tmp_140583 = zp_res_131701;
                    
                    r_131697 = r_tmp_140583;
                }
                defunc_0_lifted_lambda_res_131695 = r_131697;
                ((double *) mem_138477)[i_137218] = defunc_0_lifted_lambda_res_131695;
                ((double *) mem_138478)[i_137218] = defunc_0_lifted_lambda_res_131679;
                ((double *) mem_138479)[i_137218] = defunc_0_lifted_lambda_res_131666;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138462, i_137228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138477, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138463, i_137228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138478, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138464, i_137228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137258 = 0; i_137258 < (int64_t) 4; i_137258++) {
            // futhark/microgpt.fut:289:67-70
            
            int64_t zp_lhs_128036 = mul64((int64_t) 4, i_137258);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137248 = 0; i_137248 < (int64_t) 16; i_137248++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137238 = 0; i_137238 < (int64_t) 4; i_137238++) {
                    // futhark/microgpt.fut:289:72-79
                    
                    int64_t tmp_131859 = add64(zp_lhs_128036, i_137238);
                    
                    // futhark/microgpt.fut:289:48-81
                    
                    bool x_131860 = sle64((int64_t) 0, tmp_131859);
                    
                    // futhark/microgpt.fut:289:48-81
                    
                    bool y_131861 = slt64(tmp_131859, (int64_t) 16);
                    
                    // futhark/microgpt.fut:289:48-81
                    
                    bool bounds_check_131862 = x_131860 && y_131861;
                    
                    // futhark/microgpt.fut:289:48-81
                    
                    bool index_certs_131863;
                    
                    if (!bounds_check_131862) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_131859, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:289:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:289:12-82\n   #9  futhark/microgpt.fut:578:5-76\n   #10 futhark/microgpt.fut:595:26-601:31\n   #11 futhark/microgpt.fut:629:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_131864 = ((double *) mem_138464)[i_137248 * (int64_t) 16 + tmp_131859];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_131872 = ((double *) mem_138463)[i_137248 * (int64_t) 16 + tmp_131859];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_131883 = ((double *) mem_138462)[i_137248 * (int64_t) 16 + tmp_131859];
                    
                    ((double *) mem_138543)[i_137238] = lifted_lambda_res_131883;
                    ((double *) mem_138544)[i_137238] = lifted_lambda_res_131872;
                    ((double *) mem_138545)[i_137238] = lifted_lambda_res_131864;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138528, i_137248 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138543, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138529, i_137248 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138544, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138530, i_137248 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138545, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138510, i_137258 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138528, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138511, i_137258 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138529, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138512, i_137258 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138530, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137326 = 0; i_137326 < (int64_t) 4; i_137326++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137283 = 0; i_137283 < (int64_t) 16; i_137283++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137270 = 0; i_137270 < (int64_t) 16; i_137270++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_132265;
                    double r_132267 = 0.0;
                    
                    for (int64_t i_132266 = 0; i_132266 < (int64_t) 4; i_132266++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_132268 = ((double *) mem_138512)[i_137326 * (int64_t) 64 + i_137283 * (int64_t) 4 + i_132266];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_132269 = ((double *) mem_138511)[i_137326 * (int64_t) 64 + i_137270 * (int64_t) 4 + i_132266];
                        
                        // futhark/microgpt.fut:292:110-163
                        
                        double zt_res_132270 = zt_lhs_132268 * zt_rhs_132269;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_132271 = r_132267 + zt_res_132270;
                        double r_tmp_140605 = zp_res_132271;
                        
                        r_132267 = r_tmp_140605;
                    }
                    defunc_0_lifted_lambda_res_132265 = r_132267;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_132278;
                    double r_132280 = 0.0;
                    
                    for (int64_t i_132279 = 0; i_132279 < (int64_t) 4; i_132279++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_132281 = ((double *) mem_138512)[i_137326 * (int64_t) 64 + i_137283 * (int64_t) 4 + i_132279];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_132282 = ((double *) mem_138511)[i_137326 * (int64_t) 64 + i_137270 * (int64_t) 4 + i_132279];
                        
                        // futhark/microgpt.fut:339:87-146
                        
                        double zt_res_132283 = zt_lhs_132281 * zt_rhs_132282;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_132284 = r_132280 + zt_res_132283;
                        double r_tmp_140606 = zp_res_132284;
                        
                        r_132280 = r_tmp_140606;
                    }
                    defunc_0_lifted_lambda_res_132278 = r_132280;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_132294;
                    double r_132296 = 0.0;
                    
                    for (int64_t i_132295 = 0; i_132295 < (int64_t) 4; i_132295++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_132297 = ((double *) mem_138512)[i_137326 * (int64_t) 64 + i_137283 * (int64_t) 4 + i_132295];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_132298 = ((double *) mem_138511)[i_137326 * (int64_t) 64 + i_137270 * (int64_t) 4 + i_132295];
                        
                        // futhark/microgpt.fut:346:87-146
                        
                        double zt_res_132299 = zt_lhs_132297 * zt_rhs_132298;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_132300 = r_132296 + zt_res_132299;
                        double r_tmp_140607 = zp_res_132300;
                        
                        r_132296 = r_tmp_140607;
                    }
                    defunc_0_lifted_lambda_res_132294 = r_132296;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_132312;
                    double r_132314 = 0.0;
                    
                    for (int64_t i_132313 = 0; i_132313 < (int64_t) 4; i_132313++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_132315 = ((double *) mem_138512)[i_137326 * (int64_t) 64 + i_137283 * (int64_t) 4 + i_132313];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_132316 = ((double *) mem_138511)[i_137326 * (int64_t) 64 + i_137270 * (int64_t) 4 + i_132313];
                        
                        // futhark/microgpt.fut:360:87-146
                        
                        double zt_res_132317 = zt_lhs_132315 * zt_rhs_132316;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_132318 = r_132314 + zt_res_132317;
                        double r_tmp_140608 = zp_res_132318;
                        
                        r_132314 = r_tmp_140608;
                    }
                    defunc_0_lifted_lambda_res_132312 = r_132314;
                    ((double *) mem_138635)[i_137270] = defunc_0_lifted_lambda_res_132312;
                    ((double *) mem_138636)[i_137270] = defunc_0_lifted_lambda_res_132294;
                    ((double *) mem_138637)[i_137270] = defunc_0_lifted_lambda_res_132278;
                    ((double *) mem_138638)[i_137270] = defunc_0_lifted_lambda_res_132265;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138615, i_137283 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138635, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138616, i_137283 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138636, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138617, i_137283 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138637, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138618, i_137283 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138638, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137294 = 0; i_137294 < (int64_t) 16; i_137294++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137290 = 0; i_137290 < (int64_t) 16; i_137290++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_128494 = ((double *) mem_138618)[i_137294 * (int64_t) 16 + i_137290];
                    
                    // futhark/microgpt.fut:293:47-78
                    
                    double zs_res_128495 = zs_lhs_128494 / 2.0;
                    double zp_rhs_128496 = ((double *) masks_mem_138257.mem)[step_124785 * (int64_t) 256 + i_137294 * (int64_t) 16 + i_137290];
                    
                    // futhark/microgpt.fut:293:65-102
                    
                    double zp_res_128497 = zs_res_128495 + zp_rhs_128496;
                    
                    ((double *) mem_138684)[i_137290] = zp_res_128497;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138679, i_137294 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138684, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137308 = 0; i_137308 < (int64_t) 16; i_137308++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_136850;
                double redout_137296 = -INFINITY;
                
                for (int64_t i_137297 = 0; i_137297 < (int64_t) 16; i_137297++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_132340 = ((double *) mem_138679)[i_137308 * (int64_t) 16 + i_137297];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_128518 = fmax64(lifted_lambda_res_132340, redout_137296);
                    double redout_tmp_140612 = max_res_128518;
                    
                    redout_137296 = redout_tmp_140612;
                }
                defunc_0_reduce_res_136850 = redout_137296;
                // futhark/microgpt.fut:295:67-76
                
                double neg_res_128519 = -defunc_0_reduce_res_136850;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137300 = 0; i_137300 < (int64_t) 16; i_137300++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_128526 = ((double *) mem_138679)[i_137308 * (int64_t) 16 + i_137300];
                    
                    // futhark/microgpt.fut:295:44-76
                    
                    double zp_res_128527 = neg_res_128519 + zp_lhs_128526;
                    
                    // futhark/microgpt.fut:295:37-76
                    
                    double exp_res_128528 = futrts_exp64(zp_res_128527);
                    
                    ((double *) mem_138700)[i_137300] = exp_res_128528;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_128530;
                double r_128532 = 0.0;
                
                for (int64_t i_128531 = 0; i_128531 < (int64_t) 16; i_128531++) {
                    // futhark/microgpt.fut:296:36-46
                    
                    double lifted_lambda_res_128533 = ((double *) mem_138700)[i_128531];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_128534 = r_128532 + lifted_lambda_res_128533;
                    double r_tmp_140614 = zp_res_128534;
                    
                    r_128532 = r_tmp_140614;
                }
                defunc_0_lifted_lambda_res_128530 = r_128532;
                // futhark/microgpt.fut:297:21-32
                
                double zs_res_128535 = 1.0 / defunc_0_lifted_lambda_res_128530;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137304 = 0; i_137304 < (int64_t) 16; i_137304++) {
                    // futhark/microgpt.fut:297:5-15
                    
                    double zt_lhs_128542 = ((double *) mem_138700)[i_137304];
                    
                    // futhark/microgpt.fut:297:5-32
                    
                    double zt_res_128543 = zs_res_128535 * zt_lhs_128542;
                    
                    ((double *) mem_138707)[i_137304] = zt_res_128543;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138695, i_137308 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138707, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137316 = 0; i_137316 < (int64_t) 16; i_137316++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137312 = 0; i_137312 < (int64_t) 4; i_137312++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_128558;
                    double r_128560 = 0.0;
                    
                    for (int64_t i_128559 = 0; i_128559 < (int64_t) 16; i_128559++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_128561 = ((double *) mem_138695)[i_137316 * (int64_t) 16 + i_128559];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_128562 = ((double *) mem_138510)[i_137326 * (int64_t) 64 + i_128559 * (int64_t) 4 + i_137312];
                        
                        // futhark/microgpt.fut:298:26-72
                        
                        double zt_res_128563 = zt_lhs_128561 * zt_rhs_128562;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_128564 = r_128560 + zt_res_128563;
                        double r_tmp_140618 = zp_res_128564;
                        
                        r_128560 = r_tmp_140618;
                    }
                    defunc_0_lifted_lambda_res_128558 = r_128560;
                    ((double *) mem_138723)[i_137312] = defunc_0_lifted_lambda_res_128558;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138718, i_137316 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138723, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138591, i_137326 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_138615, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138592, i_137326 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_138616, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138593, i_137326 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_138617, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138594, i_137326 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_138718, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137337 = 0; i_137337 < (int64_t) 16; i_137337++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137333 = 0; i_137333 < (int64_t) 16; i_137333++) {
                // futhark/microgpt.fut:299:52-55
                
                int64_t tmp_125172 = sdiv64(i_137333, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-57
                
                bool x_125173 = sle64((int64_t) 0, tmp_125172);
                
                // futhark/microgpt.fut:299:41-57
                
                bool y_125174 = slt64(tmp_125172, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-57
                
                bool bounds_check_125175 = x_125173 && y_125174;
                
                // futhark/microgpt.fut:299:41-57
                
                bool index_certs_125176;
                
                if (!bounds_check_125175) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_125172, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:299:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:299:12-78\n   #6  futhark/microgpt.fut:578:5-76\n   #7  futhark/microgpt.fut:595:26-601:31\n   #8  futhark/microgpt.fut:629:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:299:72-75
                
                int64_t tmp_125177 = smod64(i_137333, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-77
                
                bool x_125178 = sle64((int64_t) 0, tmp_125177);
                
                // futhark/microgpt.fut:299:41-77
                
                bool y_125179 = slt64(tmp_125177, (int64_t) 4);
                
                // futhark/microgpt.fut:299:41-77
                
                bool bounds_check_125180 = x_125178 && y_125179;
                
                // futhark/microgpt.fut:299:41-77
                
                bool index_certs_125181;
                
                if (!bounds_check_125180) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_125177, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:299:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:299:12-78\n   #6  futhark/microgpt.fut:578:5-76\n   #7  futhark/microgpt.fut:595:26-601:31\n   #8  futhark/microgpt.fut:629:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125182 = ((double *) mem_138594)[tmp_125172 * (int64_t) 64 + i_137337 * (int64_t) 4 + tmp_125177];
                
                ((double *) mem_138759)[i_137333] = lifted_lambda_res_125182;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138754, i_137337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137345 = 0; i_137345 < (int64_t) 16; i_137345++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137341 = 0; i_137341 < (int64_t) 16; i_137341++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125197;
                double r_125199 = 0.0;
                
                for (int64_t i_125198 = 0; i_125198 < (int64_t) 16; i_125198++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125200 = ((double *) mem_param_138271.mem)[i_137341 * (int64_t) 16 + i_125198];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125201 = ((double *) mem_138754)[i_137345 * (int64_t) 16 + i_125198];
                    
                    // futhark/microgpt.fut:300:63-103
                    
                    double zt_res_125202 = zt_lhs_125200 * zt_rhs_125201;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125203 = r_125199 + zt_res_125202;
                    double r_tmp_140623 = zp_res_125203;
                    
                    r_125199 = r_tmp_140623;
                }
                defunc_0_lifted_lambda_res_125197 = r_125199;
                ((double *) mem_138775)[i_137341] = defunc_0_lifted_lambda_res_125197;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138770, i_137345 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138775, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137353 = 0; i_137353 < (int64_t) 16; i_137353++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137349 = 0; i_137349 < (int64_t) 16; i_137349++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_125218 = ((double *) mem_138770)[i_137353 * (int64_t) 16 + i_137349];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_125219 = ((double *) mem_138417)[i_137353 * (int64_t) 16 + i_137349];
                
                // futhark/microgpt.fut:301:42-80
                
                double zp_res_125220 = zp_lhs_125218 + zp_rhs_125219;
                
                ((double *) mem_138791)[i_137349] = zp_res_125220;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138786, i_137353 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138791, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137363 = 0; i_137363 < (int64_t) 16; i_137363++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_128675;
            double r_128677 = 0.0;
            
            for (int64_t i_128676 = 0; i_128676 < (int64_t) 16; i_128676++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_128678 = ((double *) mem_138786)[i_137363 * (int64_t) 16 + i_128676];
                
                // futhark/microgpt.fut:302:75-114
                
                double zt_res_128679 = zt_lhs_128678 * zt_lhs_128678;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_128680 = r_128677 + zt_res_128679;
                double r_tmp_140628 = zp_res_128680;
                
                r_128677 = r_tmp_140628;
            }
            defunc_0_lifted_lambda_res_128675 = r_128677;
            // futhark/microgpt.fut:302:54-132
            
            double zs_res_128681 = defunc_0_lifted_lambda_res_128675 / 16.0;
            
            // futhark/microgpt.fut:303:24-55
            
            double zp_res_128682 = 1.0e-5 + zs_res_128681;
            
            // futhark/microgpt.fut:303:16-55
            
            double sqrt_res_128683 = futrts_sqrt64(zp_res_128682);
            
            // futhark/microgpt.fut:304:28-39
            
            double zs_res_128684 = 1.0 / sqrt_res_128683;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137357 = 0; i_137357 < (int64_t) 16; i_137357++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_128691 = ((double *) mem_138786)[i_137363 * (int64_t) 16 + i_137357];
                
                // futhark/microgpt.fut:304:5-39
                
                double zt_res_128692 = zs_res_128684 * zt_lhs_128691;
                
                ((double *) mem_138811)[i_137357] = zt_res_128692;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_128700;
            double r_128702 = 0.0;
            
            for (int64_t i_128701 = 0; i_128701 < (int64_t) 16; i_128701++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_128703 = ((double *) mem_138786)[i_137363 * (int64_t) 16 + i_128701];
                
                // futhark/microgpt.fut:331:70-113
                
                double zt_res_128704 = zt_lhs_128703 * zt_lhs_128703;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_128705 = r_128702 + zt_res_128704;
                double r_tmp_140630 = zp_res_128705;
                
                r_128702 = r_tmp_140630;
            }
            defunc_0_lifted_lambda_res_128700 = r_128702;
            // futhark/microgpt.fut:331:48-131
            
            double zs_res_128706 = defunc_0_lifted_lambda_res_128700 / 16.0;
            
            ((double *) mem_138802)[i_137363] = zs_res_128706;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138803, i_137363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138811, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137372 = 0; i_137372 < (int64_t) 16; i_137372++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137368 = 0; i_137368 < (int64_t) 64; i_137368++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125262;
                double r_125264 = 0.0;
                
                for (int64_t i_125263 = 0; i_125263 < (int64_t) 16; i_125263++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125265 = ((double *) mem_param_138287.mem)[i_137368 * (int64_t) 16 + i_125263];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125266 = ((double *) mem_138803)[i_137372 * (int64_t) 16 + i_125263];
                    
                    // futhark/microgpt.fut:305:63-102
                    
                    double zt_res_125267 = zt_lhs_125265 * zt_rhs_125266;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125268 = r_125264 + zt_res_125267;
                    double r_tmp_140633 = zp_res_125268;
                    
                    r_125264 = r_tmp_140633;
                }
                defunc_0_lifted_lambda_res_125262 = r_125264;
                ((double *) mem_138830)[i_137368] = defunc_0_lifted_lambda_res_125262;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138825, i_137372 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138830, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137380 = 0; i_137380 < (int64_t) 16; i_137380++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137376 = 0; i_137376 < (int64_t) 64; i_137376++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_125283 = ((double *) mem_138825)[i_137380 * (int64_t) 64 + i_137376];
                
                // futhark/microgpt.fut:306:41-69
                
                double max_res_125284 = fmax64(0.0, max_arg0_125283);
                
                ((double *) mem_138846)[i_137376] = max_res_125284;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138841, i_137380 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138846, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137388 = 0; i_137388 < (int64_t) 16; i_137388++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137384 = 0; i_137384 < (int64_t) 16; i_137384++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125299;
                double r_125301 = 0.0;
                
                for (int64_t i_125300 = 0; i_125300 < (int64_t) 64; i_125300++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125302 = ((double *) mem_param_138263.mem)[i_137384 * (int64_t) 64 + i_125300];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125303 = ((double *) mem_138841)[i_137388 * (int64_t) 64 + i_125300];
                    
                    // futhark/microgpt.fut:307:63-104
                    
                    double zt_res_125304 = zt_lhs_125302 * zt_rhs_125303;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125305 = r_125301 + zt_res_125304;
                    double r_tmp_140638 = zp_res_125305;
                    
                    r_125301 = r_tmp_140638;
                }
                defunc_0_lifted_lambda_res_125299 = r_125301;
                ((double *) mem_138862)[i_137384] = defunc_0_lifted_lambda_res_125299;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138857, i_137388 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138862, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137396 = 0; i_137396 < (int64_t) 16; i_137396++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137392 = 0; i_137392 < (int64_t) 16; i_137392++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_125320 = ((double *) mem_138857)[i_137396 * (int64_t) 16 + i_137392];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_125321 = ((double *) mem_138786)[i_137396 * (int64_t) 16 + i_137392];
                
                // futhark/microgpt.fut:308:42-81
                
                double zp_res_125322 = zp_lhs_125320 + zp_rhs_125321;
                
                ((double *) mem_138878)[i_137392] = zp_res_125322;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138873, i_137396 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138878, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137404 = 0; i_137404 < (int64_t) 16; i_137404++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137400 = 0; i_137400 < (int64_t) 27; i_137400++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125337;
                double r_125339 = 0.0;
                
                for (int64_t i_125338 = 0; i_125338 < (int64_t) 16; i_125338++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125340 = ((double *) mem_param_138295.mem)[i_137400 * (int64_t) 16 + i_125338];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125341 = ((double *) mem_138873)[i_137404 * (int64_t) 16 + i_125338];
                    
                    // futhark/microgpt.fut:309:63-103
                    
                    double zt_res_125342 = zt_lhs_125340 * zt_rhs_125341;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125343 = r_125339 + zt_res_125342;
                    double r_tmp_140643 = zp_res_125343;
                    
                    r_125339 = r_tmp_140643;
                }
                defunc_0_lifted_lambda_res_125337 = r_125339;
                ((double *) mem_138894)[i_137400] = defunc_0_lifted_lambda_res_125337;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138889, i_137404 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138894, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137415 = 0; i_137415 < (int64_t) 16; i_137415++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_136870;
            double defunc_0_reduce_res_136871;
            double redout_137406;
            double redout_137407;
            
            redout_137406 = -INFINITY;
            redout_137407 = -INFINITY;
            for (int64_t i_137408 = 0; i_137408 < (int64_t) 27; i_137408++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_132433 = ((double *) mem_138889)[i_137415 * (int64_t) 27 + i_137408];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_128739 = fmax64(lifted_lambda_res_132433, redout_137406);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_128760 = fmax64(lifted_lambda_res_132433, redout_137407);
                double redout_tmp_140647 = max_res_128739;
                double redout_tmp_140648 = max_res_128760;
                
                redout_137406 = redout_tmp_140647;
                redout_137407 = redout_tmp_140648;
            }
            defunc_0_reduce_res_136870 = redout_137406;
            defunc_0_reduce_res_136871 = redout_137407;
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_140649 = 0; nest_i_140649 < (int64_t) 27; nest_i_140649++) {
                ((double *) mem_138907)[i_137415 * (int64_t) 27 + nest_i_140649] = defunc_0_reduce_res_136870;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_140650 = 0; nest_i_140650 < (int64_t) 27; nest_i_140650++) {
                ((double *) mem_138906)[i_137415 * (int64_t) 27 + nest_i_140650] = defunc_0_reduce_res_136871;
            }
            // futhark/microgpt.fut:324:139-164
            
            double neg_res_128771 = -defunc_0_reduce_res_136871;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_128772;
            double r_128774 = 0.0;
            
            for (int64_t i_128773 = 0; i_128773 < (int64_t) 27; i_128773++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_128775 = ((double *) mem_138889)[i_137415 * (int64_t) 27 + i_128773];
                
                // futhark/microgpt.fut:324:114-164
                
                double zp_res_128776 = neg_res_128771 + zp_lhs_128775;
                
                // futhark/microgpt.fut:324:107-164
                
                double neg_res_128777 = -zp_res_128776;
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_128778 = fmax64(0.0, neg_res_128777);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_128779 = fsignum64(max_res_128778);
                
                // futhark/microgpt.fut:324:88-167
                
                double neg_res_128780 = -sgn_res_128779;
                
                // futhark/microgpt.fut:324:79-168
                
                double zp_res_128781 = 1.0 + neg_res_128780;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_128782 = r_128774 + zp_res_128781;
                double r_tmp_140651 = zp_res_128782;
                
                r_128774 = r_tmp_140651;
            }
            defunc_0_lifted_lambda_res_128772 = r_128774;
            // futhark/microgpt.fut:324:48-171
            
            double zs_res_128783 = 1.0 / defunc_0_lifted_lambda_res_128772;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_140652 = 0; nest_i_140652 < (int64_t) 27; nest_i_140652++) {
                ((double *) mem_138905)[i_137415 * (int64_t) 27 + nest_i_140652] = zs_res_128783;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137429 = 0; i_137429 < (int64_t) 16; i_137429++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137425 = 0; i_137425 < (int64_t) 27; i_137425++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_125379 = ((double *) mem_138907)[i_137429 * (int64_t) 27 + i_137425];
                
                // futhark/microgpt.fut:312:85-108
                
                double neg_res_125380 = -neg_arg0_125379;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137421 = 0; i_137421 < (int64_t) 27; i_137421++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_125387 = ((double *) mem_138889)[i_137429 * (int64_t) 27 + i_137421];
                    
                    // futhark/microgpt.fut:312:62-108
                    
                    double zp_res_125388 = neg_res_125380 + zp_lhs_125387;
                    
                    // futhark/microgpt.fut:312:55-108
                    
                    double exp_res_125389 = futrts_exp64(zp_res_125388);
                    
                    ((double *) mem_138946)[i_137421] = exp_res_125389;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_138941, i_137425 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138946, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_138935, i_137429 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_138941, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137437 = 0; i_137437 < (int64_t) 16; i_137437++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137433 = 0; i_137433 < (int64_t) 27; i_137433++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125405;
                double r_125407 = 0.0;
                
                for (int64_t i_125406 = 0; i_125406 < (int64_t) 27; i_125406++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_125408 = ((double *) mem_138935)[i_137437 * (int64_t) 729 + i_137433 * (int64_t) 27 + i_125406];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125409 = r_125407 + lifted_lambda_res_125408;
                    double r_tmp_140658 = zp_res_125409;
                    
                    r_125407 = r_tmp_140658;
                }
                defunc_0_lifted_lambda_res_125405 = r_125407;
                ((double *) mem_138967)[i_137433] = defunc_0_lifted_lambda_res_125405;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138962, i_137437 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138967, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137451 = 0; i_137451 < (int64_t) 16; i_137451++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137447 = 0; i_137447 < (int64_t) 27; i_137447++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125416;
                double r_125418 = 0.0;
                
                for (int64_t i_125417 = 0; i_125417 < (int64_t) 27; i_125417++) {
                    // futhark/microgpt.fut:314:74-317:198
                    
                    bool cond_125419 = i_125417 == i_137447;
                    
                    // futhark/microgpt.fut:314:74-317:198
                    
                    double neg_arg0_125420;
                    
                    if (cond_125419) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_136882;
                        double redout_137439 = -INFINITY;
                        
                        for (int64_t i_137440 = 0; i_137440 < (int64_t) 27; i_137440++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_136888 = ((double *) mem_138889)[i_137451 * (int64_t) 27 + i_137440];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_136891 = fmax64(lifted_lambda_res_136888, redout_137439);
                            double redout_tmp_140662 = max_res_136891;
                            
                            redout_137439 = redout_tmp_140662;
                        }
                        defunc_0_reduce_res_136882 = redout_137439;
                        // futhark/microgpt.fut:315:67-76
                        
                        double neg_res_136893 = -defunc_0_reduce_res_136882;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_138987_cached_sizze_141105 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_138987, &mem_138987_cached_sizze_141105, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_137443 = 0; i_137443 < (int64_t) 27; i_137443++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_136900 = ((double *) mem_138889)[i_137451 * (int64_t) 27 + i_137443];
                            
                            // futhark/microgpt.fut:315:44-76
                            
                            double zp_res_136901 = neg_res_136893 + zp_lhs_136900;
                            
                            // futhark/microgpt.fut:315:37-76
                            
                            double exp_res_136902 = futrts_exp64(zp_res_136901);
                            
                            ((double *) mem_138987)[i_137443] = exp_res_136902;
                        }
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_136905;
                        double r_136907 = 0.0;
                        
                        for (int64_t i_136906 = 0; i_136906 < (int64_t) 27; i_136906++) {
                            // futhark/microgpt.fut:316:36-46
                            
                            double lifted_lambda_res_136908 = ((double *) mem_138987)[i_136906];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_136909 = r_136907 + lifted_lambda_res_136908;
                            double r_tmp_140664 = zp_res_136909;
                            
                            r_136907 = r_tmp_140664;
                        }
                        defunc_0_lifted_lambda_res_136905 = r_136907;
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_136914 = ((double *) mem_138369)[i_137451 * (int64_t) 27 + i_137447];
                        
                        // futhark/microgpt.fut:317:9-51
                        
                        double zt_res_136915 = -6.25e-2 * zt_rhs_136914;
                        
                        // futhark/microgpt.fut:317:67-77
                        
                        double zt_lhs_136916 = ((double *) mem_138987)[i_125417];
                        
                        // futhark/microgpt.fut:317:83-94
                        
                        double zs_res_136917 = 1.0 / defunc_0_lifted_lambda_res_136905;
                        
                        // futhark/microgpt.fut:317:67-94
                        
                        double zt_res_136918 = zt_lhs_136916 * zs_res_136917;
                        
                        // futhark/microgpt.fut:317:58-94
                        
                        double zs_res_136919 = 1.0 / zt_res_136918;
                        
                        // futhark/microgpt.fut:317:27-94
                        
                        double zt_res_136920 = zt_res_136915 * zs_res_136919;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_136921 = ((double *) mem_138935)[i_137451 * (int64_t) 729 + i_137447 * (int64_t) 27 + i_125417];
                        
                        // futhark/microgpt.fut:317:53-127
                        
                        double zt_res_136922 = zt_res_136920 * zt_rhs_136921;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_136923 = ((double *) mem_138962)[i_137451 * (int64_t) 27 + i_137447];
                        
                        // futhark/microgpt.fut:317:143-182
                        
                        double zt_res_136924 = zt_lhs_136923 * zt_lhs_136923;
                        
                        // futhark/microgpt.fut:317:134-182
                        
                        double zs_res_136925 = 1.0 / zt_res_136924;
                        
                        // futhark/microgpt.fut:317:99-182
                        
                        double zt_res_136926 = zt_res_136922 * zs_res_136925;
                        
                        neg_arg0_125420 = zt_res_136926;
                    } else {
                        neg_arg0_125420 = 0.0;
                    }
                    // futhark/microgpt.fut:314:67-317:198
                    
                    double neg_res_125472 = -neg_arg0_125420;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125473 = r_125418 + neg_res_125472;
                    double r_tmp_140661 = zp_res_125473;
                    
                    r_125418 = r_tmp_140661;
                }
                defunc_0_lifted_lambda_res_125416 = r_125418;
                ((double *) mem_138983)[i_137447] = defunc_0_lifted_lambda_res_125416;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_138978, i_137451 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_138983, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137469 = 0; i_137469 < (int64_t) 16; i_137469++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137465 = 0; i_137465 < (int64_t) 27; i_137465++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_125488 = ((double *) mem_138978)[i_137469 * (int64_t) 27 + i_137465];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137461 = 0; i_137461 < (int64_t) 27; i_137461++) {
                    // futhark/microgpt.fut:318:56-321:143
                    
                    bool cond_125491 = i_137461 == i_137465;
                    
                    // futhark/microgpt.fut:318:56-321:143
                    
                    double zp_lhs_125492;
                    
                    if (cond_125491) {
                        // futhark/microgpt.fut:115:13-33
                        
                        double defunc_0_reduce_res_136935;
                        double redout_137453 = -INFINITY;
                        
                        for (int64_t i_137454 = 0; i_137454 < (int64_t) 27; i_137454++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double lifted_lambda_res_136941 = ((double *) mem_138889)[i_137469 * (int64_t) 27 + i_137454];
                            
                            // futhark/microgpt.fut:115:13-33
                            
                            double max_res_136944 = fmax64(lifted_lambda_res_136941, redout_137453);
                            double redout_tmp_140668 = max_res_136944;
                            
                            redout_137453 = redout_tmp_140668;
                        }
                        defunc_0_reduce_res_136935 = redout_137453;
                        // futhark/microgpt.fut:319:67-76
                        
                        double neg_res_136946 = -defunc_0_reduce_res_136935;
                        
                        // futhark/microgpt.fut:4:11-25
                        if (mem_139016_cached_sizze_141109 < (int64_t) 216) {
                            err = lexical_realloc(ctx, &mem_139016, &mem_139016_cached_sizze_141109, (int64_t) 216);
                            if (err != FUTHARK_SUCCESS)
                                goto cleanup;
                        }
                        // futhark/microgpt.fut:4:11-25
                        for (int64_t i_137457 = 0; i_137457 < (int64_t) 27; i_137457++) {
                            // futhark/microgpt.fut:4:11-25
                            
                            double zp_lhs_136953 = ((double *) mem_138889)[i_137469 * (int64_t) 27 + i_137457];
                            
                            // futhark/microgpt.fut:319:44-76
                            
                            double zp_res_136954 = neg_res_136946 + zp_lhs_136953;
                            
                            // futhark/microgpt.fut:319:37-76
                            
                            double exp_res_136955 = futrts_exp64(zp_res_136954);
                            
                            ((double *) mem_139016)[i_137457] = exp_res_136955;
                        }
                        // futhark/microgpt.fut:71:13-49
                        
                        double defunc_0_lifted_lambda_res_136958;
                        double r_136960 = 0.0;
                        
                        for (int64_t i_136959 = 0; i_136959 < (int64_t) 27; i_136959++) {
                            // futhark/microgpt.fut:320:36-46
                            
                            double lifted_lambda_res_136961 = ((double *) mem_139016)[i_136959];
                            
                            // futhark/microgpt.fut:71:40-49
                            
                            double zp_res_136962 = r_136960 + lifted_lambda_res_136961;
                            double r_tmp_140670 = zp_res_136962;
                            
                            r_136960 = r_tmp_140670;
                        }
                        defunc_0_lifted_lambda_res_136958 = r_136960;
                        // futhark/microgpt.fut:4:11-25
                        
                        double zt_rhs_136967 = ((double *) mem_138369)[i_137469 * (int64_t) 27 + i_137465];
                        
                        // futhark/microgpt.fut:321:8-50
                        
                        double zt_res_136968 = -6.25e-2 * zt_rhs_136967;
                        
                        // futhark/microgpt.fut:321:66-76
                        
                        double zt_lhs_136973 = ((double *) mem_139016)[i_137461];
                        
                        // futhark/microgpt.fut:321:82-93
                        
                        double zs_res_136974 = 1.0 / defunc_0_lifted_lambda_res_136958;
                        
                        // futhark/microgpt.fut:321:66-93
                        
                        double zt_res_136975 = zt_lhs_136973 * zs_res_136974;
                        
                        // futhark/microgpt.fut:321:57-93
                        
                        double zs_res_136976 = 1.0 / zt_res_136975;
                        
                        // futhark/microgpt.fut:321:26-93
                        
                        double zt_res_136977 = zt_res_136968 * zs_res_136976;
                        
                        // futhark/microgpt.fut:4:11-25
                        
                        double zs_rhs_136978 = ((double *) mem_138962)[i_137469 * (int64_t) 27 + i_137465];
                        
                        // futhark/microgpt.fut:321:103-128
                        
                        double zs_res_136979 = 1.0 / zs_rhs_136978;
                        
                        // futhark/microgpt.fut:321:52-128
                        
                        double zt_res_136980 = zt_res_136977 * zs_res_136979;
                        
                        zp_lhs_125492 = zt_res_136980;
                    } else {
                        zp_lhs_125492 = 0.0;
                    }
                    // futhark/microgpt.fut:318:56-321:166
                    
                    double zp_res_125545 = zp_rhs_125488 + zp_lhs_125492;
                    
                    ((double *) mem_139012)[i_137461] = zp_res_125545;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139007, i_137465 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139012, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139001, i_137469 * (int64_t) 729, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint64_t *) mem_139007, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137477 = 0; i_137477 < (int64_t) 16; i_137477++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137473 = 0; i_137473 < (int64_t) 27; i_137473++) {
                double f_elem_125558 = ((double *) mem_138907)[i_137477 * (int64_t) 27 + i_137473];
                
                // futhark/microgpt.fut:322:110-135
                
                double neg_res_125563 = -f_elem_125558;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125564;
                double r_125566 = 0.0;
                
                for (int64_t i_125565 = 0; i_125565 < (int64_t) 27; i_125565++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_125567 = ((double *) mem_138889)[i_137477 * (int64_t) 27 + i_125565];
                    
                    // futhark/microgpt.fut:322:85-135
                    
                    double zp_res_125568 = neg_res_125563 + zp_lhs_125567;
                    
                    // futhark/microgpt.fut:322:78-135
                    
                    double exp_res_125569 = futrts_exp64(zp_res_125568);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125570 = ((double *) mem_139001)[i_137477 * (int64_t) 729 + i_137473 * (int64_t) 27 + i_125565];
                    
                    // futhark/microgpt.fut:322:78-170
                    
                    double zt_res_125571 = exp_res_125569 * zt_rhs_125570;
                    
                    // futhark/microgpt.fut:322:70-170
                    
                    double neg_res_125572 = -zt_res_125571;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125573 = r_125566 + neg_res_125572;
                    double r_tmp_140673 = zp_res_125573;
                    
                    r_125566 = r_tmp_140673;
                }
                defunc_0_lifted_lambda_res_125564 = r_125566;
                ((double *) mem_139040)[i_137473] = defunc_0_lifted_lambda_res_125564;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139035, i_137477 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139040, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137485 = 0; i_137485 < (int64_t) 16; i_137485++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137481 = 0; i_137481 < (int64_t) 27; i_137481++) {
                double f_elem_125630 = ((double *) mem_138889)[i_137485 * (int64_t) 27 + i_137481];
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125635;
                double r_125637 = 0.0;
                
                for (int64_t i_125636 = 0; i_125636 < (int64_t) 27; i_125636++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_125638 = ((double *) mem_138907)[i_137485 * (int64_t) 27 + i_125636];
                    
                    // futhark/microgpt.fut:325:89-113
                    
                    double neg_res_125639 = -neg_arg0_125638;
                    
                    // futhark/microgpt.fut:325:66-113
                    
                    double zp_res_125640 = f_elem_125630 + neg_res_125639;
                    
                    // futhark/microgpt.fut:325:59-113
                    
                    double exp_res_125641 = futrts_exp64(zp_res_125640);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125642 = ((double *) mem_139001)[i_137485 * (int64_t) 729 + i_125636 * (int64_t) 27 + i_137481];
                    
                    // futhark/microgpt.fut:325:59-146
                    
                    double zt_res_125643 = exp_res_125641 * zt_rhs_125642;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125644 = ((double *) mem_139035)[i_137485 * (int64_t) 27 + i_125636];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double neg_arg0_125645 = ((double *) mem_138906)[i_137485 * (int64_t) 27 + i_125636];
                    
                    // futhark/microgpt.fut:325:236-260
                    
                    double neg_res_125646 = -neg_arg0_125645;
                    
                    // futhark/microgpt.fut:325:213-260
                    
                    double zp_res_125647 = f_elem_125630 + neg_res_125646;
                    
                    // futhark/microgpt.fut:325:206-260
                    
                    double neg_res_125648 = -zp_res_125647;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_125649 = fmax64(0.0, neg_res_125648);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_125650 = fsignum64(max_res_125649);
                    
                    // futhark/microgpt.fut:325:187-263
                    
                    double neg_res_125651 = -sgn_res_125650;
                    
                    // futhark/microgpt.fut:325:178-264
                    
                    double zp_res_125652 = 1.0 + neg_res_125651;
                    
                    // futhark/microgpt.fut:325:154-264
                    
                    double zt_res_125653 = zt_lhs_125644 * zp_res_125652;
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125654 = ((double *) mem_138905)[i_137485 * (int64_t) 27 + i_125636];
                    
                    // futhark/microgpt.fut:325:173-290
                    
                    double zt_res_125655 = zt_res_125653 * zt_rhs_125654;
                    
                    // futhark/microgpt.fut:325:117-290
                    
                    double zp_res_125656 = zt_res_125643 + zt_res_125655;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125657 = r_125637 + zp_res_125656;
                    double r_tmp_140676 = zp_res_125657;
                    
                    r_125637 = r_tmp_140676;
                }
                defunc_0_lifted_lambda_res_125635 = r_125637;
                ((double *) mem_139056)[i_137481] = defunc_0_lifted_lambda_res_125635;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139051, i_137485 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139056, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137493 = 0; i_137493 < (int64_t) 16; i_137493++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137489 = 0; i_137489 < (int64_t) 16; i_137489++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125672;
                double r_125674 = 0.0;
                
                for (int64_t i_125673 = 0; i_125673 < (int64_t) 27; i_125673++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125675 = ((double *) mem_139051)[i_137493 * (int64_t) 27 + i_125673];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125676 = ((double *) mem_param_138295.mem)[i_125673 * (int64_t) 16 + i_137489];
                    
                    // futhark/microgpt.fut:326:67-111
                    
                    double zt_res_125677 = zt_lhs_125675 * zt_rhs_125676;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125678 = r_125674 + zt_res_125677;
                    double r_tmp_140679 = zp_res_125678;
                    
                    r_125674 = r_tmp_140679;
                }
                defunc_0_lifted_lambda_res_125672 = r_125674;
                ((double *) mem_139072)[i_137489] = defunc_0_lifted_lambda_res_125672;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139067, i_137493 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139072, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137501 = 0; i_137501 < (int64_t) 16; i_137501++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137497 = 0; i_137497 < (int64_t) 16; i_137497++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125693 = ((double *) mem_139067)[i_137501 * (int64_t) 16 + i_137497];
                
                ((double *) mem_139088)[i_137497] = lifted_lambda_res_125693;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139083, i_137501 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137514 = 0; i_137514 < (int64_t) 16; i_137514++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137507 = 0; i_137507 < (int64_t) 64; i_137507++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132947;
                double r_132949 = 0.0;
                
                for (int64_t i_132948 = 0; i_132948 < (int64_t) 16; i_132948++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132950 = ((double *) mem_139083)[i_137514 * (int64_t) 16 + i_132948];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132951 = ((double *) mem_param_138263.mem)[i_132948 * (int64_t) 64 + i_137507];
                    
                    // futhark/microgpt.fut:328:67-113
                    
                    double zt_res_132952 = zt_lhs_132950 * zt_rhs_132951;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132953 = r_132949 + zt_res_132952;
                    double r_tmp_140686 = zp_res_132953;
                    
                    r_132949 = r_tmp_140686;
                }
                defunc_0_lifted_lambda_res_132947 = r_132949;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132960;
                double r_132962 = 0.0;
                
                for (int64_t i_132961 = 0; i_132961 < (int64_t) 16; i_132961++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132963 = ((double *) mem_139083)[i_132961 * (int64_t) 16 + i_137514];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132964 = ((double *) mem_138841)[i_132961 * (int64_t) 64 + i_137507];
                    
                    // futhark/microgpt.fut:408:69-113
                    
                    double zt_res_132965 = zt_lhs_132963 * zt_rhs_132964;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132966 = r_132962 + zt_res_132965;
                    double r_tmp_140687 = zp_res_132966;
                    
                    r_132962 = r_tmp_140687;
                }
                defunc_0_lifted_lambda_res_132960 = r_132962;
                ((double *) mem_139109)[i_137507] = defunc_0_lifted_lambda_res_132960;
                ((double *) mem_139110)[i_137507] = defunc_0_lifted_lambda_res_132947;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139099, i_137514 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139109, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139100, i_137514 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139110, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137523 = 0; i_137523 < (int64_t) 16; i_137523++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137519 = 0; i_137519 < (int64_t) 64; i_137519++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_125729 = ((double *) mem_138825)[i_137523 * (int64_t) 64 + i_137519];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_125730 = fmax64(0.0, indicatorp_arg0_125729);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_125731 = fsignum64(max_res_125730);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_125732 = ((double *) mem_139100)[i_137523 * (int64_t) 64 + i_137519];
                
                // futhark/microgpt.fut:329:46-102
                
                double zt_res_125733 = sgn_res_125731 * zt_rhs_125732;
                
                ((double *) mem_139136)[i_137519] = zt_res_125733;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139131, i_137523 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139136, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137531 = 0; i_137531 < (int64_t) 16; i_137531++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137527 = 0; i_137527 < (int64_t) 16; i_137527++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_125748;
                double r_125750 = 0.0;
                
                for (int64_t i_125749 = 0; i_125749 < (int64_t) 64; i_125749++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_125751 = ((double *) mem_139131)[i_137531 * (int64_t) 64 + i_125749];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_125752 = ((double *) mem_param_138287.mem)[i_125749 * (int64_t) 16 + i_137527];
                    
                    // futhark/microgpt.fut:330:67-111
                    
                    double zt_res_125753 = zt_lhs_125751 * zt_rhs_125752;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_125754 = r_125750 + zt_res_125753;
                    double r_tmp_140692 = zp_res_125754;
                    
                    r_125750 = r_tmp_140692;
                }
                defunc_0_lifted_lambda_res_125748 = r_125750;
                ((double *) mem_139152)[i_137527] = defunc_0_lifted_lambda_res_125748;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139147, i_137531 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139152, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137537 = 0; i_137537 < (int64_t) 16; i_137537++) {
            // futhark/microgpt.fut:332:47-59
            
            double zp_lhs_127615 = ((double *) mem_138802)[i_137537];
            
            // futhark/microgpt.fut:332:47-87
            
            double zp_res_127616 = 1.0e-5 + zp_lhs_127615;
            
            // futhark/microgpt.fut:332:39-87
            
            double sqrt_res_127617 = futrts_sqrt64(zp_res_127616);
            
            // futhark/microgpt.fut:333:129-158
            
            double zt_res_127625 = sqrt_res_127617 * sqrt_res_127617;
            
            // futhark/microgpt.fut:333:120-158
            
            double zs_res_127626 = 1.0 / zt_res_127625;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127627;
            double r_127629 = 0.0;
            
            for (int64_t i_127628 = 0; i_127628 < (int64_t) 16; i_127628++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127630 = ((double *) mem_139147)[i_137537 * (int64_t) 16 + i_127628];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127631 = ((double *) mem_138786)[i_137537 * (int64_t) 16 + i_127628];
                
                // futhark/microgpt.fut:333:69-113
                
                double zt_res_127632 = zt_lhs_127630 * zt_rhs_127631;
                
                // futhark/microgpt.fut:333:90-158
                
                double zt_res_127633 = zs_res_127626 * zt_res_127632;
                
                // futhark/microgpt.fut:333:61-158
                
                double neg_res_127634 = -zt_res_127633;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127635 = r_127629 + neg_res_127634;
                double r_tmp_140695 = zp_res_127635;
                
                r_127629 = r_tmp_140695;
            }
            defunc_0_lifted_lambda_res_127627 = r_127629;
            ((double *) mem_139163)[i_137537] = defunc_0_lifted_lambda_res_127627;
            ((double *) mem_139164)[i_137537] = sqrt_res_127617;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137542 = 0; i_137542 < (int64_t) 16; i_137542++) {
            // futhark/microgpt.fut:334:39-51
            
            double zt_lhs_125806 = ((double *) mem_139163)[i_137542];
            
            // futhark/microgpt.fut:334:93-105
            
            double zp_lhs_125807 = ((double *) mem_138802)[i_137542];
            
            // futhark/microgpt.fut:334:93-133
            
            double zp_res_125808 = 1.0e-5 + zp_lhs_125807;
            
            // futhark/microgpt.fut:334:85-133
            
            double sqrt_res_125809 = futrts_sqrt64(zp_res_125808);
            
            // futhark/microgpt.fut:334:71-135
            
            double zt_res_125810 = 2.0 * sqrt_res_125809;
            
            // futhark/microgpt.fut:334:57-135
            
            double zs_res_125811 = 1.0 / zt_res_125810;
            
            // futhark/microgpt.fut:334:39-135
            
            double zt_res_125812 = zt_lhs_125806 * zs_res_125811;
            
            ((double *) mem_139177)[i_137542] = zt_res_125812;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137550 = 0; i_137550 < (int64_t) 16; i_137550++) {
            // futhark/microgpt.fut:335:98-110
            
            double zs_rhs_125820 = ((double *) mem_139164)[i_137550];
            
            // futhark/microgpt.fut:335:90-110
            
            double zs_res_125821 = 1.0 / zs_rhs_125820;
            
            // futhark/microgpt.fut:335:120-132
            
            double zs_lhs_125822 = ((double *) mem_139177)[i_137550];
            
            // futhark/microgpt.fut:335:120-147
            
            double zs_res_125823 = zs_lhs_125822 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137546 = 0; i_137546 < (int64_t) 16; i_137546++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_125830 = ((double *) mem_139083)[i_137550 * (int64_t) 16 + i_137546];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_125831 = ((double *) mem_139147)[i_137550 * (int64_t) 16 + i_137546];
                
                // futhark/microgpt.fut:335:64-110
                
                double zt_res_125832 = zs_res_125821 * zt_lhs_125831;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_125833 = ((double *) mem_138786)[i_137550 * (int64_t) 16 + i_137546];
                
                // futhark/microgpt.fut:335:133-172
                
                double zt_res_125834 = zs_res_125823 * zt_rhs_125833;
                
                // futhark/microgpt.fut:335:149-232
                
                double zp_res_125835 = zt_res_125834 + zt_res_125834;
                
                // futhark/microgpt.fut:335:85-232
                
                double zp_res_125836 = zt_res_125832 + zp_res_125835;
                
                // futhark/microgpt.fut:335:37-232
                
                double zp_res_125837 = zp_lhs_125830 + zp_res_125836;
                
                ((double *) mem_139189)[i_137546] = zp_res_125837;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139184, i_137550 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139189, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137558 = 0; i_137558 < (int64_t) 16; i_137558++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137554 = 0; i_137554 < (int64_t) 16; i_137554++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_125852 = ((double *) mem_139184)[i_137558 * (int64_t) 16 + i_137554];
                
                ((double *) mem_139205)[i_137554] = lifted_lambda_res_125852;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139200, i_137558 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139205, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137571 = 0; i_137571 < (int64_t) 16; i_137571++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137564 = 0; i_137564 < (int64_t) 16; i_137564++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_132990;
                double r_132992 = 0.0;
                
                for (int64_t i_132991 = 0; i_132991 < (int64_t) 16; i_132991++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_132993 = ((double *) mem_139200)[i_137571 * (int64_t) 16 + i_132991];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_132994 = ((double *) mem_param_138271.mem)[i_132991 * (int64_t) 16 + i_137564];
                    
                    // futhark/microgpt.fut:337:67-112
                    
                    double zt_res_132995 = zt_lhs_132993 * zt_rhs_132994;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_132996 = r_132992 + zt_res_132995;
                    double r_tmp_140705 = zp_res_132996;
                    
                    r_132992 = r_tmp_140705;
                }
                defunc_0_lifted_lambda_res_132990 = r_132992;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_133003;
                double r_133005 = 0.0;
                
                for (int64_t i_133004 = 0; i_133004 < (int64_t) 16; i_133004++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_133006 = ((double *) mem_139200)[i_133004 * (int64_t) 16 + i_137571];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_133007 = ((double *) mem_138754)[i_133004 * (int64_t) 16 + i_137564];
                    
                    // futhark/microgpt.fut:406:68-112
                    
                    double zt_res_133008 = zt_lhs_133006 * zt_rhs_133007;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_133009 = r_133005 + zt_res_133008;
                    double r_tmp_140706 = zp_res_133009;
                    
                    r_133005 = r_tmp_140706;
                }
                defunc_0_lifted_lambda_res_133003 = r_133005;
                ((double *) mem_139226)[i_137564] = defunc_0_lifted_lambda_res_133003;
                ((double *) mem_139227)[i_137564] = defunc_0_lifted_lambda_res_132990;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139216, i_137571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139226, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139217, i_137571 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139227, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137609 = 0; i_137609 < (int64_t) 4; i_137609++) {
            // futhark/microgpt.fut:338:74-77
            
            int64_t zp_lhs_129057 = mul64((int64_t) 4, i_137609);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137596 = 0; i_137596 < (int64_t) 16; i_137596++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137576 = 0; i_137576 < (int64_t) 4; i_137576++) {
                    // futhark/microgpt.fut:338:79-87
                    
                    int64_t tmp_133156 = add64(zp_lhs_129057, i_137576);
                    
                    // futhark/microgpt.fut:338:52-89
                    
                    bool x_133157 = sle64((int64_t) 0, tmp_133156);
                    
                    // futhark/microgpt.fut:338:52-89
                    
                    bool y_133158 = slt64(tmp_133156, (int64_t) 16);
                    
                    // futhark/microgpt.fut:338:52-89
                    
                    bool bounds_check_133159 = x_133157 && y_133158;
                    
                    // futhark/microgpt.fut:338:52-89
                    
                    bool index_certs_133160;
                    
                    if (!bounds_check_133159) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_133156, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:338:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:338:13-90\n   #9  futhark/microgpt.fut:578:5-76\n   #10 futhark/microgpt.fut:595:26-601:31\n   #11 futhark/microgpt.fut:629:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_133161 = ((double *) mem_139217)[i_137596 * (int64_t) 16 + tmp_133156];
                    
                    ((double *) mem_139292)[i_137576] = lifted_lambda_res_133161;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137584 = 0; i_137584 < (int64_t) 16; i_137584++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_133278 = ((double *) mem_138593)[i_137609 * (int64_t) 256 + i_137596 * (int64_t) 16 + i_137584];
                    
                    // futhark/microgpt.fut:340:59-101
                    
                    double zs_res_133279 = zs_lhs_133278 / 2.0;
                    double zp_rhs_133280 = ((double *) masks_mem_138257.mem)[step_124785 * (int64_t) 256 + i_137596 * (int64_t) 16 + i_137584];
                    
                    // futhark/microgpt.fut:340:88-127
                    
                    double zp_res_133281 = zs_res_133279 + zp_rhs_133280;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_133288 = ((double *) mem_138592)[i_137609 * (int64_t) 256 + i_137596 * (int64_t) 16 + i_137584];
                    
                    // futhark/microgpt.fut:347:59-101
                    
                    double zs_res_133289 = zs_lhs_133288 / 2.0;
                    
                    // futhark/microgpt.fut:347:88-127
                    
                    double zp_res_133291 = zp_rhs_133280 + zs_res_133289;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_133301 = ((double *) mem_138591)[i_137609 * (int64_t) 256 + i_137596 * (int64_t) 16 + i_137584];
                    
                    // futhark/microgpt.fut:361:59-101
                    
                    double zs_res_133302 = zs_lhs_133301 / 2.0;
                    
                    // futhark/microgpt.fut:361:88-127
                    
                    double zp_res_133304 = zp_rhs_133280 + zs_res_133302;
                    
                    ((double *) mem_139299)[i_137584] = zp_res_133304;
                    ((double *) mem_139300)[i_137584] = zp_res_133291;
                    ((double *) mem_139301)[i_137584] = zp_res_133281;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139272, i_137596 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139299, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139273, i_137596 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139300, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139274, i_137596 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139275, i_137596 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139292, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139248, i_137609 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139272, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139249, i_137609 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139273, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139250, i_137609 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139274, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139251, i_137609 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139275, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137680 = 0; i_137680 < (int64_t) 4; i_137680++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137652 = 0; i_137652 < (int64_t) 16; i_137652++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_137003;
                double defunc_0_reduce_res_137004;
                double defunc_0_reduce_res_137005;
                double defunc_0_reduce_res_137006;
                double defunc_0_reduce_res_137007;
                double redout_137616;
                double redout_137617;
                double redout_137618;
                double redout_137619;
                double redout_137620;
                
                redout_137616 = -INFINITY;
                redout_137617 = -INFINITY;
                redout_137618 = -INFINITY;
                redout_137619 = -INFINITY;
                redout_137620 = -INFINITY;
                for (int64_t i_137623 = 0; i_137623 < (int64_t) 16; i_137623++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_135140 = ((double *) mem_139250)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_137623];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_135151;
                    double r_135153 = 0.0;
                    
                    for (int64_t i_135152 = 0; i_135152 < (int64_t) 4; i_135152++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_135154 = ((double *) mem_139251)[i_137680 * (int64_t) 64 + i_137652 * (int64_t) 4 + i_135152];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_135155 = ((double *) mem_138510)[i_137680 * (int64_t) 64 + i_137623 * (int64_t) 4 + i_135152];
                        
                        // futhark/microgpt.fut:348:79-139
                        
                        double zt_res_135156 = zt_lhs_135154 * zt_rhs_135155;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_135157 = r_135153 + zt_res_135156;
                        double r_tmp_140744 = zp_res_135157;
                        
                        r_135153 = r_tmp_140744;
                    }
                    defunc_0_lifted_lambda_res_135151 = r_135153;
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_135165 = ((double *) mem_139249)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_137623];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_135213;
                    double r_135215 = 0.0;
                    
                    for (int64_t i_135214 = 0; i_135214 < (int64_t) 4; i_135214++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_135216 = ((double *) mem_139251)[i_137680 * (int64_t) 64 + i_137652 * (int64_t) 4 + i_135214];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_135217 = ((double *) mem_138510)[i_137680 * (int64_t) 64 + i_137623 * (int64_t) 4 + i_135214];
                        
                        // futhark/microgpt.fut:362:79-139
                        
                        double zt_res_135218 = zt_lhs_135216 * zt_rhs_135217;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_135219 = r_135215 + zt_res_135218;
                        double r_tmp_140745 = zp_res_135219;
                        
                        r_135215 = r_tmp_140745;
                    }
                    defunc_0_lifted_lambda_res_135213 = r_135215;
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_135230 = ((double *) mem_139248)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_137623];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_134268 = fmax64(lifted_lambda_res_135140, redout_137616);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_134336 = fmax64(lifted_lambda_res_135165, redout_137617);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_134361 = fmax64(lifted_lambda_res_135165, redout_137618);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_134442 = fmax64(lifted_lambda_res_135230, redout_137619);
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_134475 = fmax64(lifted_lambda_res_135230, redout_137620);
                    
                    ((double *) mem_139443)[i_137623] = defunc_0_lifted_lambda_res_135213;
                    ((double *) mem_139444)[i_137623] = defunc_0_lifted_lambda_res_135151;
                    
                    double redout_tmp_140737 = max_res_134268;
                    double redout_tmp_140738 = max_res_134336;
                    double redout_tmp_140739 = max_res_134361;
                    double redout_tmp_140740 = max_res_134442;
                    double redout_tmp_140741 = max_res_134475;
                    
                    redout_137616 = redout_tmp_140737;
                    redout_137617 = redout_tmp_140738;
                    redout_137618 = redout_tmp_140739;
                    redout_137619 = redout_tmp_140740;
                    redout_137620 = redout_tmp_140741;
                }
                defunc_0_reduce_res_137003 = redout_137616;
                defunc_0_reduce_res_137004 = redout_137617;
                defunc_0_reduce_res_137005 = redout_137618;
                defunc_0_reduce_res_137006 = redout_137619;
                defunc_0_reduce_res_137007 = redout_137620;
                // futhark/microgpt.fut:342:80-90
                
                double neg_res_134269 = -defunc_0_reduce_res_137003;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137628 = 0; i_137628 < (int64_t) 16; i_137628++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_134276 = ((double *) mem_139250)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_137628];
                    
                    // futhark/microgpt.fut:342:46-90
                    
                    double zp_res_134277 = neg_res_134269 + zp_lhs_134276;
                    
                    // futhark/microgpt.fut:342:39-90
                    
                    double exp_res_134278 = futrts_exp64(zp_res_134277);
                    
                    ((double *) mem_139457)[i_137628] = exp_res_134278;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_134280;
                double r_134282 = 0.0;
                
                for (int64_t i_134281 = 0; i_134281 < (int64_t) 16; i_134281++) {
                    // futhark/microgpt.fut:343:38-50
                    
                    double lifted_lambda_res_134283 = ((double *) mem_139457)[i_134281];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_134284 = r_134282 + lifted_lambda_res_134283;
                    double r_tmp_140747 = zp_res_134284;
                    
                    r_134282 = r_tmp_140747;
                }
                defunc_0_lifted_lambda_res_134280 = r_134282;
                // futhark/microgpt.fut:344:23-35
                
                double zs_res_134285 = 1.0 / defunc_0_lifted_lambda_res_134280;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137632 = 0; i_137632 < (int64_t) 16; i_137632++) {
                    // futhark/microgpt.fut:344:5-17
                    
                    double zt_lhs_134292 = ((double *) mem_139457)[i_137632];
                    
                    // futhark/microgpt.fut:344:5-35
                    
                    double zt_res_134293 = zs_res_134285 * zt_lhs_134292;
                    
                    ((double *) mem_139464)[i_137632] = zt_res_134293;
                }
                // futhark/microgpt.fut:356:148-174
                
                double neg_res_134369 = -defunc_0_reduce_res_137005;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_134370;
                double r_134372 = 0.0;
                
                for (int64_t i_134371 = 0; i_134371 < (int64_t) 16; i_134371++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_134373 = ((double *) mem_139249)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_134371];
                    
                    // futhark/microgpt.fut:356:114-174
                    
                    double zp_res_134374 = neg_res_134369 + zp_lhs_134373;
                    
                    // futhark/microgpt.fut:356:107-174
                    
                    double neg_res_134375 = -zp_res_134374;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_134376 = fmax64(0.0, neg_res_134375);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_134377 = fsignum64(max_res_134376);
                    
                    // futhark/microgpt.fut:356:88-177
                    
                    double neg_res_134378 = -sgn_res_134377;
                    
                    // futhark/microgpt.fut:356:79-178
                    
                    double zp_res_134379 = 1.0 + neg_res_134378;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_134380 = r_134372 + zp_res_134379;
                    double r_tmp_140749 = zp_res_134380;
                    
                    r_134372 = r_tmp_140749;
                }
                defunc_0_lifted_lambda_res_134370 = r_134372;
                // futhark/microgpt.fut:356:48-181
                
                double zs_res_134381 = 1.0 / defunc_0_lifted_lambda_res_134370;
                
                // futhark/microgpt.fut:370:148-174
                
                double neg_res_134483 = -defunc_0_reduce_res_137007;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_134484;
                double r_134486 = 0.0;
                
                for (int64_t i_134485 = 0; i_134485 < (int64_t) 16; i_134485++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_134487 = ((double *) mem_139248)[i_137680 * (int64_t) 256 + i_137652 * (int64_t) 16 + i_134485];
                    
                    // futhark/microgpt.fut:370:114-174
                    
                    double zp_res_134488 = neg_res_134483 + zp_lhs_134487;
                    
                    // futhark/microgpt.fut:370:107-174
                    
                    double neg_res_134489 = -zp_res_134488;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_134490 = fmax64(0.0, neg_res_134489);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_134491 = fsignum64(max_res_134490);
                    
                    // futhark/microgpt.fut:370:88-177
                    
                    double neg_res_134492 = -sgn_res_134491;
                    
                    // futhark/microgpt.fut:370:79-178
                    
                    double zp_res_134493 = 1.0 + neg_res_134492;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_134494 = r_134486 + zp_res_134493;
                    double r_tmp_140750 = zp_res_134494;
                    
                    r_134486 = r_tmp_140750;
                }
                defunc_0_lifted_lambda_res_134484 = r_134486;
                // futhark/microgpt.fut:370:48-181
                
                double zs_res_134495 = 1.0 / defunc_0_lifted_lambda_res_134484;
                
                ((double *) mem_139404)[i_137652] = zs_res_134495;
                ((double *) mem_139405)[i_137652] = defunc_0_reduce_res_137007;
                ((double *) mem_139406)[i_137652] = defunc_0_reduce_res_137006;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139407, i_137652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139443, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                ((double *) mem_139408)[i_137652] = zs_res_134381;
                ((double *) mem_139409)[i_137652] = defunc_0_reduce_res_137005;
                ((double *) mem_139410)[i_137652] = defunc_0_reduce_res_137004;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139411, i_137652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139412, i_137652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139464, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139356, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139404, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139357, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139358, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139406, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139359, i_137680 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139407, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139360, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139408, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139361, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139409, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139362, i_137680 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139410, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139363, i_137680 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139411, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139364, i_137680 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139412, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137717 = 0; i_137717 < (int64_t) 4; i_137717++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137707 = 0; i_137707 < (int64_t) 16; i_137707++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137692 = 0; i_137692 < (int64_t) 4; i_137692++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_135384;
                    double r_135386 = 0.0;
                    
                    for (int64_t i_135385 = 0; i_135385 < (int64_t) 16; i_135385++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_135387 = ((double *) mem_139251)[i_137717 * (int64_t) 64 + i_135385 * (int64_t) 4 + i_137692];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_135388 = ((double *) mem_139364)[i_137717 * (int64_t) 256 + i_135385 * (int64_t) 16 + i_137707];
                        
                        // futhark/microgpt.fut:345:67-128
                        
                        double zt_res_135389 = zt_lhs_135387 * zt_rhs_135388;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_135390 = r_135386 + zt_res_135389;
                        double r_tmp_140758 = zp_res_135390;
                        
                        r_135386 = r_tmp_140758;
                    }
                    defunc_0_lifted_lambda_res_135384 = r_135386;
                    ((double *) mem_139573)[i_137692] = defunc_0_lifted_lambda_res_135384;
                }
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135398 = ((double *) mem_139362)[i_137717 * (int64_t) 16 + i_137707];
                
                // futhark/microgpt.fut:350:99-125
                
                double neg_res_135399 = -neg_arg0_135398;
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135419 = ((double *) mem_139358)[i_137717 * (int64_t) 16 + i_137707];
                
                // futhark/microgpt.fut:364:99-125
                
                double neg_res_135420 = -neg_arg0_135419;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137698 = 0; i_137698 < (int64_t) 16; i_137698++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_135449 = ((double *) mem_139249)[i_137717 * (int64_t) 256 + i_137707 * (int64_t) 16 + i_137698];
                    
                    // futhark/microgpt.fut:350:65-125
                    
                    double zp_res_135450 = neg_res_135399 + zp_lhs_135449;
                    
                    // futhark/microgpt.fut:350:58-125
                    
                    double exp_res_135451 = futrts_exp64(zp_res_135450);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_135458 = ((double *) mem_139248)[i_137717 * (int64_t) 256 + i_137707 * (int64_t) 16 + i_137698];
                    
                    // futhark/microgpt.fut:364:65-125
                    
                    double zp_res_135459 = neg_res_135420 + zp_lhs_135458;
                    
                    // futhark/microgpt.fut:364:58-125
                    
                    double exp_res_135460 = futrts_exp64(zp_res_135459);
                    
                    ((double *) mem_139580)[i_137698] = exp_res_135460;
                    ((double *) mem_139581)[i_137698] = exp_res_135451;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139558, i_137707 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139580, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139559, i_137707 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139581, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139560, i_137707 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139573, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139540, i_137717 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139558, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139541, i_137717 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139559, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139542, i_137717 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139560, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137742 = 0; i_137742 < (int64_t) 4; i_137742++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137729 = 0; i_137729 < (int64_t) 16; i_137729++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135576;
                double r_135578 = 0.0;
                
                for (int64_t i_135577 = 0; i_135577 < (int64_t) 16; i_135577++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_135579 = ((double *) mem_139541)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135577];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135580 = r_135578 + lifted_lambda_res_135579;
                    double r_tmp_140769 = zp_res_135580;
                    
                    r_135578 = r_tmp_140769;
                }
                defunc_0_lifted_lambda_res_135576 = r_135578;
                // futhark/microgpt.fut:352:155-200
                
                double zt_res_135588 = defunc_0_lifted_lambda_res_135576 * defunc_0_lifted_lambda_res_135576;
                
                // futhark/microgpt.fut:352:146-200
                
                double zs_res_135589 = 1.0 / zt_res_135588;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135590;
                double r_135592 = 0.0;
                
                for (int64_t i_135591 = 0; i_135591 < (int64_t) 16; i_135591++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_135593 = ((double *) mem_139363)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135591];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_135594 = ((double *) mem_139541)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135591];
                    
                    // futhark/microgpt.fut:352:78-139
                    
                    double zt_res_135595 = zt_lhs_135593 * zt_rhs_135594;
                    
                    // futhark/microgpt.fut:352:107-200
                    
                    double zt_res_135596 = zs_res_135589 * zt_res_135595;
                    
                    // futhark/microgpt.fut:352:70-200
                    
                    double neg_res_135597 = -zt_res_135596;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135598 = r_135592 + neg_res_135597;
                    double r_tmp_140770 = zp_res_135598;
                    
                    r_135592 = r_tmp_140770;
                }
                defunc_0_lifted_lambda_res_135590 = r_135592;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135609;
                double r_135611 = 0.0;
                
                for (int64_t i_135610 = 0; i_135610 < (int64_t) 16; i_135610++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_135612 = ((double *) mem_139540)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135610];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135613 = r_135611 + lifted_lambda_res_135612;
                    double r_tmp_140771 = zp_res_135613;
                    
                    r_135611 = r_tmp_140771;
                }
                defunc_0_lifted_lambda_res_135609 = r_135611;
                // futhark/microgpt.fut:366:155-200
                
                double zt_res_135621 = defunc_0_lifted_lambda_res_135609 * defunc_0_lifted_lambda_res_135609;
                
                // futhark/microgpt.fut:366:146-200
                
                double zs_res_135622 = 1.0 / zt_res_135621;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135623;
                double r_135625 = 0.0;
                
                for (int64_t i_135624 = 0; i_135624 < (int64_t) 16; i_135624++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_135626 = ((double *) mem_139359)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135624];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_135627 = ((double *) mem_139540)[i_137742 * (int64_t) 256 + i_137729 * (int64_t) 16 + i_135624];
                    
                    // futhark/microgpt.fut:366:78-139
                    
                    double zt_res_135628 = zt_lhs_135626 * zt_rhs_135627;
                    
                    // futhark/microgpt.fut:366:107-200
                    
                    double zt_res_135629 = zs_res_135622 * zt_res_135628;
                    
                    // futhark/microgpt.fut:366:70-200
                    
                    double neg_res_135630 = -zt_res_135629;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135631 = r_135625 + neg_res_135630;
                    double r_tmp_140772 = zp_res_135631;
                    
                    r_135625 = r_tmp_140772;
                }
                defunc_0_lifted_lambda_res_135623 = r_135625;
                ((double *) mem_139641)[i_137729] = defunc_0_lifted_lambda_res_135623;
                ((double *) mem_139642)[i_137729] = defunc_0_lifted_lambda_res_135609;
                ((double *) mem_139643)[i_137729] = defunc_0_lifted_lambda_res_135590;
                ((double *) mem_139644)[i_137729] = defunc_0_lifted_lambda_res_135576;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139621, i_137742 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139641, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139622, i_137742 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139642, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139623, i_137742 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139624, i_137742 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139644, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137765 = 0; i_137765 < (int64_t) 4; i_137765++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137758 = 0; i_137758 < (int64_t) 16; i_137758++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_135657 = ((double *) mem_139624)[i_137765 * (int64_t) 16 + i_137758];
                
                // futhark/microgpt.fut:353:93-121
                
                double zs_res_135658 = 1.0 / zs_rhs_135657;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_135659 = ((double *) mem_139623)[i_137765 * (int64_t) 16 + i_137758];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_135678 = ((double *) mem_139621)[i_137765 * (int64_t) 16 + i_137758];
                
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_135676 = ((double *) mem_139622)[i_137765 * (int64_t) 16 + i_137758];
                
                // futhark/microgpt.fut:367:93-121
                
                double zs_res_135677 = 1.0 / zs_rhs_135676;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137751 = 0; i_137751 < (int64_t) 16; i_137751++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_135706 = ((double *) mem_139363)[i_137765 * (int64_t) 256 + i_137758 * (int64_t) 16 + i_137751];
                    
                    // futhark/microgpt.fut:353:59-121
                    
                    double zt_res_135707 = zs_res_135658 * zt_lhs_135706;
                    
                    // futhark/microgpt.fut:353:88-148
                    
                    double zp_res_135708 = zp_rhs_135659 + zt_res_135707;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_135715 = ((double *) mem_139359)[i_137765 * (int64_t) 256 + i_137758 * (int64_t) 16 + i_137751];
                    
                    // futhark/microgpt.fut:367:59-121
                    
                    double zt_res_135716 = zs_res_135677 * zt_lhs_135715;
                    
                    // futhark/microgpt.fut:367:88-148
                    
                    double zp_res_135717 = zp_rhs_135678 + zt_res_135716;
                    
                    ((double *) mem_139707)[i_137751] = zp_res_135717;
                    ((double *) mem_139708)[i_137751] = zp_res_135708;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139697, i_137758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139707, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139698, i_137758 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139708, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139685, i_137765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139697, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139686, i_137765 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139698, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137779 = 0; i_137779 < (int64_t) 4; i_137779++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137772 = 0; i_137772 < (int64_t) 16; i_137772++) {
                double f_elem_135737 = ((double *) mem_139362)[i_137779 * (int64_t) 16 + i_137772];
                double f_elem_135739 = ((double *) mem_139358)[i_137779 * (int64_t) 16 + i_137772];
                
                // futhark/microgpt.fut:354:119-145
                
                double neg_res_135744 = -f_elem_135737;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135745;
                double r_135747 = 0.0;
                
                for (int64_t i_135746 = 0; i_135746 < (int64_t) 16; i_135746++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_135748 = ((double *) mem_139249)[i_137779 * (int64_t) 256 + i_137772 * (int64_t) 16 + i_135746];
                    
                    // futhark/microgpt.fut:354:85-145
                    
                    double zp_res_135749 = neg_res_135744 + zp_lhs_135748;
                    
                    // futhark/microgpt.fut:354:78-145
                    
                    double exp_res_135750 = futrts_exp64(zp_res_135749);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_135751 = ((double *) mem_139686)[i_137779 * (int64_t) 256 + i_137772 * (int64_t) 16 + i_135746];
                    
                    // futhark/microgpt.fut:354:78-181
                    
                    double zt_res_135752 = exp_res_135750 * zt_rhs_135751;
                    
                    // futhark/microgpt.fut:354:70-181
                    
                    double neg_res_135753 = -zt_res_135752;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135754 = r_135747 + neg_res_135753;
                    double r_tmp_140783 = zp_res_135754;
                    
                    r_135747 = r_tmp_140783;
                }
                defunc_0_lifted_lambda_res_135745 = r_135747;
                // futhark/microgpt.fut:368:119-145
                
                double neg_res_135762 = -f_elem_135739;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_135763;
                double r_135765 = 0.0;
                
                for (int64_t i_135764 = 0; i_135764 < (int64_t) 16; i_135764++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_135766 = ((double *) mem_139248)[i_137779 * (int64_t) 256 + i_137772 * (int64_t) 16 + i_135764];
                    
                    // futhark/microgpt.fut:368:85-145
                    
                    double zp_res_135767 = neg_res_135762 + zp_lhs_135766;
                    
                    // futhark/microgpt.fut:368:78-145
                    
                    double exp_res_135768 = futrts_exp64(zp_res_135767);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_135769 = ((double *) mem_139685)[i_137779 * (int64_t) 256 + i_137772 * (int64_t) 16 + i_135764];
                    
                    // futhark/microgpt.fut:368:78-181
                    
                    double zt_res_135770 = exp_res_135768 * zt_rhs_135769;
                    
                    // futhark/microgpt.fut:368:70-181
                    
                    double neg_res_135771 = -zt_res_135770;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_135772 = r_135765 + neg_res_135771;
                    double r_tmp_140784 = zp_res_135772;
                    
                    r_135765 = r_tmp_140784;
                }
                defunc_0_lifted_lambda_res_135763 = r_135765;
                ((double *) mem_139749)[i_137772] = defunc_0_lifted_lambda_res_135763;
                ((double *) mem_139750)[i_137772] = defunc_0_lifted_lambda_res_135745;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139739, i_137779 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139749, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139740, i_137779 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139750, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137800 = 0; i_137800 < (int64_t) 4; i_137800++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137793 = 0; i_137793 < (int64_t) 16; i_137793++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135792 = ((double *) mem_139362)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:357:101-127
                
                double neg_res_135793 = -neg_arg0_135792;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_135794 = ((double *) mem_139740)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135795 = ((double *) mem_139361)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:357:266-292
                
                double neg_res_135796 = -neg_arg0_135795;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_135797 = ((double *) mem_139360)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_135830 = ((double *) mem_139356)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135828 = ((double *) mem_139357)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:371:266-292
                
                double neg_res_135829 = -neg_arg0_135828;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_135827 = ((double *) mem_139739)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_135825 = ((double *) mem_139358)[i_137800 * (int64_t) 16 + i_137793];
                
                // futhark/microgpt.fut:371:101-127
                
                double neg_res_135826 = -neg_arg0_135825;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137786 = 0; i_137786 < (int64_t) 16; i_137786++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_135869 = ((double *) mem_139249)[i_137800 * (int64_t) 256 + i_137793 * (int64_t) 16 + i_137786];
                    
                    // futhark/microgpt.fut:357:67-127
                    
                    double zp_res_135870 = neg_res_135793 + zp_lhs_135869;
                    
                    // futhark/microgpt.fut:357:60-127
                    
                    double exp_res_135871 = futrts_exp64(zp_res_135870);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_135872 = ((double *) mem_139686)[i_137800 * (int64_t) 256 + i_137793 * (int64_t) 16 + i_137786];
                    
                    // futhark/microgpt.fut:357:60-163
                    
                    double zt_res_135873 = exp_res_135871 * zt_rhs_135872;
                    
                    // futhark/microgpt.fut:357:232-292
                    
                    double zp_res_135874 = neg_res_135796 + zp_lhs_135869;
                    
                    // futhark/microgpt.fut:357:225-292
                    
                    double neg_res_135875 = -zp_res_135874;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_135876 = fmax64(0.0, neg_res_135875);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_135877 = fsignum64(max_res_135876);
                    
                    // futhark/microgpt.fut:357:206-295
                    
                    double neg_res_135878 = -sgn_res_135877;
                    
                    // futhark/microgpt.fut:357:197-296
                    
                    double zp_res_135879 = 1.0 + neg_res_135878;
                    
                    // futhark/microgpt.fut:357:171-296
                    
                    double zt_res_135880 = zt_lhs_135794 * zp_res_135879;
                    
                    // futhark/microgpt.fut:357:192-324
                    
                    double zt_res_135881 = zt_rhs_135797 * zt_res_135880;
                    
                    // futhark/microgpt.fut:357:131-324
                    
                    double zp_res_135882 = zt_res_135873 + zt_res_135881;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_135889 = ((double *) mem_139248)[i_137800 * (int64_t) 256 + i_137793 * (int64_t) 16 + i_137786];
                    
                    // futhark/microgpt.fut:371:67-127
                    
                    double zp_res_135890 = neg_res_135826 + zp_lhs_135889;
                    
                    // futhark/microgpt.fut:371:60-127
                    
                    double exp_res_135891 = futrts_exp64(zp_res_135890);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_135892 = ((double *) mem_139685)[i_137800 * (int64_t) 256 + i_137793 * (int64_t) 16 + i_137786];
                    
                    // futhark/microgpt.fut:371:60-163
                    
                    double zt_res_135893 = exp_res_135891 * zt_rhs_135892;
                    
                    // futhark/microgpt.fut:371:232-292
                    
                    double zp_res_135894 = neg_res_135829 + zp_lhs_135889;
                    
                    // futhark/microgpt.fut:371:225-292
                    
                    double neg_res_135895 = -zp_res_135894;
                    
                    // futhark/microgpt.fut:110:42-54
                    
                    double max_res_135896 = fmax64(0.0, neg_res_135895);
                    
                    // futhark/microgpt.fut:110:35-54
                    
                    double sgn_res_135897 = fsignum64(max_res_135896);
                    
                    // futhark/microgpt.fut:371:206-295
                    
                    double neg_res_135898 = -sgn_res_135897;
                    
                    // futhark/microgpt.fut:371:197-296
                    
                    double zp_res_135899 = 1.0 + neg_res_135898;
                    
                    // futhark/microgpt.fut:371:171-296
                    
                    double zt_res_135900 = zt_lhs_135827 * zp_res_135899;
                    
                    // futhark/microgpt.fut:371:192-324
                    
                    double zt_res_135901 = zt_rhs_135830 * zt_res_135900;
                    
                    // futhark/microgpt.fut:371:131-324
                    
                    double zp_res_135902 = zt_res_135893 + zt_res_135901;
                    
                    ((double *) mem_139793)[i_137786] = zp_res_135902;
                    ((double *) mem_139794)[i_137786] = zp_res_135882;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139783, i_137793 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139793, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139784, i_137793 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139794, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139771, i_137800 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139783, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139772, i_137800 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139784, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137821 = 0; i_137821 < (int64_t) 4; i_137821++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137814 = 0; i_137814 < (int64_t) 16; i_137814++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137807 = 0; i_137807 < (int64_t) 16; i_137807++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_135967 = ((double *) mem_139772)[i_137821 * (int64_t) 256 + i_137814 * (int64_t) 16 + i_137807];
                    
                    // futhark/microgpt.fut:358:58-100
                    
                    double zs_res_135968 = zs_lhs_135967 / 2.0;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_135975 = ((double *) mem_139771)[i_137821 * (int64_t) 256 + i_137814 * (int64_t) 16 + i_137807];
                    
                    // futhark/microgpt.fut:372:58-100
                    
                    double zs_res_135976 = zs_lhs_135975 / 2.0;
                    
                    ((double *) mem_139847)[i_137807] = zs_res_135976;
                    ((double *) mem_139848)[i_137807] = zs_res_135968;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139837, i_137814 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139847, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139838, i_137814 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139848, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139825, i_137821 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139837, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139826, i_137821 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139838, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137842 = 0; i_137842 < (int64_t) 4; i_137842++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137835 = 0; i_137835 < (int64_t) 16; i_137835++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_137828 = 0; i_137828 < (int64_t) 4; i_137828++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_136051;
                    double r_136053 = 0.0;
                    
                    for (int64_t i_136052 = 0; i_136052 < (int64_t) 16; i_136052++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_136054 = ((double *) mem_139826)[i_137842 * (int64_t) 256 + i_136052 * (int64_t) 16 + i_137835];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_136055 = ((double *) mem_138512)[i_137842 * (int64_t) 64 + i_136052 * (int64_t) 4 + i_137828];
                        
                        // futhark/microgpt.fut:359:67-127
                        
                        double zt_res_136056 = zt_lhs_136054 * zt_rhs_136055;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_136057 = r_136053 + zt_res_136056;
                        double r_tmp_140803 = zp_res_136057;
                        
                        r_136053 = r_tmp_140803;
                    }
                    defunc_0_lifted_lambda_res_136051 = r_136053;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_136064;
                    double r_136066 = 0.0;
                    
                    for (int64_t i_136065 = 0; i_136065 < (int64_t) 16; i_136065++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_136067 = ((double *) mem_139825)[i_137842 * (int64_t) 256 + i_137835 * (int64_t) 16 + i_136065];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_136068 = ((double *) mem_138511)[i_137842 * (int64_t) 64 + i_136065 * (int64_t) 4 + i_137828];
                        
                        // futhark/microgpt.fut:373:67-127
                        
                        double zt_res_136069 = zt_lhs_136067 * zt_rhs_136068;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_136070 = r_136066 + zt_res_136069;
                        double r_tmp_140804 = zp_res_136070;
                        
                        r_136066 = r_tmp_140804;
                    }
                    defunc_0_lifted_lambda_res_136064 = r_136066;
                    ((double *) mem_139901)[i_137828] = defunc_0_lifted_lambda_res_136064;
                    ((double *) mem_139902)[i_137828] = defunc_0_lifted_lambda_res_136051;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139891, i_137835 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_139892, i_137835 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139902, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139879, i_137842 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139891, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_139880, i_137842 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_139892, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137861 = 0; i_137861 < (int64_t) 16; i_137861++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137851 = 0; i_137851 < (int64_t) 16; i_137851++) {
                // futhark/microgpt.fut:374:57-60
                
                int64_t tmp_136133 = sdiv64(i_137851, (int64_t) 4);
                
                // futhark/microgpt.fut:374:44-62
                
                bool x_136134 = sle64((int64_t) 0, tmp_136133);
                
                // futhark/microgpt.fut:374:44-62
                
                bool y_136135 = slt64(tmp_136133, (int64_t) 4);
                
                // futhark/microgpt.fut:374:44-62
                
                bool bounds_check_136136 = x_136134 && y_136135;
                
                // futhark/microgpt.fut:374:44-62
                
                bool index_certs_136137;
                
                if (!bounds_check_136136) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_136133, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:374:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:374:13-85\n   #6  futhark/microgpt.fut:578:5-76\n   #7  futhark/microgpt.fut:595:26-601:31\n   #8  futhark/microgpt.fut:629:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:374:79-82
                
                int64_t tmp_136138 = smod64(i_137851, (int64_t) 4);
                
                // futhark/microgpt.fut:374:44-84
                
                bool x_136139 = sle64((int64_t) 0, tmp_136138);
                
                // futhark/microgpt.fut:374:44-84
                
                bool y_136140 = slt64(tmp_136138, (int64_t) 4);
                
                // futhark/microgpt.fut:374:44-84
                
                bool bounds_check_136141 = x_136139 && y_136140;
                
                // futhark/microgpt.fut:374:44-84
                
                bool index_certs_136142;
                
                if (!bounds_check_136141) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_136138, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:374:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:374:13-85\n   #6  futhark/microgpt.fut:578:5-76\n   #7  futhark/microgpt.fut:595:26-601:31\n   #8  futhark/microgpt.fut:629:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136143 = ((double *) mem_139542)[tmp_136133 * (int64_t) 64 + i_137861 * (int64_t) 4 + tmp_136138];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136156 = ((double *) mem_139880)[tmp_136133 * (int64_t) 64 + i_137861 * (int64_t) 4 + tmp_136138];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136172 = ((double *) mem_139879)[tmp_136133 * (int64_t) 64 + i_137861 * (int64_t) 4 + tmp_136138];
                
                ((double *) mem_139948)[i_137851] = lifted_lambda_res_136172;
                ((double *) mem_139949)[i_137851] = lifted_lambda_res_136156;
                ((double *) mem_139950)[i_137851] = lifted_lambda_res_136143;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139933, i_137861 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139948, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139934, i_137861 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139949, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139935, i_137861 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_139950, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137886 = 0; i_137886 < (int64_t) 16; i_137886++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137873 = 0; i_137873 < (int64_t) 16; i_137873++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136335;
                double r_136337 = 0.0;
                
                for (int64_t i_136336 = 0; i_136336 < (int64_t) 16; i_136336++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136338 = ((double *) mem_139935)[i_137886 * (int64_t) 16 + i_136336];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136339 = ((double *) mem_param_138291.mem)[i_136336 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:377:69-114
                    
                    double zt_res_136340 = zt_lhs_136338 * zt_rhs_136339;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136341 = r_136337 + zt_res_136340;
                    double r_tmp_140819 = zp_res_136341;
                    
                    r_136337 = r_tmp_140819;
                }
                defunc_0_lifted_lambda_res_136335 = r_136337;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136342;
                double r_136344 = 0.0;
                
                for (int64_t i_136343 = 0; i_136343 < (int64_t) 16; i_136343++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136345 = ((double *) mem_139934)[i_137886 * (int64_t) 16 + i_136343];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136346 = ((double *) mem_param_138267.mem)[i_136343 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:377:145-190
                    
                    double zt_res_136347 = zt_lhs_136345 * zt_rhs_136346;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136348 = r_136344 + zt_res_136347;
                    double r_tmp_140820 = zp_res_136348;
                    
                    r_136344 = r_tmp_140820;
                }
                defunc_0_lifted_lambda_res_136342 = r_136344;
                // futhark/microgpt.fut:377:47-192
                
                double zp_res_136349 = defunc_0_lifted_lambda_res_136335 + defunc_0_lifted_lambda_res_136342;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136350;
                double r_136352 = 0.0;
                
                for (int64_t i_136351 = 0; i_136351 < (int64_t) 16; i_136351++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136353 = ((double *) mem_139933)[i_137886 * (int64_t) 16 + i_136351];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136354 = ((double *) mem_param_138279.mem)[i_136351 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:377:222-267
                    
                    double zt_res_136355 = zt_lhs_136353 * zt_rhs_136354;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136356 = r_136352 + zt_res_136355;
                    double r_tmp_140821 = zp_res_136356;
                    
                    r_136352 = r_tmp_140821;
                }
                defunc_0_lifted_lambda_res_136350 = r_136352;
                // futhark/microgpt.fut:377:118-269
                
                double zp_res_136357 = zp_res_136349 + defunc_0_lifted_lambda_res_136350;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136364;
                double r_136366 = 0.0;
                
                for (int64_t i_136365 = 0; i_136365 < (int64_t) 16; i_136365++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136367 = ((double *) mem_139933)[i_136365 * (int64_t) 16 + i_137886];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136368 = ((double *) mem_138440)[i_136365 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:403:68-111
                    
                    double zt_res_136369 = zt_lhs_136367 * zt_rhs_136368;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136370 = r_136366 + zt_res_136369;
                    double r_tmp_140822 = zp_res_136370;
                    
                    r_136366 = r_tmp_140822;
                }
                defunc_0_lifted_lambda_res_136364 = r_136366;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136380;
                double r_136382 = 0.0;
                
                for (int64_t i_136381 = 0; i_136381 < (int64_t) 16; i_136381++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136383 = ((double *) mem_139934)[i_136381 * (int64_t) 16 + i_137886];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136384 = ((double *) mem_138440)[i_136381 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:404:68-111
                    
                    double zt_res_136385 = zt_lhs_136383 * zt_rhs_136384;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136386 = r_136382 + zt_res_136385;
                    double r_tmp_140823 = zp_res_136386;
                    
                    r_136382 = r_tmp_140823;
                }
                defunc_0_lifted_lambda_res_136380 = r_136382;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136398;
                double r_136400 = 0.0;
                
                for (int64_t i_136399 = 0; i_136399 < (int64_t) 16; i_136399++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136401 = ((double *) mem_139935)[i_136399 * (int64_t) 16 + i_137886];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136402 = ((double *) mem_138440)[i_136399 * (int64_t) 16 + i_137873];
                    
                    // futhark/microgpt.fut:405:68-111
                    
                    double zt_res_136403 = zt_lhs_136401 * zt_rhs_136402;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136404 = r_136400 + zt_res_136403;
                    double r_tmp_140824 = zp_res_136404;
                    
                    r_136400 = r_tmp_140824;
                }
                defunc_0_lifted_lambda_res_136398 = r_136400;
                ((double *) mem_140001)[i_137873] = defunc_0_lifted_lambda_res_136398;
                ((double *) mem_140002)[i_137873] = defunc_0_lifted_lambda_res_136380;
                ((double *) mem_140003)[i_137873] = defunc_0_lifted_lambda_res_136364;
                ((double *) mem_140004)[i_137873] = zp_res_136357;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139981, i_137886 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140001, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139982, i_137886 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140002, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139983, i_137886 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140003, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_139984, i_137886 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140004, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137895 = 0; i_137895 < (int64_t) 16; i_137895++) {
            // futhark/microgpt.fut:379:47-59
            
            double zp_lhs_127337 = ((double *) mem_138439)[i_137895];
            
            // futhark/microgpt.fut:379:47-87
            
            double zp_res_127338 = 1.0e-5 + zp_lhs_127337;
            
            // futhark/microgpt.fut:379:39-87
            
            double sqrt_res_127339 = futrts_sqrt64(zp_res_127338);
            
            // futhark/microgpt.fut:380:128-157
            
            double zt_res_127347 = sqrt_res_127339 * sqrt_res_127339;
            
            // futhark/microgpt.fut:380:119-157
            
            double zs_res_127348 = 1.0 / zt_res_127347;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127349;
            double r_127351 = 0.0;
            
            for (int64_t i_127350 = 0; i_127350 < (int64_t) 16; i_127350++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127352 = ((double *) mem_139984)[i_137895 * (int64_t) 16 + i_127350];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127353 = ((double *) mem_138417)[i_137895 * (int64_t) 16 + i_127350];
                
                // futhark/microgpt.fut:380:69-112
                
                double zt_res_127354 = zt_lhs_127352 * zt_rhs_127353;
                
                // futhark/microgpt.fut:380:90-157
                
                double zt_res_127355 = zs_res_127348 * zt_res_127354;
                
                // futhark/microgpt.fut:380:61-157
                
                double neg_res_127356 = -zt_res_127355;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127357 = r_127351 + neg_res_127356;
                double r_tmp_140827 = zp_res_127357;
                
                r_127351 = r_tmp_140827;
            }
            defunc_0_lifted_lambda_res_127349 = r_127351;
            ((double *) mem_140045)[i_137895] = defunc_0_lifted_lambda_res_127349;
            ((double *) mem_140046)[i_137895] = sqrt_res_127339;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137900 = 0; i_137900 < (int64_t) 16; i_137900++) {
            // futhark/microgpt.fut:381:39-51
            
            double zt_lhs_126943 = ((double *) mem_140045)[i_137900];
            
            // futhark/microgpt.fut:381:93-105
            
            double zp_lhs_126944 = ((double *) mem_138439)[i_137900];
            
            // futhark/microgpt.fut:381:93-133
            
            double zp_res_126945 = 1.0e-5 + zp_lhs_126944;
            
            // futhark/microgpt.fut:381:85-133
            
            double sqrt_res_126946 = futrts_sqrt64(zp_res_126945);
            
            // futhark/microgpt.fut:381:71-135
            
            double zt_res_126947 = 2.0 * sqrt_res_126946;
            
            // futhark/microgpt.fut:381:57-135
            
            double zs_res_126948 = 1.0 / zt_res_126947;
            
            // futhark/microgpt.fut:381:39-135
            
            double zt_res_126949 = zt_lhs_126943 * zs_res_126948;
            
            ((double *) mem_140059)[i_137900] = zt_res_126949;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137908 = 0; i_137908 < (int64_t) 16; i_137908++) {
            // futhark/microgpt.fut:382:98-110
            
            double zs_rhs_126957 = ((double *) mem_140046)[i_137908];
            
            // futhark/microgpt.fut:382:90-110
            
            double zs_res_126958 = 1.0 / zs_rhs_126957;
            
            // futhark/microgpt.fut:382:120-132
            
            double zs_lhs_126959 = ((double *) mem_140059)[i_137908];
            
            // futhark/microgpt.fut:382:120-147
            
            double zs_res_126960 = zs_lhs_126959 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137904 = 0; i_137904 < (int64_t) 16; i_137904++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_126967 = ((double *) mem_139200)[i_137908 * (int64_t) 16 + i_137904];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_126968 = ((double *) mem_139984)[i_137908 * (int64_t) 16 + i_137904];
                
                // futhark/microgpt.fut:382:64-110
                
                double zt_res_126969 = zs_res_126958 * zt_lhs_126968;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_126970 = ((double *) mem_138417)[i_137908 * (int64_t) 16 + i_137904];
                
                // futhark/microgpt.fut:382:133-171
                
                double zt_res_126971 = zs_res_126960 * zt_rhs_126970;
                
                // futhark/microgpt.fut:382:149-230
                
                double zp_res_126972 = zt_res_126971 + zt_res_126971;
                
                // futhark/microgpt.fut:382:85-230
                
                double zp_res_126973 = zt_res_126969 + zp_res_126972;
                
                // futhark/microgpt.fut:382:37-230
                
                double zp_res_126974 = zp_lhs_126967 + zp_res_126973;
                
                ((double *) mem_140071)[i_137904] = zp_res_126974;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140066, i_137908 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140071, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137914 = 0; i_137914 < (int64_t) 16; i_137914++) {
            // futhark/microgpt.fut:384:47-59
            
            double zp_lhs_127299 = ((double *) mem_138416)[i_137914];
            
            // futhark/microgpt.fut:384:47-87
            
            double zp_res_127300 = 1.0e-5 + zp_lhs_127299;
            
            // futhark/microgpt.fut:384:39-87
            
            double sqrt_res_127301 = futrts_sqrt64(zp_res_127300);
            
            // futhark/microgpt.fut:385:128-157
            
            double zt_res_127309 = sqrt_res_127301 * sqrt_res_127301;
            
            // futhark/microgpt.fut:385:119-157
            
            double zs_res_127310 = 1.0 / zt_res_127309;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_127311;
            double r_127313 = 0.0;
            
            for (int64_t i_127312 = 0; i_127312 < (int64_t) 16; i_127312++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_127314 = ((double *) mem_140066)[i_137914 * (int64_t) 16 + i_127312];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_127315 = ((double *) mem_138400)[i_137914 * (int64_t) 16 + i_127312];
                
                // futhark/microgpt.fut:385:69-112
                
                double zt_res_127316 = zt_lhs_127314 * zt_rhs_127315;
                
                // futhark/microgpt.fut:385:90-157
                
                double zt_res_127317 = zs_res_127310 * zt_res_127316;
                
                // futhark/microgpt.fut:385:61-157
                
                double neg_res_127318 = -zt_res_127317;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_127319 = r_127313 + neg_res_127318;
                double r_tmp_140833 = zp_res_127319;
                
                r_127313 = r_tmp_140833;
            }
            defunc_0_lifted_lambda_res_127311 = r_127313;
            ((double *) mem_140082)[i_137914] = defunc_0_lifted_lambda_res_127311;
            ((double *) mem_140083)[i_137914] = sqrt_res_127301;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137919 = 0; i_137919 < (int64_t) 16; i_137919++) {
            // futhark/microgpt.fut:386:39-51
            
            double zt_lhs_127026 = ((double *) mem_140082)[i_137919];
            
            // futhark/microgpt.fut:386:93-105
            
            double zp_lhs_127027 = ((double *) mem_138416)[i_137919];
            
            // futhark/microgpt.fut:386:93-133
            
            double zp_res_127028 = 1.0e-5 + zp_lhs_127027;
            
            // futhark/microgpt.fut:386:85-133
            
            double sqrt_res_127029 = futrts_sqrt64(zp_res_127028);
            
            // futhark/microgpt.fut:386:71-135
            
            double zt_res_127030 = 2.0 * sqrt_res_127029;
            
            // futhark/microgpt.fut:386:57-135
            
            double zs_res_127031 = 1.0 / zt_res_127030;
            
            // futhark/microgpt.fut:386:39-135
            
            double zt_res_127032 = zt_lhs_127026 * zs_res_127031;
            
            ((double *) mem_140096)[i_137919] = zt_res_127032;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137927 = 0; i_137927 < (int64_t) 16; i_137927++) {
            // futhark/microgpt.fut:387:72-84
            
            double zs_rhs_127040 = ((double *) mem_140083)[i_137927];
            
            // futhark/microgpt.fut:387:64-84
            
            double zs_res_127041 = 1.0 / zs_rhs_127040;
            
            // futhark/microgpt.fut:387:94-106
            
            double zs_lhs_127042 = ((double *) mem_140096)[i_137927];
            
            // futhark/microgpt.fut:387:94-121
            
            double zs_res_127043 = zs_lhs_127042 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137923 = 0; i_137923 < (int64_t) 16; i_137923++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_127050 = ((double *) mem_140066)[i_137927 * (int64_t) 16 + i_137923];
                
                // futhark/microgpt.fut:387:38-84
                
                double zt_res_127051 = zs_res_127041 * zt_lhs_127050;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_127052 = ((double *) mem_138400)[i_137927 * (int64_t) 16 + i_137923];
                
                // futhark/microgpt.fut:387:107-145
                
                double zt_res_127053 = zs_res_127043 * zt_rhs_127052;
                
                // futhark/microgpt.fut:387:123-204
                
                double zp_res_127054 = zt_res_127053 + zt_res_127053;
                
                // futhark/microgpt.fut:387:59-204
                
                double zp_res_127055 = zt_res_127051 + zp_res_127054;
                
                ((double *) mem_140108)[i_137923] = zp_res_127055;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140103, i_137927 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140108, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137940 = 0; i_137940 < (int64_t) 16; i_137940++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137933 = 0; i_137933 < (int64_t) 16; i_137933++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_136430 = ((double *) mem_140103)[i_137940 * (int64_t) 16 + i_137933];
                
                ((double *) mem_140129)[i_137933] = lifted_lambda_res_136430;
                ((double *) mem_140130)[i_137933] = lifted_lambda_res_136430;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140119, i_137940 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140129, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140120, i_137940 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140130, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137949 = 0; i_137949 < (int64_t) 64; i_137949++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137945 = 0; i_137945 < (int64_t) 16; i_137945++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_127169;
                double r_127171 = 0.0;
                
                for (int64_t i_127170 = 0; i_127170 < (int64_t) 16; i_127170++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_127172 = ((double *) mem_139131)[i_127170 * (int64_t) 64 + i_137949];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_127173 = ((double *) mem_138803)[i_127170 * (int64_t) 16 + i_137945];
                    
                    // futhark/microgpt.fut:407:67-111
                    
                    double zt_res_127174 = zt_lhs_127172 * zt_rhs_127173;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_127175 = r_127171 + zt_res_127174;
                    double r_tmp_140843 = zp_res_127175;
                    
                    r_127171 = r_tmp_140843;
                }
                defunc_0_lifted_lambda_res_127169 = r_127171;
                ((double *) mem_140156)[i_137945] = defunc_0_lifted_lambda_res_127169;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140151, i_137949 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140156, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_137962 = 0; i_137962 < (int64_t) 27; i_137962++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_137955 = 0; i_137955 < (int64_t) 16; i_137955++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136458;
                double r_136460 = 0.0;
                
                for (int64_t i_136459 = 0; i_136459 < (int64_t) 16; i_136459++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_136461 = ((double *) mem_139051)[i_136459 * (int64_t) 27 + i_137962];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_136462 = ((double *) mem_138873)[i_136459 * (int64_t) 16 + i_137955];
                    
                    // futhark/microgpt.fut:409:68-111
                    
                    double zt_res_136463 = zt_lhs_136461 * zt_rhs_136462;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136464 = r_136460 + zt_res_136463;
                    double r_tmp_140848 = zp_res_136464;
                    
                    r_136460 = r_tmp_140848;
                }
                defunc_0_lifted_lambda_res_136458 = r_136460;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_136467;
                double r_136469 = 0.0;
                
                for (int64_t i_136468 = 0; i_136468 < (int64_t) 16; i_136468++) {
                    int64_t zeze_lhs_136470 = ((int64_t *) seqs_mem_138259.mem)[step_124785 * (int64_t) 16 + i_136468];
                    
                    // futhark/microgpt.fut:579:58-109
                    
                    bool cond_136471 = zeze_lhs_136470 == i_137962;
                    
                    // futhark/microgpt.fut:579:58-109
                    
                    double lifted_lambda_res_136472;
                    
                    if (cond_136471) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_137065 = ((double *) mem_140119)[i_136468 * (int64_t) 16 + i_137955];
                        
                        lifted_lambda_res_136472 = lifted_lambda_res_t_res_137065;
                    } else {
                        lifted_lambda_res_136472 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_136478 = r_136469 + lifted_lambda_res_136472;
                    double r_tmp_140849 = zp_res_136478;
                    
                    r_136469 = r_tmp_140849;
                }
                defunc_0_lifted_lambda_res_136467 = r_136469;
                ((double *) mem_140177)[i_137955] = defunc_0_lifted_lambda_res_136467;
                ((double *) mem_140178)[i_137955] = defunc_0_lifted_lambda_res_136458;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140167, i_137962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140177, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_140168, i_137962 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_140178, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_127253 = sitofp_i64_f64(step_124785);
        
        // futhark/microgpt.fut:514:46-65
        
        double zm_rhs_127254 = i64_res_127253 / 500.0;
        
        // futhark/microgpt.fut:514:24-65
        
        double zt_rhs_127255 = 1.0 - zm_rhs_127254;
        
        // futhark/microgpt.fut:514:19-65
        
        double lt_r_127256 = 1.0e-2 * zt_rhs_127255;
        
        // futhark/microgpt.fut:516:5-52
        if (memblock_alloc(ctx, &mem_140199, (int64_t) 3456, "mem_140199")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-52
        // futhark/microgpt.fut:516:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140199.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138283.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:516:5-52
        if (memblock_alloc(ctx, &mem_140201, (int64_t) 3456, "mem_140201")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-52
        // futhark/microgpt.fut:516:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140201.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138319.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:516:5-52
        if (memblock_alloc(ctx, &mem_140203, (int64_t) 3456, "mem_140203")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-52
        // futhark/microgpt.fut:516:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140203.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138355.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:516:5-52
        if (memblock_alloc(ctx, &mem_140205, (int64_t) 3456, "mem_140205")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:516:5-52
        // futhark/microgpt.fut:516:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140205.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140167, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:516:5-52
        if (futrts_adam_opt_w_12499(ctx, &ext_mem_140209, &ext_mem_140208, &ext_mem_140207, mem_140199, mem_140201, mem_140203, mem_140205, (int64_t) 27, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140199, "mem_140199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140201, "mem_140201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140203, "mem_140203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140205, "mem_140205") != 0)
            return 1;
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_140210, (int64_t) 2048, "mem_140210")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140210.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138275.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_140212, (int64_t) 2048, "mem_140212")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140212.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138311.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_140214, (int64_t) 2048, "mem_140214")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140214.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138347.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (memblock_alloc(ctx, &mem_140216, (int64_t) 2048, "mem_140216")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:518:5-52
        // futhark/microgpt.fut:518:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140216.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140120, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:518:5-52
        if (futrts_adam_opt_w_12500(ctx, &ext_mem_140220, &ext_mem_140219, &ext_mem_140218, mem_140210, mem_140212, mem_140214, mem_140216, (int64_t) 16, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140210, "mem_140210") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140212, "mem_140212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140214, "mem_140214") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140216, "mem_140216") != 0)
            return 1;
        // futhark/microgpt.fut:520:5-56
        if (memblock_alloc(ctx, &mem_140221, (int64_t) 2048, "mem_140221")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-56
        // futhark/microgpt.fut:520:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140221.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138279.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:520:5-56
        if (memblock_alloc(ctx, &mem_140223, (int64_t) 2048, "mem_140223")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-56
        // futhark/microgpt.fut:520:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140223.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138315.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:520:5-56
        if (memblock_alloc(ctx, &mem_140225, (int64_t) 2048, "mem_140225")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-56
        // futhark/microgpt.fut:520:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140225.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138351.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:520:5-56
        if (memblock_alloc(ctx, &mem_140227, (int64_t) 2048, "mem_140227")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:520:5-56
        // futhark/microgpt.fut:520:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140227.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139983, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:520:5-56
        if (futrts_adam_opt_w_12500(ctx, &ext_mem_140231, &ext_mem_140230, &ext_mem_140229, mem_140221, mem_140223, mem_140225, mem_140227, (int64_t) 16, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140221, "mem_140221") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140223, "mem_140223") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140225, "mem_140225") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140227, "mem_140227") != 0)
            return 1;
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_140232, (int64_t) 2048, "mem_140232")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140232.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138267.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_140234, (int64_t) 2048, "mem_140234")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140234.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138303.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_140236, (int64_t) 2048, "mem_140236")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140236.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138339.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (memblock_alloc(ctx, &mem_140238, (int64_t) 2048, "mem_140238")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:522:5-56
        // futhark/microgpt.fut:522:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140238.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139982, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:522:5-56
        if (futrts_adam_opt_w_12500(ctx, &ext_mem_140242, &ext_mem_140241, &ext_mem_140240, mem_140232, mem_140234, mem_140236, mem_140238, (int64_t) 16, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140232, "mem_140232") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140234, "mem_140234") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140236, "mem_140236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140238, "mem_140238") != 0)
            return 1;
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_140243, (int64_t) 2048, "mem_140243")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140243.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138291.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_140245, (int64_t) 2048, "mem_140245")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140245.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138327.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_140247, (int64_t) 2048, "mem_140247")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140247.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138363.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (memblock_alloc(ctx, &mem_140249, (int64_t) 2048, "mem_140249")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:524:5-56
        // futhark/microgpt.fut:524:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140249.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139981, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:524:5-56
        if (futrts_adam_opt_w_12500(ctx, &ext_mem_140253, &ext_mem_140252, &ext_mem_140251, mem_140243, mem_140245, mem_140247, mem_140249, (int64_t) 16, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140243, "mem_140243") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140245, "mem_140245") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140247, "mem_140247") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140249, "mem_140249") != 0)
            return 1;
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_140254, (int64_t) 2048, "mem_140254")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140254.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138271.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_140256, (int64_t) 2048, "mem_140256")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140256.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138307.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_140258, (int64_t) 2048, "mem_140258")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140258.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138343.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (memblock_alloc(ctx, &mem_140260, (int64_t) 2048, "mem_140260")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:526:5-56
        // futhark/microgpt.fut:526:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140260.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_139216, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:526:5-56
        if (futrts_adam_opt_w_12500(ctx, &ext_mem_140264, &ext_mem_140263, &ext_mem_140262, mem_140254, mem_140256, mem_140258, mem_140260, (int64_t) 16, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140254, "mem_140254") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140256, "mem_140256") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140258, "mem_140258") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140260, "mem_140260") != 0)
            return 1;
        // futhark/microgpt.fut:528:5-52
        if (memblock_alloc(ctx, &mem_140265, (int64_t) 8192, "mem_140265")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-52
        // futhark/microgpt.fut:528:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140265.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138287.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:528:5-52
        if (memblock_alloc(ctx, &mem_140267, (int64_t) 8192, "mem_140267")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-52
        // futhark/microgpt.fut:528:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140267.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138323.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:528:5-52
        if (memblock_alloc(ctx, &mem_140269, (int64_t) 8192, "mem_140269")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-52
        // futhark/microgpt.fut:528:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140269.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138359.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:528:5-52
        if (memblock_alloc(ctx, &mem_140271, (int64_t) 8192, "mem_140271")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:528:5-52
        // futhark/microgpt.fut:528:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140271.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140151, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:528:5-52
        if (futrts_adam_opt_w_12499(ctx, &ext_mem_140275, &ext_mem_140274, &ext_mem_140273, mem_140265, mem_140267, mem_140269, mem_140271, (int64_t) 64, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140265, "mem_140265") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140267, "mem_140267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140269, "mem_140269") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140271, "mem_140271") != 0)
            return 1;
        // futhark/microgpt.fut:530:5-60
        if (memblock_alloc(ctx, &mem_140276, (int64_t) 8192, "mem_140276")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-60
        // futhark/microgpt.fut:530:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140276.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_138263.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:530:5-60
        if (memblock_alloc(ctx, &mem_140278, (int64_t) 8192, "mem_140278")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-60
        // futhark/microgpt.fut:530:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140278.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_138299.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:530:5-60
        if (memblock_alloc(ctx, &mem_140280, (int64_t) 8192, "mem_140280")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-60
        // futhark/microgpt.fut:530:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140280.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_138335.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:530:5-60
        if (memblock_alloc(ctx, &mem_140282, (int64_t) 8192, "mem_140282")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:530:5-60
        // futhark/microgpt.fut:530:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140282.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_139099, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:530:5-60
        if (futrts_adam_opt_w_12499(ctx, &ext_mem_140286, &ext_mem_140285, &ext_mem_140284, mem_140276, mem_140278, mem_140280, mem_140282, (int64_t) 16, (int64_t) 64, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140276, "mem_140276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140278, "mem_140278") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140280, "mem_140280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140282, "mem_140282") != 0)
            return 1;
        // futhark/microgpt.fut:532:5-56
        if (memblock_alloc(ctx, &mem_140287, (int64_t) 3456, "mem_140287")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-56
        // futhark/microgpt.fut:532:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140287.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138295.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:532:5-56
        if (memblock_alloc(ctx, &mem_140289, (int64_t) 3456, "mem_140289")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-56
        // futhark/microgpt.fut:532:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140289.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138331.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:532:5-56
        if (memblock_alloc(ctx, &mem_140291, (int64_t) 3456, "mem_140291")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-56
        // futhark/microgpt.fut:532:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140291.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_138367.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:532:5-56
        if (memblock_alloc(ctx, &mem_140293, (int64_t) 3456, "mem_140293")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:532:5-56
        // futhark/microgpt.fut:532:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_140293.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_140168, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:532:5-56
        if (futrts_adam_opt_w_12499(ctx, &ext_mem_140297, &ext_mem_140296, &ext_mem_140295, mem_140287, mem_140289, mem_140291, mem_140293, (int64_t) 27, (int64_t) 16, step_124785, lt_r_127256) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_140287, "mem_140287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140289, "mem_140289") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140291, "mem_140291") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140293, "mem_140293") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140505, &ext_mem_140286, "ext_mem_140286") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140506, &ext_mem_140242, "ext_mem_140242") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140507, &ext_mem_140264, "ext_mem_140264") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140508, &ext_mem_140220, "ext_mem_140220") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140509, &ext_mem_140231, "ext_mem_140231") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140510, &ext_mem_140209, "ext_mem_140209") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140511, &ext_mem_140275, "ext_mem_140275") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140512, &ext_mem_140253, "ext_mem_140253") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140513, &ext_mem_140297, "ext_mem_140297") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140514, &ext_mem_140285, "ext_mem_140285") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140515, &ext_mem_140241, "ext_mem_140241") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140516, &ext_mem_140263, "ext_mem_140263") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140517, &ext_mem_140219, "ext_mem_140219") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140518, &ext_mem_140230, "ext_mem_140230") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140519, &ext_mem_140208, "ext_mem_140208") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140520, &ext_mem_140274, "ext_mem_140274") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140521, &ext_mem_140252, "ext_mem_140252") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140522, &ext_mem_140296, "ext_mem_140296") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140523, &ext_mem_140284, "ext_mem_140284") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140524, &ext_mem_140240, "ext_mem_140240") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140525, &ext_mem_140262, "ext_mem_140262") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140526, &ext_mem_140218, "ext_mem_140218") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140527, &ext_mem_140229, "ext_mem_140229") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140528, &ext_mem_140207, "ext_mem_140207") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140529, &ext_mem_140273, "ext_mem_140273") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140530, &ext_mem_140251, "ext_mem_140251") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_140531, &ext_mem_140295, "ext_mem_140295") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138263, &mem_param_tmp_140505, "mem_param_tmp_140505") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138267, &mem_param_tmp_140506, "mem_param_tmp_140506") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138271, &mem_param_tmp_140507, "mem_param_tmp_140507") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138275, &mem_param_tmp_140508, "mem_param_tmp_140508") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138279, &mem_param_tmp_140509, "mem_param_tmp_140509") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138283, &mem_param_tmp_140510, "mem_param_tmp_140510") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138287, &mem_param_tmp_140511, "mem_param_tmp_140511") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138291, &mem_param_tmp_140512, "mem_param_tmp_140512") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138295, &mem_param_tmp_140513, "mem_param_tmp_140513") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138299, &mem_param_tmp_140514, "mem_param_tmp_140514") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138303, &mem_param_tmp_140515, "mem_param_tmp_140515") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138307, &mem_param_tmp_140516, "mem_param_tmp_140516") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138311, &mem_param_tmp_140517, "mem_param_tmp_140517") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138315, &mem_param_tmp_140518, "mem_param_tmp_140518") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138319, &mem_param_tmp_140519, "mem_param_tmp_140519") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138323, &mem_param_tmp_140520, "mem_param_tmp_140520") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138327, &mem_param_tmp_140521, "mem_param_tmp_140521") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138331, &mem_param_tmp_140522, "mem_param_tmp_140522") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138335, &mem_param_tmp_140523, "mem_param_tmp_140523") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138339, &mem_param_tmp_140524, "mem_param_tmp_140524") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138343, &mem_param_tmp_140525, "mem_param_tmp_140525") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138347, &mem_param_tmp_140526, "mem_param_tmp_140526") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138351, &mem_param_tmp_140527, "mem_param_tmp_140527") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138355, &mem_param_tmp_140528, "mem_param_tmp_140528") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138359, &mem_param_tmp_140529, "mem_param_tmp_140529") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138363, &mem_param_tmp_140530, "mem_param_tmp_140530") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_138367, &mem_param_tmp_140531, "mem_param_tmp_140531") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_140405, &mem_param_138263, "mem_param_138263") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140404, &mem_param_138267, "mem_param_138267") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140403, &mem_param_138271, "mem_param_138271") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140402, &mem_param_138275, "mem_param_138275") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140401, &mem_param_138279, "mem_param_138279") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140400, &mem_param_138283, "mem_param_138283") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140399, &mem_param_138287, "mem_param_138287") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140398, &mem_param_138291, "mem_param_138291") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140397, &mem_param_138295, "mem_param_138295") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140396, &mem_param_138299, "mem_param_138299") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140395, &mem_param_138303, "mem_param_138303") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140394, &mem_param_138307, "mem_param_138307") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140393, &mem_param_138311, "mem_param_138311") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140392, &mem_param_138315, "mem_param_138315") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140391, &mem_param_138319, "mem_param_138319") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140390, &mem_param_138323, "mem_param_138323") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140389, &mem_param_138327, "mem_param_138327") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140388, &mem_param_138331, "mem_param_138331") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140387, &mem_param_138335, "mem_param_138335") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140386, &mem_param_138339, "mem_param_138339") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140385, &mem_param_138343, "mem_param_138343") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140384, &mem_param_138347, "mem_param_138347") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140383, &mem_param_138351, "mem_param_138351") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140382, &mem_param_138355, "mem_param_138355") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140381, &mem_param_138359, "mem_param_138359") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140380, &mem_param_138363, "mem_param_138363") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_140379, &mem_param_138367, "mem_param_138367") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140478, &ext_mem_140400, "ext_mem_140400") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140479, &ext_mem_140402, "ext_mem_140402") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140480, &ext_mem_140401, "ext_mem_140401") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140481, &ext_mem_140404, "ext_mem_140404") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140482, &ext_mem_140398, "ext_mem_140398") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140483, &ext_mem_140403, "ext_mem_140403") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140484, &ext_mem_140399, "ext_mem_140399") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140485, &ext_mem_140405, "ext_mem_140405") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140486, &ext_mem_140397, "ext_mem_140397") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140487, &ext_mem_140391, "ext_mem_140391") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140488, &ext_mem_140393, "ext_mem_140393") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140489, &ext_mem_140392, "ext_mem_140392") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140490, &ext_mem_140395, "ext_mem_140395") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140491, &ext_mem_140389, "ext_mem_140389") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140492, &ext_mem_140394, "ext_mem_140394") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140493, &ext_mem_140390, "ext_mem_140390") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140494, &ext_mem_140396, "ext_mem_140396") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140495, &ext_mem_140388, "ext_mem_140388") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140496, &ext_mem_140382, "ext_mem_140382") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140497, &ext_mem_140384, "ext_mem_140384") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140498, &ext_mem_140383, "ext_mem_140383") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140499, &ext_mem_140386, "ext_mem_140386") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140500, &ext_mem_140380, "ext_mem_140380") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140501, &ext_mem_140385, "ext_mem_140385") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140502, &ext_mem_140381, "ext_mem_140381") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140503, &ext_mem_140387, "ext_mem_140387") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140504, &ext_mem_140379, "ext_mem_140379") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141003, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141004, &mem_out_140479, "mem_out_140479") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141005, &mem_out_140480, "mem_out_140480") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141006, &mem_out_140481, "mem_out_140481") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141007, &mem_out_140482, "mem_out_140482") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141008, &mem_out_140483, "mem_out_140483") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141009, &mem_out_140484, "mem_out_140484") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141010, &mem_out_140485, "mem_out_140485") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141011, &mem_out_140486, "mem_out_140486") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141012, &mem_out_140487, "mem_out_140487") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141013, &mem_out_140488, "mem_out_140488") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141014, &mem_out_140489, "mem_out_140489") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141015, &mem_out_140490, "mem_out_140490") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141016, &mem_out_140491, "mem_out_140491") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141017, &mem_out_140492, "mem_out_140492") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141018, &mem_out_140493, "mem_out_140493") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141019, &mem_out_140494, "mem_out_140494") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141020, &mem_out_140495, "mem_out_140495") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141021, &mem_out_140496, "mem_out_140496") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141022, &mem_out_140497, "mem_out_140497") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141023, &mem_out_140498, "mem_out_140498") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141024, &mem_out_140499, "mem_out_140499") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141025, &mem_out_140500, "mem_out_140500") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141026, &mem_out_140501, "mem_out_140501") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141027, &mem_out_140502, "mem_out_140502") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141028, &mem_out_140503, "mem_out_140503") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141029, &mem_out_140504, "mem_out_140504") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_138368);
        free(mem_138369);
        free(mem_138378);
        free(mem_138385);
        free(mem_138400);
        free(mem_138405);
        free(mem_138416);
        free(mem_138417);
        free(mem_138425);
        free(mem_138439);
        free(mem_138440);
        free(mem_138448);
        free(mem_138462);
        free(mem_138463);
        free(mem_138464);
        free(mem_138477);
        free(mem_138478);
        free(mem_138479);
        free(mem_138510);
        free(mem_138511);
        free(mem_138512);
        free(mem_138528);
        free(mem_138529);
        free(mem_138530);
        free(mem_138543);
        free(mem_138544);
        free(mem_138545);
        free(mem_138591);
        free(mem_138592);
        free(mem_138593);
        free(mem_138594);
        free(mem_138615);
        free(mem_138616);
        free(mem_138617);
        free(mem_138618);
        free(mem_138635);
        free(mem_138636);
        free(mem_138637);
        free(mem_138638);
        free(mem_138679);
        free(mem_138684);
        free(mem_138695);
        free(mem_138700);
        free(mem_138707);
        free(mem_138718);
        free(mem_138723);
        free(mem_138754);
        free(mem_138759);
        free(mem_138770);
        free(mem_138775);
        free(mem_138786);
        free(mem_138791);
        free(mem_138802);
        free(mem_138803);
        free(mem_138811);
        free(mem_138825);
        free(mem_138830);
        free(mem_138841);
        free(mem_138846);
        free(mem_138857);
        free(mem_138862);
        free(mem_138873);
        free(mem_138878);
        free(mem_138889);
        free(mem_138894);
        free(mem_138905);
        free(mem_138906);
        free(mem_138907);
        free(mem_138935);
        free(mem_138941);
        free(mem_138946);
        free(mem_138962);
        free(mem_138967);
        free(mem_138978);
        free(mem_138983);
        free(mem_138987);
        free(mem_139001);
        free(mem_139007);
        free(mem_139012);
        free(mem_139016);
        free(mem_139035);
        free(mem_139040);
        free(mem_139051);
        free(mem_139056);
        free(mem_139067);
        free(mem_139072);
        free(mem_139083);
        free(mem_139088);
        free(mem_139099);
        free(mem_139100);
        free(mem_139109);
        free(mem_139110);
        free(mem_139131);
        free(mem_139136);
        free(mem_139147);
        free(mem_139152);
        free(mem_139163);
        free(mem_139164);
        free(mem_139177);
        free(mem_139184);
        free(mem_139189);
        free(mem_139200);
        free(mem_139205);
        free(mem_139216);
        free(mem_139217);
        free(mem_139226);
        free(mem_139227);
        free(mem_139248);
        free(mem_139249);
        free(mem_139250);
        free(mem_139251);
        free(mem_139272);
        free(mem_139273);
        free(mem_139274);
        free(mem_139275);
        free(mem_139292);
        free(mem_139299);
        free(mem_139300);
        free(mem_139301);
        free(mem_139356);
        free(mem_139357);
        free(mem_139358);
        free(mem_139359);
        free(mem_139360);
        free(mem_139361);
        free(mem_139362);
        free(mem_139363);
        free(mem_139364);
        free(mem_139404);
        free(mem_139405);
        free(mem_139406);
        free(mem_139407);
        free(mem_139408);
        free(mem_139409);
        free(mem_139410);
        free(mem_139411);
        free(mem_139412);
        free(mem_139443);
        free(mem_139444);
        free(mem_139457);
        free(mem_139464);
        free(mem_139540);
        free(mem_139541);
        free(mem_139542);
        free(mem_139558);
        free(mem_139559);
        free(mem_139560);
        free(mem_139573);
        free(mem_139580);
        free(mem_139581);
        free(mem_139621);
        free(mem_139622);
        free(mem_139623);
        free(mem_139624);
        free(mem_139641);
        free(mem_139642);
        free(mem_139643);
        free(mem_139644);
        free(mem_139685);
        free(mem_139686);
        free(mem_139697);
        free(mem_139698);
        free(mem_139707);
        free(mem_139708);
        free(mem_139739);
        free(mem_139740);
        free(mem_139749);
        free(mem_139750);
        free(mem_139771);
        free(mem_139772);
        free(mem_139783);
        free(mem_139784);
        free(mem_139793);
        free(mem_139794);
        free(mem_139825);
        free(mem_139826);
        free(mem_139837);
        free(mem_139838);
        free(mem_139847);
        free(mem_139848);
        free(mem_139879);
        free(mem_139880);
        free(mem_139891);
        free(mem_139892);
        free(mem_139901);
        free(mem_139902);
        free(mem_139933);
        free(mem_139934);
        free(mem_139935);
        free(mem_139948);
        free(mem_139949);
        free(mem_139950);
        free(mem_139981);
        free(mem_139982);
        free(mem_139983);
        free(mem_139984);
        free(mem_140001);
        free(mem_140002);
        free(mem_140003);
        free(mem_140004);
        free(mem_140045);
        free(mem_140046);
        free(mem_140059);
        free(mem_140066);
        free(mem_140071);
        free(mem_140082);
        free(mem_140083);
        free(mem_140096);
        free(mem_140103);
        free(mem_140108);
        free(mem_140119);
        free(mem_140120);
        free(mem_140129);
        free(mem_140130);
        free(mem_140151);
        free(mem_140156);
        free(mem_140167);
        free(mem_140168);
        free(mem_140177);
        free(mem_140178);
        if (memblock_unref(ctx, &mem_param_tmp_140531, "mem_param_tmp_140531") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140530, "mem_param_tmp_140530") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140529, "mem_param_tmp_140529") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140528, "mem_param_tmp_140528") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140527, "mem_param_tmp_140527") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140526, "mem_param_tmp_140526") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140525, "mem_param_tmp_140525") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140524, "mem_param_tmp_140524") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140523, "mem_param_tmp_140523") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140522, "mem_param_tmp_140522") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140521, "mem_param_tmp_140521") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140520, "mem_param_tmp_140520") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140519, "mem_param_tmp_140519") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140518, "mem_param_tmp_140518") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140517, "mem_param_tmp_140517") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140516, "mem_param_tmp_140516") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140515, "mem_param_tmp_140515") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140514, "mem_param_tmp_140514") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140513, "mem_param_tmp_140513") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140512, "mem_param_tmp_140512") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140511, "mem_param_tmp_140511") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140510, "mem_param_tmp_140510") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140509, "mem_param_tmp_140509") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140508, "mem_param_tmp_140508") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140507, "mem_param_tmp_140507") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140506, "mem_param_tmp_140506") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_140505, "mem_param_tmp_140505") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140295, "ext_mem_140295") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140296, "ext_mem_140296") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140297, "ext_mem_140297") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140293, "mem_140293") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140291, "mem_140291") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140289, "mem_140289") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140287, "mem_140287") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140284, "ext_mem_140284") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140285, "ext_mem_140285") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140286, "ext_mem_140286") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140282, "mem_140282") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140280, "mem_140280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140278, "mem_140278") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140276, "mem_140276") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140273, "ext_mem_140273") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140274, "ext_mem_140274") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140275, "ext_mem_140275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140271, "mem_140271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140269, "mem_140269") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140267, "mem_140267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140265, "mem_140265") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140262, "ext_mem_140262") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140263, "ext_mem_140263") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140264, "ext_mem_140264") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140260, "mem_140260") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140258, "mem_140258") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140256, "mem_140256") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140254, "mem_140254") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140251, "ext_mem_140251") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140252, "ext_mem_140252") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140253, "ext_mem_140253") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140249, "mem_140249") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140247, "mem_140247") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140245, "mem_140245") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140243, "mem_140243") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140240, "ext_mem_140240") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140241, "ext_mem_140241") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140242, "ext_mem_140242") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140238, "mem_140238") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140236, "mem_140236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140234, "mem_140234") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140232, "mem_140232") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140229, "ext_mem_140229") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140230, "ext_mem_140230") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140231, "ext_mem_140231") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140227, "mem_140227") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140225, "mem_140225") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140223, "mem_140223") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140221, "mem_140221") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140218, "ext_mem_140218") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140219, "ext_mem_140219") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140220, "ext_mem_140220") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140216, "mem_140216") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140214, "mem_140214") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140212, "mem_140212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140210, "mem_140210") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140207, "ext_mem_140207") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140208, "ext_mem_140208") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140209, "ext_mem_140209") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140205, "mem_140205") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140203, "mem_140203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140201, "mem_140201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_140199, "mem_140199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138367, "mem_param_138367") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138363, "mem_param_138363") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138359, "mem_param_138359") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138355, "mem_param_138355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138351, "mem_param_138351") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138347, "mem_param_138347") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138343, "mem_param_138343") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138339, "mem_param_138339") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138335, "mem_param_138335") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138331, "mem_param_138331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138327, "mem_param_138327") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138323, "mem_param_138323") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138319, "mem_param_138319") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138315, "mem_param_138315") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138311, "mem_param_138311") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138307, "mem_param_138307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138303, "mem_param_138303") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138299, "mem_param_138299") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138295, "mem_param_138295") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138291, "mem_param_138291") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138287, "mem_param_138287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138283, "mem_param_138283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138279, "mem_param_138279") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138275, "mem_param_138275") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138271, "mem_param_138271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138267, "mem_param_138267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_138263, "mem_param_138263") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140379, "ext_mem_140379") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140380, "ext_mem_140380") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140381, "ext_mem_140381") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140382, "ext_mem_140382") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140383, "ext_mem_140383") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140384, "ext_mem_140384") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140385, "ext_mem_140385") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140386, "ext_mem_140386") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140387, "ext_mem_140387") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140388, "ext_mem_140388") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140389, "ext_mem_140389") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140390, "ext_mem_140390") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140391, "ext_mem_140391") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140392, "ext_mem_140392") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140393, "ext_mem_140393") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140394, "ext_mem_140394") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140395, "ext_mem_140395") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140396, "ext_mem_140396") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140397, "ext_mem_140397") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140398, "ext_mem_140398") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140399, "ext_mem_140399") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140400, "ext_mem_140400") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140401, "ext_mem_140401") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140402, "ext_mem_140402") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140403, "ext_mem_140403") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140404, "ext_mem_140404") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_140405, "ext_mem_140405") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140504, "mem_out_140504") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140503, "mem_out_140503") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140502, "mem_out_140502") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140501, "mem_out_140501") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140500, "mem_out_140500") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140499, "mem_out_140499") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140498, "mem_out_140498") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140497, "mem_out_140497") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140496, "mem_out_140496") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140495, "mem_out_140495") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140494, "mem_out_140494") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140493, "mem_out_140493") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140492, "mem_out_140492") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140491, "mem_out_140491") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140490, "mem_out_140490") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140489, "mem_out_140489") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140488, "mem_out_140488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140487, "mem_out_140487") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140486, "mem_out_140486") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140485, "mem_out_140485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140484, "mem_out_140484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140483, "mem_out_140483") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140482, "mem_out_140482") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140481, "mem_out_140481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140480, "mem_out_140480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140479, "mem_out_140479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_141250, struct memblock *mem_out_p_141251, struct memblock *mem_out_p_141252, struct memblock *mem_out_p_141253, struct memblock *mem_out_p_141254, struct memblock *mem_out_p_141255, struct memblock *mem_out_p_141256, struct memblock *mem_out_p_141257, struct memblock *mem_out_p_141258)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mem_138221 = ctx->constants->mem_138221;
    struct memblock mem_138222 = ctx->constants->mem_138222;
    struct memblock mem_138223 = ctx->constants->mem_138223;
    struct memblock mem_138224 = ctx->constants->mem_138224;
    struct memblock mem_138225 = ctx->constants->mem_138225;
    struct memblock mem_138226 = ctx->constants->mem_138226;
    struct memblock mem_138227 = ctx->constants->mem_138227;
    struct memblock mem_138228 = ctx->constants->mem_138228;
    struct memblock mem_138229 = ctx->constants->mem_138229;
    
    if (memblock_set(ctx, &mem_out_140478, &mem_138228, "mem_138228") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140479, &mem_138224, "mem_138224") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140480, &mem_138226, "mem_138226") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140481, &mem_138222, "mem_138222") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140482, &mem_138223, "mem_138223") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140483, &mem_138221, "mem_138221") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140484, &mem_138227, "mem_138227") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140485, &mem_138225, "mem_138225") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_140486, &mem_138229, "mem_138229") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141250, &mem_out_140478, "mem_out_140478") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141251, &mem_out_140479, "mem_out_140479") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141252, &mem_out_140480, "mem_out_140480") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141253, &mem_out_140481, "mem_out_140481") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141254, &mem_out_140482, "mem_out_140482") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141255, &mem_out_140483, "mem_out_140483") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141256, &mem_out_140484, "mem_out_140484") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141257, &mem_out_140485, "mem_out_140485") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_141258, &mem_out_140486, "mem_out_140486") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_140486, "mem_out_140486") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140485, "mem_out_140485") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140484, "mem_out_140484") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140483, "mem_out_140483") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140482, "mem_out_140482") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140481, "mem_out_140481") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140480, "mem_out_140480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140479, "mem_out_140479") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_140478, "mem_out_140478") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_cal_loss(struct futhark_context *ctx, struct futhark_opaque_tup2_f64_arr1d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3)
{
    double prim_out_140479 = 0.0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mask_mem_138241;
    
    mask_mem_138241.references = NULL;
    
    struct memblock target_mem_138240;
    
    target_mem_138240.references = NULL;
    
    struct memblock tokens_mem_138239;
    
    tokens_mem_138239.references = NULL;
    
    struct memblock wvoc_mem_138238;
    
    wvoc_mem_138238.references = NULL;
    
    struct memblock wval_mem_138237;
    
    wval_mem_138237.references = NULL;
    
    struct memblock wup_mem_138236;
    
    wup_mem_138236.references = NULL;
    
    struct memblock wte_mem_138235;
    
    wte_mem_138235.references = NULL;
    
    struct memblock wqry_mem_138234;
    
    wqry_mem_138234.references = NULL;
    
    struct memblock wpe_mem_138233;
    
    wpe_mem_138233.references = NULL;
    
    struct memblock wout_mem_138232;
    
    wout_mem_138232.references = NULL;
    
    struct memblock wkey_mem_138231;
    
    wkey_mem_138231.references = NULL;
    
    struct memblock wdown_mem_138230;
    
    wdown_mem_138230.references = NULL;
    wdown_mem_138230 = in0->v0->mem;
    wkey_mem_138231 = in0->v1->mem;
    wout_mem_138232 = in0->v2->mem;
    wpe_mem_138233 = in0->v3->mem;
    wqry_mem_138234 = in0->v4->mem;
    wte_mem_138235 = in0->v5->mem;
    wup_mem_138236 = in0->v6->mem;
    wval_mem_138237 = in0->v7->mem;
    wvoc_mem_138238 = in0->v8->mem;
    tokens_mem_138239 = in1->mem;
    target_mem_138240 = in2->mem;
    mask_mem_138241 = in3->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && (((int64_t) 16 == in2->shape[0] && (int64_t) 27 == in2->shape[1]) && ((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_cal_loss(ctx, &mem_out_140478, &prim_out_140479, wdown_mem_138230, wkey_mem_138231, wout_mem_138232, wpe_mem_138233, wqry_mem_138234, wte_mem_138235, wup_mem_138236, wval_mem_138237, wvoc_mem_138238, tokens_mem_138239, target_mem_138240, mask_mem_138241);
        if (ret == 0) {
            struct memblock mem_138221 = ctx->constants->mem_138221;
            struct memblock mem_138222 = ctx->constants->mem_138222;
            struct memblock mem_138223 = ctx->constants->mem_138223;
            struct memblock mem_138224 = ctx->constants->mem_138224;
            struct memblock mem_138225 = ctx->constants->mem_138225;
            struct memblock mem_138226 = ctx->constants->mem_138226;
            struct memblock mem_138227 = ctx->constants->mem_138227;
            struct memblock mem_138228 = ctx->constants->mem_138228;
            struct memblock mem_138229 = ctx->constants->mem_138229;
            
            assert((*out = (struct futhark_opaque_tup2_f64_arr1d_f64 *) malloc(sizeof(struct futhark_opaque_tup2_f64_arr1d_f64))) != NULL);
            (*out)->v0 = prim_out_140479;
            assert(((*out)->v1 = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d))) != NULL);
            (*out)->v1->mem = mem_out_140478;
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
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock mask_mem_138240;
    
    mask_mem_138240.references = NULL;
    
    struct memblock tokens_mem_138239;
    
    tokens_mem_138239.references = NULL;
    
    struct memblock wvoc_mem_138238;
    
    wvoc_mem_138238.references = NULL;
    
    struct memblock wval_mem_138237;
    
    wval_mem_138237.references = NULL;
    
    struct memblock wup_mem_138236;
    
    wup_mem_138236.references = NULL;
    
    struct memblock wte_mem_138235;
    
    wte_mem_138235.references = NULL;
    
    struct memblock wqry_mem_138234;
    
    wqry_mem_138234.references = NULL;
    
    struct memblock wpe_mem_138233;
    
    wpe_mem_138233.references = NULL;
    
    struct memblock wout_mem_138232;
    
    wout_mem_138232.references = NULL;
    
    struct memblock wkey_mem_138231;
    
    wkey_mem_138231.references = NULL;
    
    struct memblock wdown_mem_138230;
    
    wdown_mem_138230.references = NULL;
    wdown_mem_138230 = in0->v0->mem;
    wkey_mem_138231 = in0->v1->mem;
    wout_mem_138232 = in0->v2->mem;
    wpe_mem_138233 = in0->v3->mem;
    wqry_mem_138234 = in0->v4->mem;
    wte_mem_138235 = in0->v5->mem;
    wup_mem_138236 = in0->v6->mem;
    wval_mem_138237 = in0->v7->mem;
    wvoc_mem_138238 = in0->v8->mem;
    tokens_mem_138239 = in1->mem;
    mask_mem_138240 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_140478, wdown_mem_138230, wkey_mem_138231, wout_mem_138232, wpe_mem_138233, wqry_mem_138234, wte_mem_138235, wup_mem_138236, wval_mem_138237, wvoc_mem_138238, tokens_mem_138239, mask_mem_138240);
        if (ret == 0) {
            struct memblock mem_138221 = ctx->constants->mem_138221;
            struct memblock mem_138222 = ctx->constants->mem_138222;
            struct memblock mem_138223 = ctx->constants->mem_138223;
            struct memblock mem_138224 = ctx->constants->mem_138224;
            struct memblock mem_138225 = ctx->constants->mem_138225;
            struct memblock mem_138226 = ctx->constants->mem_138226;
            struct memblock mem_138227 = ctx->constants->mem_138227;
            struct memblock mem_138228 = ctx->constants->mem_138228;
            struct memblock mem_138229 = ctx->constants->mem_138229;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_140478;
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
    
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock wvoc_mem_138238;
    
    wvoc_mem_138238.references = NULL;
    
    struct memblock wdown_mem_138237;
    
    wdown_mem_138237.references = NULL;
    
    struct memblock wup_mem_138236;
    
    wup_mem_138236.references = NULL;
    
    struct memblock wout_mem_138235;
    
    wout_mem_138235.references = NULL;
    
    struct memblock wval_mem_138234;
    
    wval_mem_138234.references = NULL;
    
    struct memblock wkey_mem_138233;
    
    wkey_mem_138233.references = NULL;
    
    struct memblock wqry_mem_138232;
    
    wqry_mem_138232.references = NULL;
    
    struct memblock wpe_mem_138231;
    
    wpe_mem_138231.references = NULL;
    
    struct memblock wte_mem_138230;
    
    wte_mem_138230.references = NULL;
    wte_mem_138230 = in0->mem;
    wpe_mem_138231 = in1->mem;
    wqry_mem_138232 = in2->mem;
    wkey_mem_138233 = in3->mem;
    wval_mem_138234 = in4->mem;
    wout_mem_138235 = in5->mem;
    wup_mem_138236 = in6->mem;
    wdown_mem_138237 = in7->mem;
    wvoc_mem_138238 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_140478, &mem_out_140479, &mem_out_140480, &mem_out_140481, &mem_out_140482, &mem_out_140483, &mem_out_140484, &mem_out_140485, &mem_out_140486, wte_mem_138230, wpe_mem_138231, wqry_mem_138232, wkey_mem_138233, wval_mem_138234, wout_mem_138235, wup_mem_138236, wdown_mem_138237, wvoc_mem_138238);
        if (ret == 0) {
            struct memblock mem_138221 = ctx->constants->mem_138221;
            struct memblock mem_138222 = ctx->constants->mem_138222;
            struct memblock mem_138223 = ctx->constants->mem_138223;
            struct memblock mem_138224 = ctx->constants->mem_138224;
            struct memblock mem_138225 = ctx->constants->mem_138225;
            struct memblock mem_138226 = ctx->constants->mem_138226;
            struct memblock mem_138227 = ctx->constants->mem_138227;
            struct memblock mem_138228 = ctx->constants->mem_138228;
            struct memblock mem_138229 = ctx->constants->mem_138229;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_140478;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_140479;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_140480;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_140481;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_140482;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_140483;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_140484;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_140485;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_140486;
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
    
    struct memblock mem_out_140504;
    
    mem_out_140504.references = NULL;
    
    struct memblock mem_out_140503;
    
    mem_out_140503.references = NULL;
    
    struct memblock mem_out_140502;
    
    mem_out_140502.references = NULL;
    
    struct memblock mem_out_140501;
    
    mem_out_140501.references = NULL;
    
    struct memblock mem_out_140500;
    
    mem_out_140500.references = NULL;
    
    struct memblock mem_out_140499;
    
    mem_out_140499.references = NULL;
    
    struct memblock mem_out_140498;
    
    mem_out_140498.references = NULL;
    
    struct memblock mem_out_140497;
    
    mem_out_140497.references = NULL;
    
    struct memblock mem_out_140496;
    
    mem_out_140496.references = NULL;
    
    struct memblock mem_out_140495;
    
    mem_out_140495.references = NULL;
    
    struct memblock mem_out_140494;
    
    mem_out_140494.references = NULL;
    
    struct memblock mem_out_140493;
    
    mem_out_140493.references = NULL;
    
    struct memblock mem_out_140492;
    
    mem_out_140492.references = NULL;
    
    struct memblock mem_out_140491;
    
    mem_out_140491.references = NULL;
    
    struct memblock mem_out_140490;
    
    mem_out_140490.references = NULL;
    
    struct memblock mem_out_140489;
    
    mem_out_140489.references = NULL;
    
    struct memblock mem_out_140488;
    
    mem_out_140488.references = NULL;
    
    struct memblock mem_out_140487;
    
    mem_out_140487.references = NULL;
    
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    
    struct memblock seqs_mem_138259;
    
    seqs_mem_138259.references = NULL;
    
    struct memblock dls_mem_138258;
    
    dls_mem_138258.references = NULL;
    
    struct memblock masks_mem_138257;
    
    masks_mem_138257.references = NULL;
    
    struct memblock wvoc_mem_138256;
    
    wvoc_mem_138256.references = NULL;
    
    struct memblock wval_mem_138255;
    
    wval_mem_138255.references = NULL;
    
    struct memblock wup_mem_138254;
    
    wup_mem_138254.references = NULL;
    
    struct memblock wte_mem_138253;
    
    wte_mem_138253.references = NULL;
    
    struct memblock wqry_mem_138252;
    
    wqry_mem_138252.references = NULL;
    
    struct memblock wpe_mem_138251;
    
    wpe_mem_138251.references = NULL;
    
    struct memblock wout_mem_138250;
    
    wout_mem_138250.references = NULL;
    
    struct memblock wkey_mem_138249;
    
    wkey_mem_138249.references = NULL;
    
    struct memblock wdown_mem_138248;
    
    wdown_mem_138248.references = NULL;
    
    struct memblock wvoc_mem_138247;
    
    wvoc_mem_138247.references = NULL;
    
    struct memblock wval_mem_138246;
    
    wval_mem_138246.references = NULL;
    
    struct memblock wup_mem_138245;
    
    wup_mem_138245.references = NULL;
    
    struct memblock wte_mem_138244;
    
    wte_mem_138244.references = NULL;
    
    struct memblock wqry_mem_138243;
    
    wqry_mem_138243.references = NULL;
    
    struct memblock wpe_mem_138242;
    
    wpe_mem_138242.references = NULL;
    
    struct memblock wout_mem_138241;
    
    wout_mem_138241.references = NULL;
    
    struct memblock wkey_mem_138240;
    
    wkey_mem_138240.references = NULL;
    
    struct memblock wdown_mem_138239;
    
    wdown_mem_138239.references = NULL;
    
    struct memblock wvoc_mem_138238;
    
    wvoc_mem_138238.references = NULL;
    
    struct memblock wval_mem_138237;
    
    wval_mem_138237.references = NULL;
    
    struct memblock wup_mem_138236;
    
    wup_mem_138236.references = NULL;
    
    struct memblock wte_mem_138235;
    
    wte_mem_138235.references = NULL;
    
    struct memblock wqry_mem_138234;
    
    wqry_mem_138234.references = NULL;
    
    struct memblock wpe_mem_138233;
    
    wpe_mem_138233.references = NULL;
    
    struct memblock wout_mem_138232;
    
    wout_mem_138232.references = NULL;
    
    struct memblock wkey_mem_138231;
    
    wkey_mem_138231.references = NULL;
    
    struct memblock wdown_mem_138230;
    
    wdown_mem_138230.references = NULL;
    wdown_mem_138230 = in0->v0->mem;
    wkey_mem_138231 = in0->v1->mem;
    wout_mem_138232 = in0->v2->mem;
    wpe_mem_138233 = in0->v3->mem;
    wqry_mem_138234 = in0->v4->mem;
    wte_mem_138235 = in0->v5->mem;
    wup_mem_138236 = in0->v6->mem;
    wval_mem_138237 = in0->v7->mem;
    wvoc_mem_138238 = in0->v8->mem;
    wdown_mem_138239 = in1->v0->mem;
    wkey_mem_138240 = in1->v1->mem;
    wout_mem_138241 = in1->v2->mem;
    wpe_mem_138242 = in1->v3->mem;
    wqry_mem_138243 = in1->v4->mem;
    wte_mem_138244 = in1->v5->mem;
    wup_mem_138245 = in1->v6->mem;
    wval_mem_138246 = in1->v7->mem;
    wvoc_mem_138247 = in1->v8->mem;
    wdown_mem_138248 = in2->v0->mem;
    wkey_mem_138249 = in2->v1->mem;
    wout_mem_138250 = in2->v2->mem;
    wpe_mem_138251 = in2->v3->mem;
    wqry_mem_138252 = in2->v4->mem;
    wte_mem_138253 = in2->v5->mem;
    wup_mem_138254 = in2->v6->mem;
    wval_mem_138255 = in2->v7->mem;
    wvoc_mem_138256 = in2->v8->mem;
    masks_mem_138257 = in3->mem;
    dls_mem_138258 = in4->mem;
    seqs_mem_138259 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_140478, &mem_out_140479, &mem_out_140480, &mem_out_140481, &mem_out_140482, &mem_out_140483, &mem_out_140484, &mem_out_140485, &mem_out_140486, &mem_out_140487, &mem_out_140488, &mem_out_140489, &mem_out_140490, &mem_out_140491, &mem_out_140492, &mem_out_140493, &mem_out_140494, &mem_out_140495, &mem_out_140496, &mem_out_140497, &mem_out_140498, &mem_out_140499, &mem_out_140500, &mem_out_140501, &mem_out_140502, &mem_out_140503, &mem_out_140504, wdown_mem_138230, wkey_mem_138231, wout_mem_138232, wpe_mem_138233, wqry_mem_138234, wte_mem_138235, wup_mem_138236, wval_mem_138237, wvoc_mem_138238, wdown_mem_138239, wkey_mem_138240, wout_mem_138241, wpe_mem_138242, wqry_mem_138243, wte_mem_138244, wup_mem_138245, wval_mem_138246, wvoc_mem_138247, wdown_mem_138248, wkey_mem_138249, wout_mem_138250, wpe_mem_138251, wqry_mem_138252, wte_mem_138253, wup_mem_138254, wval_mem_138255, wvoc_mem_138256, masks_mem_138257, dls_mem_138258, seqs_mem_138259);
        if (ret == 0) {
            struct memblock mem_138221 = ctx->constants->mem_138221;
            struct memblock mem_138222 = ctx->constants->mem_138222;
            struct memblock mem_138223 = ctx->constants->mem_138223;
            struct memblock mem_138224 = ctx->constants->mem_138224;
            struct memblock mem_138225 = ctx->constants->mem_138225;
            struct memblock mem_138226 = ctx->constants->mem_138226;
            struct memblock mem_138227 = ctx->constants->mem_138227;
            struct memblock mem_138228 = ctx->constants->mem_138228;
            struct memblock mem_138229 = ctx->constants->mem_138229;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_140478;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_140479;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_140480;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_140481;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_140482;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_140483;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_140484;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_140485;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_140486;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_140487;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_140488;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_140489;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_140490;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_140491;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_140492;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_140493;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_140494;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_140495;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_140496;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_140497;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_140498;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_140499;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_140500;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_140501;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_140502;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_140503;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_140504;
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
    
    struct memblock mem_out_140486;
    
    mem_out_140486.references = NULL;
    
    struct memblock mem_out_140485;
    
    mem_out_140485.references = NULL;
    
    struct memblock mem_out_140484;
    
    mem_out_140484.references = NULL;
    
    struct memblock mem_out_140483;
    
    mem_out_140483.references = NULL;
    
    struct memblock mem_out_140482;
    
    mem_out_140482.references = NULL;
    
    struct memblock mem_out_140481;
    
    mem_out_140481.references = NULL;
    
    struct memblock mem_out_140480;
    
    mem_out_140480.references = NULL;
    
    struct memblock mem_out_140479;
    
    mem_out_140479.references = NULL;
    
    struct memblock mem_out_140478;
    
    mem_out_140478.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_140478, &mem_out_140479, &mem_out_140480, &mem_out_140481, &mem_out_140482, &mem_out_140483, &mem_out_140484, &mem_out_140485, &mem_out_140486);
        if (ret == 0) {
            struct memblock mem_138221 = ctx->constants->mem_138221;
            struct memblock mem_138222 = ctx->constants->mem_138222;
            struct memblock mem_138223 = ctx->constants->mem_138223;
            struct memblock mem_138224 = ctx->constants->mem_138224;
            struct memblock mem_138225 = ctx->constants->mem_138225;
            struct memblock mem_138226 = ctx->constants->mem_138226;
            struct memblock mem_138227 = ctx->constants->mem_138227;
            struct memblock mem_138228 = ctx->constants->mem_138228;
            struct memblock mem_138229 = ctx->constants->mem_138229;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_140478;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_140479;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_140480;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_140481;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_140482;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_140483;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_140484;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_140485;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_140486;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
