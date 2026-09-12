
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
struct futhark_f64_4d;
struct futhark_f64_4d *futhark_new_f64_4d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3);
struct futhark_f64_4d *futhark_new_raw_f64_4d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3);
int futhark_free_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr);
int futhark_values_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr, double *data);
int futhark_index_f64_4d(struct futhark_context *ctx, double *out, struct futhark_f64_4d *arr, int64_t i0, int64_t i1, int64_t i2, int64_t i3);
unsigned char *futhark_values_raw_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr);
const int64_t *futhark_shape_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr);
struct futhark_i64_1d;
struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0);
struct futhark_i64_1d *futhark_new_raw_i64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0);
int futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data);
int futhark_index_i64_1d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_1d *arr, int64_t i0);
unsigned char *futhark_values_raw_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
const int64_t *futhark_shape_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr);
struct futhark_i64_3d;
struct futhark_i64_3d *futhark_new_i64_3d(struct futhark_context *ctx, const int64_t *data, int64_t dim0, int64_t dim1, int64_t dim2);
struct futhark_i64_3d *futhark_new_raw_i64_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2);
int futhark_free_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);
int futhark_values_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr, int64_t *data);
int futhark_index_i64_3d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_3d *arr, int64_t i0, int64_t i1, int64_t i2);
unsigned char *futhark_values_raw_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);
const int64_t *futhark_shape_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);

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
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_4d *in3, const struct futhark_i64_3d *in4);
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
const struct type type_ZMZNZMZNZMZNZMZNf64;
const struct type type_ZMZNZMZNZMZNi64;
const struct type type_ZMZNZMZNf64;
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
void *futhark_new_f64_4d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f64_4d(ctx, p, shape[0], shape[1], shape[2], shape[3]);
}
int futhark_new_f64_4d_wrap(struct futhark_context *ctx, struct futhark_f64_4d * *outp, double *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 4; ++i)
        n_values *= shape[i];
    
    double *values = alloca(n_values * sizeof(double));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f64_4d(ctx, values, shape[0], shape[1], shape[2], shape[3]);
    return 0;
}
int futhark_new_f64_4d_set(struct futhark_context *ctx, struct futhark_f64_4d * arr, double *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f64_4d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 4; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((double *) futhark_values_raw_f64_4d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f64_4d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f64_4d * arr, const int64_t *is)
{
    return futhark_index_f64_4d(ctx, dest, arr, is[0], is[1], is[2], is[3]);
}
const struct array type_ZMZNZMZNZMZNZMZNf64_array = {.rank =4, .element_type =&type_f64, .new =(array_new_fn) futhark_new_f64_4d_wrap, .set =(array_set_fn) futhark_new_f64_4d_set, .shape =(array_shape_fn) futhark_shape_f64_4d, .index =(array_index_fn) futhark_index_f64_4d_wrap};
const struct array_aux type_ZMZNZMZNZMZNZMZNf64_aux = {.name ="[][][][]f64", .rank =4, .info =&f64_info, .new =(aux_array_new_fn) futhark_new_f64_4d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f64_4d, .shape =(aux_array_shape_fn) futhark_shape_f64_4d, .values =(aux_array_values_fn) futhark_values_f64_4d};
const struct type type_ZMZNZMZNZMZNZMZNf64 = {.name ="[][][][]f64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNZMZNZMZNf64_aux, .kind =ARRAY, .info =&type_ZMZNZMZNZMZNZMZNf64_array};
void *futhark_new_i64_3d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_i64_3d(ctx, p, shape[0], shape[1], shape[2]);
}
int futhark_new_i64_3d_wrap(struct futhark_context *ctx, struct futhark_i64_3d * *outp, int64_t *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 3; ++i)
        n_values *= shape[i];
    
    int64_t *values = alloca(n_values * sizeof(int64_t));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_i64_3d(ctx, values, shape[0], shape[1], shape[2]);
    return 0;
}
int futhark_new_i64_3d_set(struct futhark_context *ctx, struct futhark_i64_3d * arr, int64_t *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_i64_3d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 3; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((int64_t *) futhark_values_raw_i64_3d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_i64_3d_wrap(struct futhark_context *ctx, void *dest, struct futhark_i64_3d * arr, const int64_t *is)
{
    return futhark_index_i64_3d(ctx, dest, arr, is[0], is[1], is[2]);
}
const struct array type_ZMZNZMZNZMZNi64_array = {.rank =3, .element_type =&type_i64, .new =(array_new_fn) futhark_new_i64_3d_wrap, .set =(array_set_fn) futhark_new_i64_3d_set, .shape =(array_shape_fn) futhark_shape_i64_3d, .index =(array_index_fn) futhark_index_i64_3d_wrap};
const struct array_aux type_ZMZNZMZNZMZNi64_aux = {.name ="[][][]i64", .rank =3, .info =&i64_info, .new =(aux_array_new_fn) futhark_new_i64_3d_aux_wrap, .free =(aux_array_free_fn) futhark_free_i64_3d, .shape =(aux_array_shape_fn) futhark_shape_i64_3d, .values =(aux_array_values_fn) futhark_values_i64_3d};
const struct type type_ZMZNZMZNZMZNi64 = {.name ="[][][]i64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNZMZNi64_aux, .kind =ARRAY, .info =&type_ZMZNZMZNZMZNi64_array};
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
const struct type *train_in_types[] = {&type_params, &type_params, &type_params, &type_ZMZNZMZNZMZNZMZNf64, &type_ZMZNZMZNZMZNi64, NULL};
bool train_in_unique[] = {false, false, false, false, false};
const char *train_tuning_params[] = {NULL};
const char *train_attrs[] = {NULL};
int call_train(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_opaque_params * in2 = *(struct futhark_opaque_params * *) ins[2];
    struct futhark_f64_4d * in3 = *(struct futhark_f64_4d * *) ins[3];
    struct futhark_i64_3d * in4 = *(struct futhark_i64_3d * *) ins[4];
    
    return futhark_entry_train(ctx, out, in0, in1, in2, in3, in4);
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
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZMZNZMZNZMZNZMZNf64, &type_ZMZNZMZNZMZNi64, &type_ZMZNZMZNf64, &type_ZMZNi64, &type_params, NULL};
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
    struct memblock mem_83335;
    struct memblock mem_83336;
    struct memblock mem_83337;
    struct memblock mem_83338;
    struct memblock mem_83339;
    struct memblock mem_83340;
    struct memblock mem_83341;
    struct memblock mem_83342;
    struct memblock mem_83343;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10353(struct futhark_context *ctx, struct memblock *mem_out_p_85461, struct memblock *mem_out_p_85462, struct memblock *mem_out_p_85463, struct memblock w_mem_83344, struct memblock mw_mem_83345, struct memblock vw_mem_83346, struct memblock dw_mem_83347, int64_t n_60344, int64_t m_60345, int64_t step_60350, double lt_r_60351);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10354(struct futhark_context *ctx, struct memblock *mem_out_p_85466, struct memblock *mem_out_p_85467, struct memblock *mem_out_p_85468, struct memblock w_mem_83344, struct memblock mw_mem_83345, struct memblock vw_mem_83346, struct memblock dw_mem_83347, int64_t n_61377, int64_t m_61378, int64_t step_61383, double lt_r_61384);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85471, struct memblock wdown_mem_83344, struct memblock wkey_mem_83345, struct memblock wout_mem_83346, struct memblock wpe_mem_83347, struct memblock wqry_mem_83348, struct memblock wte_mem_83349, struct memblock wup_mem_83350, struct memblock wval_mem_83351, struct memblock wvoc_mem_83352, struct memblock tokens_mem_83353, struct memblock mask_mem_83354);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock wte_mem_83344, struct memblock wpe_mem_83345, struct memblock wqry_mem_83346, struct memblock wkey_mem_83347, struct memblock wval_mem_83348, struct memblock wout_mem_83349, struct memblock wup_mem_83350, struct memblock wdown_mem_83351, struct memblock wvoc_mem_83352);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock *mem_out_p_85540, struct memblock *mem_out_p_85541, struct memblock *mem_out_p_85542, struct memblock *mem_out_p_85543, struct memblock *mem_out_p_85544, struct memblock *mem_out_p_85545, struct memblock *mem_out_p_85546, struct memblock *mem_out_p_85547, struct memblock *mem_out_p_85548, struct memblock *mem_out_p_85549, struct memblock *mem_out_p_85550, struct memblock *mem_out_p_85551, struct memblock *mem_out_p_85552, struct memblock *mem_out_p_85553, struct memblock *mem_out_p_85554, struct memblock *mem_out_p_85555, struct memblock *mem_out_p_85556, struct memblock *mem_out_p_85557, struct memblock *mem_out_p_85558, struct memblock *mem_out_p_85559, struct memblock *mem_out_p_85560, struct memblock *mem_out_p_85561, struct memblock wdown_mem_83344, struct memblock wkey_mem_83345, struct memblock wout_mem_83346, struct memblock wpe_mem_83347, struct memblock wqry_mem_83348, struct memblock wte_mem_83349, struct memblock wup_mem_83350, struct memblock wval_mem_83351, struct memblock wvoc_mem_83352, struct memblock wdown_mem_83353, struct memblock wkey_mem_83354, struct memblock wout_mem_83355, struct memblock wpe_mem_83356, struct memblock wqry_mem_83357, struct memblock wte_mem_83358, struct memblock wup_mem_83359, struct memblock wval_mem_83360, struct memblock wvoc_mem_83361, struct memblock wdown_mem_83362, struct memblock wkey_mem_83363, struct memblock wout_mem_83364, struct memblock wpe_mem_83365, struct memblock wqry_mem_83366, struct memblock wte_mem_83367, struct memblock wup_mem_83368, struct memblock wval_mem_83369, struct memblock wvoc_mem_83370, struct memblock masks_mem_83371, struct memblock seqs_mem_83372, int64_t num_batches_62011, int64_t batchsizze_62012);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85731, struct memblock *mem_out_p_85732, struct memblock *mem_out_p_85733, struct memblock *mem_out_p_85734, struct memblock *mem_out_p_85735, struct memblock *mem_out_p_85736, struct memblock *mem_out_p_85737, struct memblock *mem_out_p_85738, struct memblock *mem_out_p_85739);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_83335 (ctx->constants->mem_83335)
    #define mem_83336 (ctx->constants->mem_83336)
    #define mem_83337 (ctx->constants->mem_83337)
    #define mem_83338 (ctx->constants->mem_83338)
    #define mem_83339 (ctx->constants->mem_83339)
    #define mem_83340 (ctx->constants->mem_83340)
    #define mem_83341 (ctx->constants->mem_83341)
    #define mem_83342 (ctx->constants->mem_83342)
    #define mem_83343 (ctx->constants->mem_83343)
    mem_83335.references = NULL;
    mem_83336.references = NULL;
    mem_83337.references = NULL;
    mem_83338.references = NULL;
    mem_83339.references = NULL;
    mem_83340.references = NULL;
    mem_83341.references = NULL;
    mem_83342.references = NULL;
    mem_83343.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83335, (int64_t) 3456, "mem_83335")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85443 = 0; nest_i_85443 < (int64_t) 27; nest_i_85443++) {
        for (int64_t nest_i_85444 = 0; nest_i_85444 < (int64_t) 16; nest_i_85444++) {
            ((double *) mem_83335.mem)[nest_i_85443 * (int64_t) 16 + nest_i_85444] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83336, (int64_t) 2048, "mem_83336")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85445 = 0; nest_i_85445 < (int64_t) 16; nest_i_85445++) {
        for (int64_t nest_i_85446 = 0; nest_i_85446 < (int64_t) 16; nest_i_85446++) {
            ((double *) mem_83336.mem)[nest_i_85445 * (int64_t) 16 + nest_i_85446] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83337, (int64_t) 2048, "mem_83337")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85447 = 0; nest_i_85447 < (int64_t) 16; nest_i_85447++) {
        for (int64_t nest_i_85448 = 0; nest_i_85448 < (int64_t) 16; nest_i_85448++) {
            ((double *) mem_83337.mem)[nest_i_85447 * (int64_t) 16 + nest_i_85448] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83338, (int64_t) 2048, "mem_83338")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85449 = 0; nest_i_85449 < (int64_t) 16; nest_i_85449++) {
        for (int64_t nest_i_85450 = 0; nest_i_85450 < (int64_t) 16; nest_i_85450++) {
            ((double *) mem_83338.mem)[nest_i_85449 * (int64_t) 16 + nest_i_85450] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83339, (int64_t) 2048, "mem_83339")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85451 = 0; nest_i_85451 < (int64_t) 16; nest_i_85451++) {
        for (int64_t nest_i_85452 = 0; nest_i_85452 < (int64_t) 16; nest_i_85452++) {
            ((double *) mem_83339.mem)[nest_i_85451 * (int64_t) 16 + nest_i_85452] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83340, (int64_t) 2048, "mem_83340")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85453 = 0; nest_i_85453 < (int64_t) 16; nest_i_85453++) {
        for (int64_t nest_i_85454 = 0; nest_i_85454 < (int64_t) 16; nest_i_85454++) {
            ((double *) mem_83340.mem)[nest_i_85453 * (int64_t) 16 + nest_i_85454] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83341, (int64_t) 8192, "mem_83341")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85455 = 0; nest_i_85455 < (int64_t) 64; nest_i_85455++) {
        for (int64_t nest_i_85456 = 0; nest_i_85456 < (int64_t) 16; nest_i_85456++) {
            ((double *) mem_83341.mem)[nest_i_85455 * (int64_t) 16 + nest_i_85456] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83342, (int64_t) 8192, "mem_83342")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85457 = 0; nest_i_85457 < (int64_t) 16; nest_i_85457++) {
        for (int64_t nest_i_85458 = 0; nest_i_85458 < (int64_t) 64; nest_i_85458++) {
            ((double *) mem_83342.mem)[nest_i_85457 * (int64_t) 64 + nest_i_85458] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83343, (int64_t) 3456, "mem_83343")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85459 = 0; nest_i_85459 < (int64_t) 27; nest_i_85459++) {
        for (int64_t nest_i_85460 = 0; nest_i_85460 < (int64_t) 16; nest_i_85460++) {
            ((double *) mem_83343.mem)[nest_i_85459 * (int64_t) 16 + nest_i_85460] = 0.0;
        }
    }
    #undef mem_83335
    #undef mem_83336
    #undef mem_83337
    #undef mem_83338
    #undef mem_83339
    #undef mem_83340
    #undef mem_83341
    #undef mem_83342
    #undef mem_83343
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_83335, "ctx->constants->mem_83335") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83336, "ctx->constants->mem_83336") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83337, "ctx->constants->mem_83337") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83338, "ctx->constants->mem_83338") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83339, "ctx->constants->mem_83339") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83340, "ctx->constants->mem_83340") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83341, "ctx->constants->mem_83341") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83342, "ctx->constants->mem_83342") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83343, "ctx->constants->mem_83343") != 0)
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
struct futhark_i64_3d {
    struct memblock mem;
    int64_t shape[3];
};
struct futhark_i64_3d *futhark_new_i64_3d(struct futhark_context *ctx, const int64_t *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_i64_3d *bad = NULL;
    struct futhark_i64_3d *arr = (struct futhark_i64_3d *) malloc(sizeof(struct futhark_i64_3d));
    
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
struct futhark_i64_3d *futhark_new_raw_i64_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_i64_3d *bad = NULL;
    struct futhark_i64_3d *arr = (struct futhark_i64_3d *) malloc(sizeof(struct futhark_i64_3d));
    
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
int futhark_free_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr, int64_t *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_i64_3d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_3d *arr, int64_t i0, int64_t i1, int64_t i2)
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
unsigned char *futhark_values_raw_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr)
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
struct futhark_f64_4d {
    struct memblock mem;
    int64_t shape[4];
};
struct futhark_f64_4d *futhark_new_f64_4d(struct futhark_context *ctx, const double *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
{
    int err = 0;
    struct futhark_f64_4d *bad = NULL;
    struct futhark_f64_4d *arr = (struct futhark_f64_4d *) malloc(sizeof(struct futhark_f64_4d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    arr->shape[3] = dim3;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3] * 8, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1 * dim2 * dim3) * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1 * dim2 * dim3) * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f64_4d *futhark_new_raw_f64_4d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
{
    int err = 0;
    struct futhark_f64_4d *bad = NULL;
    struct futhark_f64_4d *arr = (struct futhark_f64_4d *) malloc(sizeof(struct futhark_f64_4d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    arr->shape[3] = dim3;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr, double *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3]) * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3]) * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f64_4d(struct futhark_context *ctx, double *out, struct futhark_f64_4d *arr, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && ((i1 >= 0 && i1 < arr->shape[1]) && ((i2 >= 0 && i2 < arr->shape[2]) && (i3 >= 0 && i3 < arr->shape[3])))) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * (arr->shape[1] * arr->shape[2] * arr->shape[3]) + i1 * (arr->shape[2] * arr->shape[3]) + i2 * arr->shape[3] + i3 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_4d(struct futhark_context *ctx, struct futhark_f64_4d *arr)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10353(struct futhark_context *ctx, struct memblock *mem_out_p_85461, struct memblock *mem_out_p_85462, struct memblock *mem_out_p_85463, struct memblock w_mem_83344, struct memblock mw_mem_83345, struct memblock vw_mem_83346, struct memblock dw_mem_83347, int64_t n_60344, int64_t m_60345, int64_t step_60350, double lt_r_60351)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83388_cached_sizze_85464 = 0;
    unsigned char *mem_83388 = NULL;
    int64_t mem_83391_cached_sizze_85465 = 0;
    unsigned char *mem_83391 = NULL;
    struct memblock mem_83426;
    
    mem_83426.references = NULL;
    
    struct memblock mem_83353;
    
    mem_83353.references = NULL;
    
    struct memblock mem_83350;
    
    mem_83350.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83348 = (int64_t) 8 * n_60344;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83349 = m_60345 * binop_x_83348;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83350, bytes_83349, "mem_83350")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83353, bytes_83349, "mem_83353")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82496 = 0; i_82496 < n_60344; i_82496++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82489 = 0; i_82489 < m_60345; i_82489++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78609 = ((double *) mw_mem_83345.mem)[i_82496 * m_60345 + i_82489];
            
            // futhark/microgpt.fut:382:10-20
            
            double zp_lhs_78610 = 0.85 * zt_rhs_78609;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78611 = ((double *) dw_mem_83347.mem)[i_82496 * m_60345 + i_82489];
            
            // futhark/microgpt.fut:382:35-45
            
            double zp_rhs_78612 = 0.15000000000000002 * zt_rhs_78611;
            
            // futhark/microgpt.fut:382:21-45
            
            double lifted_lambda_res_78613 = zp_lhs_78610 + zp_rhs_78612;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78620 = ((double *) vw_mem_83346.mem)[i_82496 * m_60345 + i_82489];
            
            // futhark/microgpt.fut:384:10-20
            
            double zp_lhs_78621 = 0.99 * zt_rhs_78620;
            
            // futhark/microgpt.fut:384:35-45
            
            double zt_lhs_78623 = 1.0000000000000009e-2 * zt_rhs_78611;
            
            // futhark/microgpt.fut:384:46-56
            
            double zp_rhs_78624 = zt_rhs_78611 * zt_lhs_78623;
            
            // futhark/microgpt.fut:384:21-56
            
            double lifted_lambda_res_78625 = zp_lhs_78621 + zp_rhs_78624;
            
            ((double *) mem_83350.mem)[i_82496 * m_60345 + i_82489] = lifted_lambda_res_78625;
            ((double *) mem_83353.mem)[i_82496 * m_60345 + i_82489] = lifted_lambda_res_78613;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65360 = sitofp_i64_f64(step_60350);
    
    // futhark/microgpt.fut:386:54-57
    
    double ztzt_rhs_65361 = 1.0 + i64_res_65360;
    
    // futhark/microgpt.fut:386:30-57
    
    double zm_rhs_65362 = fpow64(0.85, ztzt_rhs_65361);
    
    // futhark/microgpt.fut:386:23-57
    
    double zs_rhs_65363 = 1.0 - zm_rhs_65362;
    
    // futhark/microgpt.fut:388:31-58
    
    double zm_rhs_65401 = fpow64(0.99, ztzt_rhs_65361);
    
    // futhark/microgpt.fut:388:23-58
    
    double zs_rhs_65402 = 1.0 - zm_rhs_65401;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83388_cached_sizze_85464 < bytes_83349) {
        err = lexical_realloc(ctx, &mem_83388, &mem_83388_cached_sizze_85464, bytes_83349);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83391_cached_sizze_85465 < bytes_83349) {
        err = lexical_realloc(ctx, &mem_83391, &mem_83391_cached_sizze_85465, bytes_83349);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82510 = 0; i_82510 < n_60344; i_82510++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82503 = 0; i_82503 < m_60345; i_82503++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78645 = ((double *) mem_83353.mem)[i_82510 * m_60345 + i_82503];
            
            // futhark/microgpt.fut:386:18-57
            
            double lifted_lambda_res_78646 = zs_lhs_78645 / zs_rhs_65363;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78653 = ((double *) mem_83350.mem)[i_82510 * m_60345 + i_82503];
            
            // futhark/microgpt.fut:388:18-58
            
            double lifted_lambda_res_78654 = zs_lhs_78653 / zs_rhs_65402;
            
            ((double *) mem_83388)[i_82510 * m_60345 + i_82503] = lifted_lambda_res_78654;
            ((double *) mem_83391)[i_82510 * m_60345 + i_82503] = lifted_lambda_res_78646;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83426, bytes_83349, "mem_83426")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82519 = 0; i_82519 < n_60344; i_82519++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82515 = 0; i_82515 < m_60345; i_82515++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64524 = ((double *) w_mem_83344.mem)[i_82519 * m_60345 + i_82515];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64525 = ((double *) mem_83391)[i_82519 * m_60345 + i_82515];
            
            // futhark/microgpt.fut:390:21-34
            
            double zs_lhs_64526 = lt_r_60351 * zt_rhs_64525;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64527 = ((double *) mem_83388)[i_82519 * m_60345 + i_82515];
            
            // futhark/microgpt.fut:390:51-57
            
            double zp_lhs_64528 = fpow64(ztzt_lhs_64527, 0.5);
            
            // futhark/microgpt.fut:390:59-71
            
            double zs_rhs_64529 = 1.0e-8 + zp_lhs_64528;
            
            // futhark/microgpt.fut:390:35-71
            
            double zm_rhs_64530 = zs_lhs_64526 / zs_rhs_64529;
            
            // futhark/microgpt.fut:390:13-71
            
            double lifted_lambda_res_64531 = zm_lhs_64524 - zm_rhs_64530;
            
            ((double *) mem_83426.mem)[i_82519 * m_60345 + i_82515] = lifted_lambda_res_64531;
        }
    }
    if (memblock_set(ctx, &mem_out_85142, &mem_83426, "mem_83426") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &mem_83353, "mem_83353") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &mem_83350, "mem_83350") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85461, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85462, &mem_out_85143, "mem_out_85143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85463, &mem_out_85144, "mem_out_85144") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83388);
        free(mem_83391);
        if (memblock_unref(ctx, &mem_83426, "mem_83426") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83353, "mem_83353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83350, "mem_83350") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85144, "mem_out_85144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85143, "mem_out_85143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10354(struct futhark_context *ctx, struct memblock *mem_out_p_85466, struct memblock *mem_out_p_85467, struct memblock *mem_out_p_85468, struct memblock w_mem_83344, struct memblock mw_mem_83345, struct memblock vw_mem_83346, struct memblock dw_mem_83347, int64_t n_61377, int64_t m_61378, int64_t step_61383, double lt_r_61384)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83388_cached_sizze_85469 = 0;
    unsigned char *mem_83388 = NULL;
    int64_t mem_83391_cached_sizze_85470 = 0;
    unsigned char *mem_83391 = NULL;
    struct memblock mem_83426;
    
    mem_83426.references = NULL;
    
    struct memblock mem_83353;
    
    mem_83353.references = NULL;
    
    struct memblock mem_83350;
    
    mem_83350.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83348 = (int64_t) 8 * n_61377;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83349 = m_61378 * binop_x_83348;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83350, bytes_83349, "mem_83350")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83353, bytes_83349, "mem_83353")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82496 = 0; i_82496 < n_61377; i_82496++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82489 = 0; i_82489 < m_61378; i_82489++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78609 = ((double *) mw_mem_83345.mem)[i_82496 * m_61378 + i_82489];
            
            // futhark/microgpt.fut:382:10-20
            
            double zp_lhs_78610 = 0.85 * zt_rhs_78609;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78611 = ((double *) dw_mem_83347.mem)[i_82496 * m_61378 + i_82489];
            
            // futhark/microgpt.fut:382:35-45
            
            double zp_rhs_78612 = 0.15000000000000002 * zt_rhs_78611;
            
            // futhark/microgpt.fut:382:21-45
            
            double lifted_lambda_res_78613 = zp_lhs_78610 + zp_rhs_78612;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78620 = ((double *) vw_mem_83346.mem)[i_82496 * m_61378 + i_82489];
            
            // futhark/microgpt.fut:384:10-20
            
            double zp_lhs_78621 = 0.99 * zt_rhs_78620;
            
            // futhark/microgpt.fut:384:35-45
            
            double zt_lhs_78623 = 1.0000000000000009e-2 * zt_rhs_78611;
            
            // futhark/microgpt.fut:384:46-56
            
            double zp_rhs_78624 = zt_rhs_78611 * zt_lhs_78623;
            
            // futhark/microgpt.fut:384:21-56
            
            double lifted_lambda_res_78625 = zp_lhs_78621 + zp_rhs_78624;
            
            ((double *) mem_83350.mem)[i_82496 * m_61378 + i_82489] = lifted_lambda_res_78625;
            ((double *) mem_83353.mem)[i_82496 * m_61378 + i_82489] = lifted_lambda_res_78613;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65360 = sitofp_i64_f64(step_61383);
    
    // futhark/microgpt.fut:386:54-57
    
    double ztzt_rhs_65361 = 1.0 + i64_res_65360;
    
    // futhark/microgpt.fut:386:30-57
    
    double zm_rhs_65362 = fpow64(0.85, ztzt_rhs_65361);
    
    // futhark/microgpt.fut:386:23-57
    
    double zs_rhs_65363 = 1.0 - zm_rhs_65362;
    
    // futhark/microgpt.fut:388:31-58
    
    double zm_rhs_65401 = fpow64(0.99, ztzt_rhs_65361);
    
    // futhark/microgpt.fut:388:23-58
    
    double zs_rhs_65402 = 1.0 - zm_rhs_65401;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83388_cached_sizze_85469 < bytes_83349) {
        err = lexical_realloc(ctx, &mem_83388, &mem_83388_cached_sizze_85469, bytes_83349);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83391_cached_sizze_85470 < bytes_83349) {
        err = lexical_realloc(ctx, &mem_83391, &mem_83391_cached_sizze_85470, bytes_83349);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82510 = 0; i_82510 < n_61377; i_82510++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82503 = 0; i_82503 < m_61378; i_82503++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78645 = ((double *) mem_83353.mem)[i_82510 * m_61378 + i_82503];
            
            // futhark/microgpt.fut:386:18-57
            
            double lifted_lambda_res_78646 = zs_lhs_78645 / zs_rhs_65363;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78653 = ((double *) mem_83350.mem)[i_82510 * m_61378 + i_82503];
            
            // futhark/microgpt.fut:388:18-58
            
            double lifted_lambda_res_78654 = zs_lhs_78653 / zs_rhs_65402;
            
            ((double *) mem_83388)[i_82510 * m_61378 + i_82503] = lifted_lambda_res_78654;
            ((double *) mem_83391)[i_82510 * m_61378 + i_82503] = lifted_lambda_res_78646;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83426, bytes_83349, "mem_83426")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82519 = 0; i_82519 < n_61377; i_82519++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82515 = 0; i_82515 < m_61378; i_82515++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64524 = ((double *) w_mem_83344.mem)[i_82519 * m_61378 + i_82515];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64525 = ((double *) mem_83391)[i_82519 * m_61378 + i_82515];
            
            // futhark/microgpt.fut:390:21-34
            
            double zs_lhs_64526 = lt_r_61384 * zt_rhs_64525;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64527 = ((double *) mem_83388)[i_82519 * m_61378 + i_82515];
            
            // futhark/microgpt.fut:390:51-57
            
            double zp_lhs_64528 = fpow64(ztzt_lhs_64527, 0.5);
            
            // futhark/microgpt.fut:390:59-71
            
            double zs_rhs_64529 = 1.0e-8 + zp_lhs_64528;
            
            // futhark/microgpt.fut:390:35-71
            
            double zm_rhs_64530 = zs_lhs_64526 / zs_rhs_64529;
            
            // futhark/microgpt.fut:390:13-71
            
            double lifted_lambda_res_64531 = zm_lhs_64524 - zm_rhs_64530;
            
            ((double *) mem_83426.mem)[i_82519 * m_61378 + i_82515] = lifted_lambda_res_64531;
        }
    }
    if (memblock_set(ctx, &mem_out_85142, &mem_83426, "mem_83426") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &mem_83353, "mem_83353") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &mem_83350, "mem_83350") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85466, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85467, &mem_out_85143, "mem_out_85143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85468, &mem_out_85144, "mem_out_85144") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83388);
        free(mem_83391);
        if (memblock_unref(ctx, &mem_83426, "mem_83426") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83353, "mem_83353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83350, "mem_83350") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85144, "mem_out_85144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85143, "mem_out_85143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85471, struct memblock wdown_mem_83344, struct memblock wkey_mem_83345, struct memblock wout_mem_83346, struct memblock wpe_mem_83347, struct memblock wqry_mem_83348, struct memblock wte_mem_83349, struct memblock wup_mem_83350, struct memblock wval_mem_83351, struct memblock wvoc_mem_83352, struct memblock tokens_mem_83353, struct memblock mask_mem_83354)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83355_cached_sizze_85472 = 0;
    unsigned char *mem_83355 = NULL;
    int64_t mem_83360_cached_sizze_85473 = 0;
    unsigned char *mem_83360 = NULL;
    int64_t mem_83371_cached_sizze_85474 = 0;
    unsigned char *mem_83371 = NULL;
    int64_t mem_83376_cached_sizze_85475 = 0;
    unsigned char *mem_83376 = NULL;
    int64_t mem_83387_cached_sizze_85476 = 0;
    unsigned char *mem_83387 = NULL;
    int64_t mem_83392_cached_sizze_85477 = 0;
    unsigned char *mem_83392 = NULL;
    int64_t mem_83399_cached_sizze_85478 = 0;
    unsigned char *mem_83399 = NULL;
    int64_t mem_83410_cached_sizze_85479 = 0;
    unsigned char *mem_83410 = NULL;
    int64_t mem_83415_cached_sizze_85480 = 0;
    unsigned char *mem_83415 = NULL;
    int64_t mem_83422_cached_sizze_85481 = 0;
    unsigned char *mem_83422 = NULL;
    int64_t mem_83433_cached_sizze_85482 = 0;
    unsigned char *mem_83433 = NULL;
    int64_t mem_83434_cached_sizze_85483 = 0;
    unsigned char *mem_83434 = NULL;
    int64_t mem_83435_cached_sizze_85484 = 0;
    unsigned char *mem_83435 = NULL;
    int64_t mem_83448_cached_sizze_85485 = 0;
    unsigned char *mem_83448 = NULL;
    int64_t mem_83449_cached_sizze_85486 = 0;
    unsigned char *mem_83449 = NULL;
    int64_t mem_83450_cached_sizze_85487 = 0;
    unsigned char *mem_83450 = NULL;
    int64_t mem_83481_cached_sizze_85488 = 0;
    unsigned char *mem_83481 = NULL;
    int64_t mem_83482_cached_sizze_85489 = 0;
    unsigned char *mem_83482 = NULL;
    int64_t mem_83483_cached_sizze_85490 = 0;
    unsigned char *mem_83483 = NULL;
    int64_t mem_83499_cached_sizze_85491 = 0;
    unsigned char *mem_83499 = NULL;
    int64_t mem_83500_cached_sizze_85492 = 0;
    unsigned char *mem_83500 = NULL;
    int64_t mem_83501_cached_sizze_85493 = 0;
    unsigned char *mem_83501 = NULL;
    int64_t mem_83514_cached_sizze_85494 = 0;
    unsigned char *mem_83514 = NULL;
    int64_t mem_83515_cached_sizze_85495 = 0;
    unsigned char *mem_83515 = NULL;
    int64_t mem_83516_cached_sizze_85496 = 0;
    unsigned char *mem_83516 = NULL;
    int64_t mem_83562_cached_sizze_85497 = 0;
    unsigned char *mem_83562 = NULL;
    int64_t mem_83568_cached_sizze_85498 = 0;
    unsigned char *mem_83568 = NULL;
    int64_t mem_83573_cached_sizze_85499 = 0;
    unsigned char *mem_83573 = NULL;
    int64_t mem_83584_cached_sizze_85500 = 0;
    unsigned char *mem_83584 = NULL;
    int64_t mem_83589_cached_sizze_85501 = 0;
    unsigned char *mem_83589 = NULL;
    int64_t mem_83600_cached_sizze_85502 = 0;
    unsigned char *mem_83600 = NULL;
    int64_t mem_83605_cached_sizze_85503 = 0;
    unsigned char *mem_83605 = NULL;
    int64_t mem_83612_cached_sizze_85504 = 0;
    unsigned char *mem_83612 = NULL;
    int64_t mem_83619_cached_sizze_85505 = 0;
    unsigned char *mem_83619 = NULL;
    int64_t mem_83630_cached_sizze_85506 = 0;
    unsigned char *mem_83630 = NULL;
    int64_t mem_83635_cached_sizze_85507 = 0;
    unsigned char *mem_83635 = NULL;
    int64_t mem_83651_cached_sizze_85508 = 0;
    unsigned char *mem_83651 = NULL;
    int64_t mem_83656_cached_sizze_85509 = 0;
    unsigned char *mem_83656 = NULL;
    int64_t mem_83667_cached_sizze_85510 = 0;
    unsigned char *mem_83667 = NULL;
    int64_t mem_83672_cached_sizze_85511 = 0;
    unsigned char *mem_83672 = NULL;
    int64_t mem_83683_cached_sizze_85512 = 0;
    unsigned char *mem_83683 = NULL;
    int64_t mem_83688_cached_sizze_85513 = 0;
    unsigned char *mem_83688 = NULL;
    int64_t mem_83699_cached_sizze_85514 = 0;
    unsigned char *mem_83699 = NULL;
    int64_t mem_83704_cached_sizze_85515 = 0;
    unsigned char *mem_83704 = NULL;
    int64_t mem_83711_cached_sizze_85516 = 0;
    unsigned char *mem_83711 = NULL;
    int64_t mem_83722_cached_sizze_85517 = 0;
    unsigned char *mem_83722 = NULL;
    int64_t mem_83727_cached_sizze_85518 = 0;
    unsigned char *mem_83727 = NULL;
    int64_t mem_83738_cached_sizze_85519 = 0;
    unsigned char *mem_83738 = NULL;
    int64_t mem_83743_cached_sizze_85520 = 0;
    unsigned char *mem_83743 = NULL;
    int64_t mem_83754_cached_sizze_85521 = 0;
    unsigned char *mem_83754 = NULL;
    int64_t mem_83759_cached_sizze_85522 = 0;
    unsigned char *mem_83759 = NULL;
    int64_t mem_83770_cached_sizze_85523 = 0;
    unsigned char *mem_83770 = NULL;
    int64_t mem_83775_cached_sizze_85524 = 0;
    unsigned char *mem_83775 = NULL;
    int64_t mem_83791_cached_sizze_85525 = 0;
    unsigned char *mem_83791 = NULL;
    struct memblock mem_83786;
    
    mem_83786.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83355_cached_sizze_85472 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83355, &mem_83355_cached_sizze_85472, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83360_cached_sizze_85473 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83360, &mem_83360_cached_sizze_85473, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82491 = 0; i_82491 < (int64_t) 16; i_82491++) {
        // futhark/microgpt.fut:348:41-50
        
        int64_t tmp_73027 = ((int64_t *) tokens_mem_83353.mem)[i_82491];
        
        // futhark/microgpt.fut:348:37-51
        
        bool x_73028 = sle64((int64_t) 0, tmp_73027);
        
        // futhark/microgpt.fut:348:37-51
        
        bool y_73029 = slt64(tmp_73027, (int64_t) 27);
        
        // futhark/microgpt.fut:348:37-51
        
        bool bounds_check_73030 = x_73028 && y_73029;
        
        // futhark/microgpt.fut:348:37-51
        
        bool index_certs_73031;
        
        if (!bounds_check_73030) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73027, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:348:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:348:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82487 = 0; i_82487 < (int64_t) 16; i_82487++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73038 = ((double *) wte_mem_83349.mem)[tmp_73027 * (int64_t) 16 + i_82487];
            
            ((double *) mem_83360)[i_82487] = lifted_lambda_res_73038;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83355, i_82491 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83360, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83371_cached_sizze_85474 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83371, &mem_83371_cached_sizze_85474, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83376_cached_sizze_85475 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83376, &mem_83376_cached_sizze_85475, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82499 = 0; i_82499 < (int64_t) 16; i_82499++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82495 = 0; i_82495 < (int64_t) 16; i_82495++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73070 = ((double *) wpe_mem_83347.mem)[i_82499 * (int64_t) 16 + i_82495];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73071 = ((double *) mem_83355)[i_82499 * (int64_t) 16 + i_82495];
            
            // futhark/microgpt.fut:149:38-70
            
            double zp_res_73072 = zp_lhs_73070 + zp_rhs_73071;
            
            ((double *) mem_83376)[i_82495] = zp_res_73072;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83371, i_82499 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83376, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83387_cached_sizze_85476 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83387, &mem_83387_cached_sizze_85476, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83392_cached_sizze_85477 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83392, &mem_83392_cached_sizze_85477, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83399_cached_sizze_85478 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83399, &mem_83399_cached_sizze_85478, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82511 = 0; i_82511 < (int64_t) 16; i_82511++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82503 = 0; i_82503 < (int64_t) 16; i_82503++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73087 = ((double *) mem_83371)[i_82511 * (int64_t) 16 + i_82503];
            
            // futhark/microgpt.fut:150:64-93
            
            double zt_res_73088 = zt_lhs_73087 * zt_lhs_73087;
            
            ((double *) mem_83392)[i_82503] = zt_res_73088;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73090;
        double r_73092 = 0.0;
        
        for (int64_t i_73091 = 0; i_73091 < (int64_t) 16; i_73091++) {
            // futhark/microgpt.fut:151:35-43
            
            double lifted_lambda_res_73093 = ((double *) mem_83392)[i_73091];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73094 = r_73092 + lifted_lambda_res_73093;
            double r_tmp_85149 = zp_res_73094;
            
            r_73092 = r_tmp_85149;
        }
        defunc_0_lifted_lambda_res_73090 = r_73092;
        // futhark/microgpt.fut:151:17-60
        
        double zs_res_73095 = defunc_0_lifted_lambda_res_73090 / 16.0;
        
        // futhark/microgpt.fut:152:24-55
        
        double zp_res_73096 = 1.0e-5 + zs_res_73095;
        
        // futhark/microgpt.fut:152:16-55
        
        double sqrt_res_73097 = futrts_sqrt64(zp_res_73096);
        
        // futhark/microgpt.fut:153:42-53
        
        double zs_res_73098 = 1.0 / sqrt_res_73097;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82507 = 0; i_82507 < (int64_t) 16; i_82507++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73105 = ((double *) mem_83371)[i_82511 * (int64_t) 16 + i_82507];
            
            // futhark/microgpt.fut:153:24-53
            
            double zt_res_73106 = zs_res_73098 * zt_lhs_73105;
            
            ((double *) mem_83399)[i_82507] = zt_res_73106;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83387, i_82511 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83399, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83410_cached_sizze_85479 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83410, &mem_83410_cached_sizze_85479, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83415_cached_sizze_85480 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83415, &mem_83415_cached_sizze_85480, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83422_cached_sizze_85481 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83422, &mem_83422_cached_sizze_85481, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82523 = 0; i_82523 < (int64_t) 16; i_82523++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82515 = 0; i_82515 < (int64_t) 16; i_82515++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73121 = ((double *) mem_83387)[i_82523 * (int64_t) 16 + i_82515];
            
            // futhark/microgpt.fut:154:64-93
            
            double zt_res_73122 = zt_lhs_73121 * zt_lhs_73121;
            
            ((double *) mem_83415)[i_82515] = zt_res_73122;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73124;
        double r_73126 = 0.0;
        
        for (int64_t i_73125 = 0; i_73125 < (int64_t) 16; i_73125++) {
            // futhark/microgpt.fut:155:35-43
            
            double lifted_lambda_res_73127 = ((double *) mem_83415)[i_73125];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73128 = r_73126 + lifted_lambda_res_73127;
            double r_tmp_85153 = zp_res_73128;
            
            r_73126 = r_tmp_85153;
        }
        defunc_0_lifted_lambda_res_73124 = r_73126;
        // futhark/microgpt.fut:155:17-60
        
        double zs_res_73129 = defunc_0_lifted_lambda_res_73124 / 16.0;
        
        // futhark/microgpt.fut:156:24-55
        
        double zp_res_73130 = 1.0e-5 + zs_res_73129;
        
        // futhark/microgpt.fut:156:16-55
        
        double sqrt_res_73131 = futrts_sqrt64(zp_res_73130);
        
        // futhark/microgpt.fut:157:42-53
        
        double zs_res_73132 = 1.0 / sqrt_res_73131;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82519 = 0; i_82519 < (int64_t) 16; i_82519++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73139 = ((double *) mem_83387)[i_82523 * (int64_t) 16 + i_82519];
            
            // futhark/microgpt.fut:157:24-53
            
            double zt_res_73140 = zs_res_73132 * zt_lhs_73139;
            
            ((double *) mem_83422)[i_82519] = zt_res_73140;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83410, i_82523 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83422, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83433_cached_sizze_85482 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83433, &mem_83433_cached_sizze_85482, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83434_cached_sizze_85483 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83434, &mem_83434_cached_sizze_85483, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83435_cached_sizze_85484 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83435, &mem_83435_cached_sizze_85484, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83448_cached_sizze_85485 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83448, &mem_83448_cached_sizze_85485, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83449_cached_sizze_85486 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83449, &mem_83449_cached_sizze_85486, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83450_cached_sizze_85487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83450, &mem_83450_cached_sizze_85487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82541 = 0; i_82541 < (int64_t) 16; i_82541++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82531 = 0; i_82531 < (int64_t) 16; i_82531++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78828;
            double r_78830 = 0.0;
            
            for (int64_t i_78829 = 0; i_78829 < (int64_t) 16; i_78829++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78831 = ((double *) wqry_mem_83348.mem)[i_82531 * (int64_t) 16 + i_78829];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78832 = ((double *) mem_83410)[i_82541 * (int64_t) 16 + i_78829];
                
                // futhark/microgpt.fut:158:72-103
                
                double zt_res_78833 = zt_lhs_78831 * zt_rhs_78832;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78834 = r_78830 + zt_res_78833;
                double r_tmp_85161 = zp_res_78834;
                
                r_78830 = r_tmp_85161;
            }
            defunc_0_lifted_lambda_res_78828 = r_78830;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78841;
            double r_78843 = 0.0;
            
            for (int64_t i_78842 = 0; i_78842 < (int64_t) 16; i_78842++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78844 = ((double *) wkey_mem_83345.mem)[i_82531 * (int64_t) 16 + i_78842];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78845 = ((double *) mem_83410)[i_82541 * (int64_t) 16 + i_78842];
                
                // futhark/microgpt.fut:159:72-103
                
                double zt_res_78846 = zt_lhs_78844 * zt_rhs_78845;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78847 = r_78843 + zt_res_78846;
                double r_tmp_85162 = zp_res_78847;
                
                r_78843 = r_tmp_85162;
            }
            defunc_0_lifted_lambda_res_78841 = r_78843;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78857;
            double r_78859 = 0.0;
            
            for (int64_t i_78858 = 0; i_78858 < (int64_t) 16; i_78858++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78860 = ((double *) wval_mem_83351.mem)[i_82531 * (int64_t) 16 + i_78858];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78861 = ((double *) mem_83410)[i_82541 * (int64_t) 16 + i_78858];
                
                // futhark/microgpt.fut:160:72-103
                
                double zt_res_78862 = zt_lhs_78860 * zt_rhs_78861;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78863 = r_78859 + zt_res_78862;
                double r_tmp_85163 = zp_res_78863;
                
                r_78859 = r_tmp_85163;
            }
            defunc_0_lifted_lambda_res_78857 = r_78859;
            ((double *) mem_83448)[i_82531] = defunc_0_lifted_lambda_res_78857;
            ((double *) mem_83449)[i_82531] = defunc_0_lifted_lambda_res_78841;
            ((double *) mem_83450)[i_82531] = defunc_0_lifted_lambda_res_78828;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83433, i_82541 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83448, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83434, i_82541 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83449, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83435, i_82541 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83450, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83481_cached_sizze_85488 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83481, &mem_83481_cached_sizze_85488, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83482_cached_sizze_85489 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83482, &mem_83482_cached_sizze_85489, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83483_cached_sizze_85490 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83483, &mem_83483_cached_sizze_85490, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83499_cached_sizze_85491 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83499, &mem_83499_cached_sizze_85491, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83500_cached_sizze_85492 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83500, &mem_83500_cached_sizze_85492, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83501_cached_sizze_85493 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83501, &mem_83501_cached_sizze_85493, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83514_cached_sizze_85494 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83514, &mem_83514_cached_sizze_85494, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83515_cached_sizze_85495 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83515, &mem_83515_cached_sizze_85495, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83516_cached_sizze_85496 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83516, &mem_83516_cached_sizze_85496, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82571 = 0; i_82571 < (int64_t) 4; i_82571++) {
        // futhark/microgpt.fut:161:83-86
        
        int64_t zp_lhs_78703 = mul64((int64_t) 4, i_82571);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82561 = 0; i_82561 < (int64_t) 16; i_82561++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82551 = 0; i_82551 < (int64_t) 4; i_82551++) {
                // futhark/microgpt.fut:161:88-93
                
                int64_t tmp_79021 = add64(zp_lhs_78703, i_82551);
                
                // futhark/microgpt.fut:161:69-95
                
                bool x_79022 = sle64((int64_t) 0, tmp_79021);
                
                // futhark/microgpt.fut:161:69-95
                
                bool y_79023 = slt64(tmp_79021, (int64_t) 16);
                
                // futhark/microgpt.fut:161:69-95
                
                bool bounds_check_79024 = x_79022 && y_79023;
                
                // futhark/microgpt.fut:161:69-95
                
                bool index_certs_79025;
                
                if (!bounds_check_79024) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79021, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:161:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:161:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:161:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:161:15-100\n   #10 futhark/microgpt.fut:349:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79026 = ((double *) mem_83435)[i_82561 * (int64_t) 16 + tmp_79021];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79034 = ((double *) mem_83434)[i_82561 * (int64_t) 16 + tmp_79021];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79045 = ((double *) mem_83433)[i_82561 * (int64_t) 16 + tmp_79021];
                
                ((double *) mem_83514)[i_82551] = lifted_lambda_res_79045;
                ((double *) mem_83515)[i_82551] = lifted_lambda_res_79034;
                ((double *) mem_83516)[i_82551] = lifted_lambda_res_79026;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83499, i_82561 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83500, i_82561 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83515, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83501, i_82561 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83516, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83481, i_82571 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83499, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83482, i_82571 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83500, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83483, i_82571 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83501, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83562_cached_sizze_85497 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83562, &mem_83562_cached_sizze_85497, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83568_cached_sizze_85498 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83568, &mem_83568_cached_sizze_85498, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83573_cached_sizze_85499 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83573, &mem_83573_cached_sizze_85499, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83584_cached_sizze_85500 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83584, &mem_83584_cached_sizze_85500, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83589_cached_sizze_85501 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83589, &mem_83589_cached_sizze_85501, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83600_cached_sizze_85502 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83600, &mem_83600_cached_sizze_85502, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83605_cached_sizze_85503 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83605, &mem_83605_cached_sizze_85503, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83612_cached_sizze_85504 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83612, &mem_83612_cached_sizze_85504, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83619_cached_sizze_85505 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83619, &mem_83619_cached_sizze_85505, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83630_cached_sizze_85506 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83630, &mem_83630_cached_sizze_85506, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83635_cached_sizze_85507 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83635, &mem_83635_cached_sizze_85507, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82619 = 0; i_82619 < (int64_t) 4; i_82619++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82581 = 0; i_82581 < (int64_t) 16; i_82581++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82577 = 0; i_82577 < (int64_t) 16; i_82577++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73285;
                double r_73287 = 0.0;
                
                for (int64_t i_73286 = 0; i_73286 < (int64_t) 4; i_73286++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73288 = ((double *) mem_83483)[i_82619 * (int64_t) 64 + i_82581 * (int64_t) 4 + i_73286];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73289 = ((double *) mem_83482)[i_82619 * (int64_t) 64 + i_82577 * (int64_t) 4 + i_73286];
                    
                    // futhark/microgpt.fut:164:100-139
                    
                    double zt_res_73290 = zt_lhs_73288 * zt_rhs_73289;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73291 = r_73287 + zt_res_73290;
                    double r_tmp_85176 = zp_res_73291;
                    
                    r_73287 = r_tmp_85176;
                }
                defunc_0_lifted_lambda_res_73285 = r_73287;
                ((double *) mem_83573)[i_82577] = defunc_0_lifted_lambda_res_73285;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83568, i_82581 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83573, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82589 = 0; i_82589 < (int64_t) 16; i_82589++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82585 = 0; i_82585 < (int64_t) 16; i_82585++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_73306 = ((double *) mem_83568)[i_82589 * (int64_t) 16 + i_82585];
                
                // futhark/microgpt.fut:165:43-70
                
                double zs_res_73307 = zs_lhs_73306 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_73308 = ((double *) mask_mem_83354.mem)[i_82589 * (int64_t) 16 + i_82585];
                
                // futhark/microgpt.fut:165:57-90
                
                double zp_res_73309 = zs_res_73307 + zp_rhs_73308;
                
                ((double *) mem_83589)[i_82585] = zp_res_73309;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83584, i_82589 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83589, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82607 = 0; i_82607 < (int64_t) 16; i_82607++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_79120;
            double redout_82591 = -INFINITY;
            
            for (int64_t i_82592 = 0; i_82592 < (int64_t) 16; i_82592++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79072 = ((double *) mem_83584)[i_82607 * (int64_t) 16 + i_82592];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_73330 = fmax64(lifted_lambda_res_79072, redout_82591);
                double redout_tmp_85180 = max_res_73330;
                
                redout_82591 = redout_tmp_85180;
            }
            defunc_0_reduce_res_79120 = redout_82591;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_73331 = -defunc_0_reduce_res_79120;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82595 = 0; i_82595 < (int64_t) 16; i_82595++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_73338 = ((double *) mem_83584)[i_82607 * (int64_t) 16 + i_82595];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_73339 = neg_res_73331 + lifted_lambda_res_73338;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_73340 = futrts_exp64(zp_res_73339);
                
                ((double *) mem_83605)[i_82595] = exp_res_73340;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73342;
            double r_73344 = 0.0;
            
            for (int64_t i_73343 = 0; i_73343 < (int64_t) 16; i_73343++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_73345 = ((double *) mem_83605)[i_73343];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73346 = r_73344 + lifted_lambda_res_73345;
                double r_tmp_85182 = zp_res_73346;
                
                r_73344 = r_tmp_85182;
            }
            defunc_0_lifted_lambda_res_73342 = r_73344;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82599 = 0; i_82599 < (int64_t) 16; i_82599++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_73353 = ((double *) mem_83605)[i_82599];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_73354 = zs_lhs_73353 / defunc_0_lifted_lambda_res_73342;
                
                ((double *) mem_83612)[i_82599] = zs_res_73354;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82603 = 0; i_82603 < (int64_t) 16; i_82603++) {
                // futhark/microgpt.fut:167:23-31
                
                double lifted_lambda_res_73362 = ((double *) mem_83612)[i_82603];
                
                ((double *) mem_83619)[i_82603] = lifted_lambda_res_73362;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83600, i_82607 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83619, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82615 = 0; i_82615 < (int64_t) 16; i_82615++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82611 = 0; i_82611 < (int64_t) 4; i_82611++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73377;
                double r_73379 = 0.0;
                
                for (int64_t i_73378 = 0; i_73378 < (int64_t) 16; i_73378++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73380 = ((double *) mem_83600)[i_82615 * (int64_t) 16 + i_73378];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73381 = ((double *) mem_83481)[i_82619 * (int64_t) 64 + i_73378 * (int64_t) 4 + i_82611];
                    
                    // futhark/microgpt.fut:168:61-96
                    
                    double zt_res_73382 = zt_lhs_73380 * zt_rhs_73381;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73383 = r_73379 + zt_res_73382;
                    double r_tmp_85187 = zp_res_73383;
                    
                    r_73379 = r_tmp_85187;
                }
                defunc_0_lifted_lambda_res_73377 = r_73379;
                ((double *) mem_83635)[i_82611] = defunc_0_lifted_lambda_res_73377;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83630, i_82615 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83635, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83562, i_82619 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83630, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83651_cached_sizze_85508 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83651, &mem_83651_cached_sizze_85508, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83656_cached_sizze_85509 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83656, &mem_83656_cached_sizze_85509, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82627 = 0; i_82627 < (int64_t) 16; i_82627++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82623 = 0; i_82623 < (int64_t) 16; i_82623++) {
            // futhark/microgpt.fut:169:61-64
            
            int64_t tmp_73395 = sdiv64(i_82623, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool x_73396 = sle64((int64_t) 0, tmp_73395);
            
            // futhark/microgpt.fut:169:53-66
            
            bool y_73397 = slt64(tmp_73395, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool bounds_check_73398 = x_73396 && y_73397;
            
            // futhark/microgpt.fut:169:53-66
            
            bool index_certs_73399;
            
            if (!bounds_check_73398) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73395, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:349:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:169:77-80
            
            int64_t tmp_73400 = smod64(i_82623, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool x_73401 = sle64((int64_t) 0, tmp_73400);
            
            // futhark/microgpt.fut:169:53-82
            
            bool y_73402 = slt64(tmp_73400, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool bounds_check_73403 = x_73401 && y_73402;
            
            // futhark/microgpt.fut:169:53-82
            
            bool index_certs_73404;
            
            if (!bounds_check_73403) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73400, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:349:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73405 = ((double *) mem_83562)[tmp_73395 * (int64_t) 64 + i_82627 * (int64_t) 4 + tmp_73400];
            
            ((double *) mem_83656)[i_82623] = lifted_lambda_res_73405;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83651, i_82627 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83667_cached_sizze_85510 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83667, &mem_83667_cached_sizze_85510, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83672_cached_sizze_85511 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83672, &mem_83672_cached_sizze_85511, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82635 = 0; i_82635 < (int64_t) 16; i_82635++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82631 = 0; i_82631 < (int64_t) 16; i_82631++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73420;
            double r_73422 = 0.0;
            
            for (int64_t i_73421 = 0; i_73421 < (int64_t) 16; i_73421++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73423 = ((double *) wout_mem_83346.mem)[i_82631 * (int64_t) 16 + i_73421];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73424 = ((double *) mem_83651)[i_82635 * (int64_t) 16 + i_73421];
                
                // futhark/microgpt.fut:170:73-105
                
                double zt_res_73425 = zt_lhs_73423 * zt_rhs_73424;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73426 = r_73422 + zt_res_73425;
                double r_tmp_85192 = zp_res_73426;
                
                r_73422 = r_tmp_85192;
            }
            defunc_0_lifted_lambda_res_73420 = r_73422;
            ((double *) mem_83672)[i_82631] = defunc_0_lifted_lambda_res_73420;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83667, i_82635 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83672, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83683_cached_sizze_85512 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83683, &mem_83683_cached_sizze_85512, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83688_cached_sizze_85513 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83688, &mem_83688_cached_sizze_85513, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82643 = 0; i_82643 < (int64_t) 16; i_82643++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82639 = 0; i_82639 < (int64_t) 16; i_82639++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73441 = ((double *) mem_83667)[i_82643 * (int64_t) 16 + i_82639];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73442 = ((double *) mem_83387)[i_82643 * (int64_t) 16 + i_82639];
            
            // futhark/microgpt.fut:171:42-72
            
            double zp_res_73443 = zp_lhs_73441 + zp_rhs_73442;
            
            ((double *) mem_83688)[i_82639] = zp_res_73443;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83683, i_82643 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83688, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83699_cached_sizze_85514 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83699, &mem_83699_cached_sizze_85514, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83704_cached_sizze_85515 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83704, &mem_83704_cached_sizze_85515, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83711_cached_sizze_85516 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83711, &mem_83711_cached_sizze_85516, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82655 = 0; i_82655 < (int64_t) 16; i_82655++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82647 = 0; i_82647 < (int64_t) 16; i_82647++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73458 = ((double *) mem_83683)[i_82655 * (int64_t) 16 + i_82647];
            
            // futhark/microgpt.fut:172:65-96
            
            double zt_res_73459 = zt_lhs_73458 * zt_lhs_73458;
            
            ((double *) mem_83704)[i_82647] = zt_res_73459;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73461;
        double r_73463 = 0.0;
        
        for (int64_t i_73462 = 0; i_73462 < (int64_t) 16; i_73462++) {
            // futhark/microgpt.fut:173:35-43
            
            double lifted_lambda_res_73464 = ((double *) mem_83704)[i_73462];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73465 = r_73463 + lifted_lambda_res_73464;
            double r_tmp_85197 = zp_res_73465;
            
            r_73463 = r_tmp_85197;
        }
        defunc_0_lifted_lambda_res_73461 = r_73463;
        // futhark/microgpt.fut:173:17-60
        
        double zs_res_73466 = defunc_0_lifted_lambda_res_73461 / 16.0;
        
        // futhark/microgpt.fut:174:24-55
        
        double zp_res_73467 = 1.0e-5 + zs_res_73466;
        
        // futhark/microgpt.fut:174:16-55
        
        double sqrt_res_73468 = futrts_sqrt64(zp_res_73467);
        
        // futhark/microgpt.fut:175:43-54
        
        double zs_res_73469 = 1.0 / sqrt_res_73468;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82651 = 0; i_82651 < (int64_t) 16; i_82651++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73476 = ((double *) mem_83683)[i_82655 * (int64_t) 16 + i_82651];
            
            // futhark/microgpt.fut:175:24-54
            
            double zt_res_73477 = zs_res_73469 * zt_lhs_73476;
            
            ((double *) mem_83711)[i_82651] = zt_res_73477;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83699, i_82655 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83711, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83722_cached_sizze_85517 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83722, &mem_83722_cached_sizze_85517, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83727_cached_sizze_85518 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83727, &mem_83727_cached_sizze_85518, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82663 = 0; i_82663 < (int64_t) 16; i_82663++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82659 = 0; i_82659 < (int64_t) 64; i_82659++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73493;
            double r_73495 = 0.0;
            
            for (int64_t i_73494 = 0; i_73494 < (int64_t) 16; i_73494++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73496 = ((double *) wup_mem_83350.mem)[i_82659 * (int64_t) 16 + i_73494];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73497 = ((double *) mem_83699)[i_82663 * (int64_t) 16 + i_73494];
                
                // futhark/microgpt.fut:176:73-104
                
                double zt_res_73498 = zt_lhs_73496 * zt_rhs_73497;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73499 = r_73495 + zt_res_73498;
                double r_tmp_85201 = zp_res_73499;
                
                r_73495 = r_tmp_85201;
            }
            defunc_0_lifted_lambda_res_73493 = r_73495;
            ((double *) mem_83727)[i_82659] = defunc_0_lifted_lambda_res_73493;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83722, i_82663 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83727, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83738_cached_sizze_85519 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83738, &mem_83738_cached_sizze_85519, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83743_cached_sizze_85520 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83743, &mem_83743_cached_sizze_85520, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82671 = 0; i_82671 < (int64_t) 16; i_82671++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82667 = 0; i_82667 < (int64_t) 64; i_82667++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_73514 = ((double *) mem_83722)[i_82671 * (int64_t) 64 + i_82667];
            
            // futhark/microgpt.fut:177:42-66
            
            double max_res_73515 = fmax64(0.0, max_arg0_73514);
            
            ((double *) mem_83743)[i_82667] = max_res_73515;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83738, i_82671 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83743, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83754_cached_sizze_85521 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83754, &mem_83754_cached_sizze_85521, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83759_cached_sizze_85522 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83759, &mem_83759_cached_sizze_85522, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82679 = 0; i_82679 < (int64_t) 16; i_82679++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82675 = 0; i_82675 < (int64_t) 16; i_82675++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73530;
            double r_73532 = 0.0;
            
            for (int64_t i_73531 = 0; i_73531 < (int64_t) 64; i_73531++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73533 = ((double *) wdown_mem_83344.mem)[i_82675 * (int64_t) 64 + i_73531];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73534 = ((double *) mem_83738)[i_82679 * (int64_t) 64 + i_73531];
                
                // futhark/microgpt.fut:178:73-106
                
                double zt_res_73535 = zt_lhs_73533 * zt_rhs_73534;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73536 = r_73532 + zt_res_73535;
                double r_tmp_85206 = zp_res_73536;
                
                r_73532 = r_tmp_85206;
            }
            defunc_0_lifted_lambda_res_73530 = r_73532;
            ((double *) mem_83759)[i_82675] = defunc_0_lifted_lambda_res_73530;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83754, i_82679 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83770_cached_sizze_85523 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83770, &mem_83770_cached_sizze_85523, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83775_cached_sizze_85524 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83775, &mem_83775_cached_sizze_85524, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82687 = 0; i_82687 < (int64_t) 16; i_82687++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82683 = 0; i_82683 < (int64_t) 16; i_82683++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73551 = ((double *) mem_83754)[i_82687 * (int64_t) 16 + i_82683];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73552 = ((double *) mem_83683)[i_82687 * (int64_t) 16 + i_82683];
            
            // futhark/microgpt.fut:179:42-73
            
            double zp_res_73553 = zp_lhs_73551 + zp_rhs_73552;
            
            ((double *) mem_83775)[i_82683] = zp_res_73553;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83770, i_82687 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83775, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83786, (int64_t) 3456, "mem_83786")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83791_cached_sizze_85525 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83791, &mem_83791_cached_sizze_85525, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82695 = 0; i_82695 < (int64_t) 16; i_82695++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82691 = 0; i_82691 < (int64_t) 27; i_82691++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73569;
            double r_73571 = 0.0;
            
            for (int64_t i_73570 = 0; i_73570 < (int64_t) 16; i_73570++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73572 = ((double *) wvoc_mem_83352.mem)[i_82691 * (int64_t) 16 + i_73570];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73573 = ((double *) mem_83770)[i_82695 * (int64_t) 16 + i_73570];
                
                // futhark/microgpt.fut:180:62-94
                
                double zt_res_73574 = zt_lhs_73572 * zt_rhs_73573;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73575 = r_73571 + zt_res_73574;
                double r_tmp_85211 = zp_res_73575;
                
                r_73571 = r_tmp_85211;
            }
            defunc_0_lifted_lambda_res_73569 = r_73571;
            ((double *) mem_83791)[i_82691] = defunc_0_lifted_lambda_res_73569;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83786.mem, i_82695 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83791, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_85142, &mem_83786, "mem_83786") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85471, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83355);
        free(mem_83360);
        free(mem_83371);
        free(mem_83376);
        free(mem_83387);
        free(mem_83392);
        free(mem_83399);
        free(mem_83410);
        free(mem_83415);
        free(mem_83422);
        free(mem_83433);
        free(mem_83434);
        free(mem_83435);
        free(mem_83448);
        free(mem_83449);
        free(mem_83450);
        free(mem_83481);
        free(mem_83482);
        free(mem_83483);
        free(mem_83499);
        free(mem_83500);
        free(mem_83501);
        free(mem_83514);
        free(mem_83515);
        free(mem_83516);
        free(mem_83562);
        free(mem_83568);
        free(mem_83573);
        free(mem_83584);
        free(mem_83589);
        free(mem_83600);
        free(mem_83605);
        free(mem_83612);
        free(mem_83619);
        free(mem_83630);
        free(mem_83635);
        free(mem_83651);
        free(mem_83656);
        free(mem_83667);
        free(mem_83672);
        free(mem_83683);
        free(mem_83688);
        free(mem_83699);
        free(mem_83704);
        free(mem_83711);
        free(mem_83722);
        free(mem_83727);
        free(mem_83738);
        free(mem_83743);
        free(mem_83754);
        free(mem_83759);
        free(mem_83770);
        free(mem_83775);
        free(mem_83791);
        if (memblock_unref(ctx, &mem_83786, "mem_83786") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock wte_mem_83344, struct memblock wpe_mem_83345, struct memblock wqry_mem_83346, struct memblock wkey_mem_83347, struct memblock wval_mem_83348, struct memblock wout_mem_83349, struct memblock wup_mem_83350, struct memblock wdown_mem_83351, struct memblock wvoc_mem_83352)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    if (memblock_set(ctx, &mem_out_85142, &wdown_mem_83351, "wdown_mem_83351") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &wkey_mem_83347, "wkey_mem_83347") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &wout_mem_83349, "wout_mem_83349") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85145, &wpe_mem_83345, "wpe_mem_83345") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85146, &wqry_mem_83346, "wqry_mem_83346") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85147, &wte_mem_83344, "wte_mem_83344") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85148, &wup_mem_83350, "wup_mem_83350") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85149, &wval_mem_83348, "wval_mem_83348") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85150, &wvoc_mem_83352, "wvoc_mem_83352") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85526, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85527, &mem_out_85143, "mem_out_85143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85528, &mem_out_85144, "mem_out_85144") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85529, &mem_out_85145, "mem_out_85145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85530, &mem_out_85146, "mem_out_85146") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85531, &mem_out_85147, "mem_out_85147") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85532, &mem_out_85148, "mem_out_85148") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85533, &mem_out_85149, "mem_out_85149") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85534, &mem_out_85150, "mem_out_85150") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85150, "mem_out_85150") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85149, "mem_out_85149") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85148, "mem_out_85148") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85147, "mem_out_85147") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85146, "mem_out_85146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85145, "mem_out_85145") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85144, "mem_out_85144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85143, "mem_out_85143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock *mem_out_p_85540, struct memblock *mem_out_p_85541, struct memblock *mem_out_p_85542, struct memblock *mem_out_p_85543, struct memblock *mem_out_p_85544, struct memblock *mem_out_p_85545, struct memblock *mem_out_p_85546, struct memblock *mem_out_p_85547, struct memblock *mem_out_p_85548, struct memblock *mem_out_p_85549, struct memblock *mem_out_p_85550, struct memblock *mem_out_p_85551, struct memblock *mem_out_p_85552, struct memblock *mem_out_p_85553, struct memblock *mem_out_p_85554, struct memblock *mem_out_p_85555, struct memblock *mem_out_p_85556, struct memblock *mem_out_p_85557, struct memblock *mem_out_p_85558, struct memblock *mem_out_p_85559, struct memblock *mem_out_p_85560, struct memblock *mem_out_p_85561, struct memblock wdown_mem_83344, struct memblock wkey_mem_83345, struct memblock wout_mem_83346, struct memblock wpe_mem_83347, struct memblock wqry_mem_83348, struct memblock wte_mem_83349, struct memblock wup_mem_83350, struct memblock wval_mem_83351, struct memblock wvoc_mem_83352, struct memblock wdown_mem_83353, struct memblock wkey_mem_83354, struct memblock wout_mem_83355, struct memblock wpe_mem_83356, struct memblock wqry_mem_83357, struct memblock wte_mem_83358, struct memblock wup_mem_83359, struct memblock wval_mem_83360, struct memblock wvoc_mem_83361, struct memblock wdown_mem_83362, struct memblock wkey_mem_83363, struct memblock wout_mem_83364, struct memblock wpe_mem_83365, struct memblock wqry_mem_83366, struct memblock wte_mem_83367, struct memblock wup_mem_83368, struct memblock wval_mem_83369, struct memblock wvoc_mem_83370, struct memblock masks_mem_83371, struct memblock seqs_mem_83372, int64_t num_batches_62011, int64_t batchsizze_62012)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83481_cached_sizze_85562 = 0;
    unsigned char *mem_83481 = NULL;
    int64_t mem_83482_cached_sizze_85563 = 0;
    unsigned char *mem_83482 = NULL;
    int64_t mem_83491_cached_sizze_85564 = 0;
    unsigned char *mem_83491 = NULL;
    int64_t mem_83498_cached_sizze_85565 = 0;
    unsigned char *mem_83498 = NULL;
    int64_t mem_83513_cached_sizze_85566 = 0;
    unsigned char *mem_83513 = NULL;
    int64_t mem_83514_cached_sizze_85567 = 0;
    unsigned char *mem_83514 = NULL;
    int64_t mem_83523_cached_sizze_85568 = 0;
    unsigned char *mem_83523 = NULL;
    int64_t mem_83530_cached_sizze_85569 = 0;
    unsigned char *mem_83530 = NULL;
    int64_t mem_83545_cached_sizze_85570 = 0;
    unsigned char *mem_83545 = NULL;
    int64_t mem_83546_cached_sizze_85571 = 0;
    unsigned char *mem_83546 = NULL;
    int64_t mem_83555_cached_sizze_85572 = 0;
    unsigned char *mem_83555 = NULL;
    int64_t mem_83556_cached_sizze_85573 = 0;
    unsigned char *mem_83556 = NULL;
    int64_t mem_83577_cached_sizze_85574 = 0;
    unsigned char *mem_83577 = NULL;
    int64_t mem_83578_cached_sizze_85575 = 0;
    unsigned char *mem_83578 = NULL;
    int64_t mem_83579_cached_sizze_85576 = 0;
    unsigned char *mem_83579 = NULL;
    int64_t mem_83591_cached_sizze_85577 = 0;
    unsigned char *mem_83591 = NULL;
    int64_t mem_83592_cached_sizze_85578 = 0;
    unsigned char *mem_83592 = NULL;
    int64_t mem_83616_cached_sizze_85579 = 0;
    unsigned char *mem_83616 = NULL;
    int64_t mem_83617_cached_sizze_85580 = 0;
    unsigned char *mem_83617 = NULL;
    int64_t mem_83618_cached_sizze_85581 = 0;
    unsigned char *mem_83618 = NULL;
    int64_t mem_83619_cached_sizze_85582 = 0;
    unsigned char *mem_83619 = NULL;
    int64_t mem_83620_cached_sizze_85583 = 0;
    unsigned char *mem_83620 = NULL;
    int64_t mem_83639_cached_sizze_85584 = 0;
    unsigned char *mem_83639 = NULL;
    int64_t mem_83640_cached_sizze_85585 = 0;
    unsigned char *mem_83640 = NULL;
    int64_t mem_83641_cached_sizze_85586 = 0;
    unsigned char *mem_83641 = NULL;
    int64_t mem_83678_cached_sizze_85587 = 0;
    unsigned char *mem_83678 = NULL;
    int64_t mem_83679_cached_sizze_85588 = 0;
    unsigned char *mem_83679 = NULL;
    int64_t mem_83680_cached_sizze_85589 = 0;
    unsigned char *mem_83680 = NULL;
    int64_t mem_83696_cached_sizze_85590 = 0;
    unsigned char *mem_83696 = NULL;
    int64_t mem_83697_cached_sizze_85591 = 0;
    unsigned char *mem_83697 = NULL;
    int64_t mem_83698_cached_sizze_85592 = 0;
    unsigned char *mem_83698 = NULL;
    int64_t mem_83711_cached_sizze_85593 = 0;
    unsigned char *mem_83711 = NULL;
    int64_t mem_83712_cached_sizze_85594 = 0;
    unsigned char *mem_83712 = NULL;
    int64_t mem_83713_cached_sizze_85595 = 0;
    unsigned char *mem_83713 = NULL;
    int64_t mem_83759_cached_sizze_85596 = 0;
    unsigned char *mem_83759 = NULL;
    int64_t mem_83760_cached_sizze_85597 = 0;
    unsigned char *mem_83760 = NULL;
    int64_t mem_83771_cached_sizze_85598 = 0;
    unsigned char *mem_83771 = NULL;
    int64_t mem_83772_cached_sizze_85599 = 0;
    unsigned char *mem_83772 = NULL;
    int64_t mem_83781_cached_sizze_85600 = 0;
    unsigned char *mem_83781 = NULL;
    int64_t mem_83782_cached_sizze_85601 = 0;
    unsigned char *mem_83782 = NULL;
    int64_t mem_83803_cached_sizze_85602 = 0;
    unsigned char *mem_83803 = NULL;
    int64_t mem_83808_cached_sizze_85603 = 0;
    unsigned char *mem_83808 = NULL;
    int64_t mem_83819_cached_sizze_85604 = 0;
    unsigned char *mem_83819 = NULL;
    int64_t mem_83824_cached_sizze_85605 = 0;
    unsigned char *mem_83824 = NULL;
    int64_t mem_83831_cached_sizze_85606 = 0;
    unsigned char *mem_83831 = NULL;
    int64_t mem_83838_cached_sizze_85607 = 0;
    unsigned char *mem_83838 = NULL;
    int64_t mem_83849_cached_sizze_85608 = 0;
    unsigned char *mem_83849 = NULL;
    int64_t mem_83854_cached_sizze_85609 = 0;
    unsigned char *mem_83854 = NULL;
    int64_t mem_83875_cached_sizze_85610 = 0;
    unsigned char *mem_83875 = NULL;
    int64_t mem_83876_cached_sizze_85611 = 0;
    unsigned char *mem_83876 = NULL;
    int64_t mem_83884_cached_sizze_85612 = 0;
    unsigned char *mem_83884 = NULL;
    int64_t mem_83898_cached_sizze_85613 = 0;
    unsigned char *mem_83898 = NULL;
    int64_t mem_83903_cached_sizze_85614 = 0;
    unsigned char *mem_83903 = NULL;
    int64_t mem_83914_cached_sizze_85615 = 0;
    unsigned char *mem_83914 = NULL;
    int64_t mem_83919_cached_sizze_85616 = 0;
    unsigned char *mem_83919 = NULL;
    int64_t mem_83930_cached_sizze_85617 = 0;
    unsigned char *mem_83930 = NULL;
    int64_t mem_83931_cached_sizze_85618 = 0;
    unsigned char *mem_83931 = NULL;
    int64_t mem_83940_cached_sizze_85619 = 0;
    unsigned char *mem_83940 = NULL;
    int64_t mem_83941_cached_sizze_85620 = 0;
    unsigned char *mem_83941 = NULL;
    int64_t mem_83962_cached_sizze_85621 = 0;
    unsigned char *mem_83962 = NULL;
    int64_t mem_83963_cached_sizze_85622 = 0;
    unsigned char *mem_83963 = NULL;
    int64_t mem_83971_cached_sizze_85623 = 0;
    unsigned char *mem_83971 = NULL;
    int64_t mem_83985_cached_sizze_85624 = 0;
    unsigned char *mem_83985 = NULL;
    int64_t mem_83986_cached_sizze_85625 = 0;
    unsigned char *mem_83986 = NULL;
    int64_t mem_83994_cached_sizze_85626 = 0;
    unsigned char *mem_83994 = NULL;
    int64_t mem_84008_cached_sizze_85627 = 0;
    unsigned char *mem_84008 = NULL;
    int64_t mem_84013_cached_sizze_85628 = 0;
    unsigned char *mem_84013 = NULL;
    int64_t mem_84024_cached_sizze_85629 = 0;
    unsigned char *mem_84024 = NULL;
    int64_t mem_84029_cached_sizze_85630 = 0;
    unsigned char *mem_84029 = NULL;
    int64_t mem_84040_cached_sizze_85631 = 0;
    unsigned char *mem_84040 = NULL;
    int64_t mem_84045_cached_sizze_85632 = 0;
    unsigned char *mem_84045 = NULL;
    int64_t mem_84056_cached_sizze_85633 = 0;
    unsigned char *mem_84056 = NULL;
    int64_t mem_84057_cached_sizze_85634 = 0;
    unsigned char *mem_84057 = NULL;
    int64_t mem_84066_cached_sizze_85635 = 0;
    unsigned char *mem_84066 = NULL;
    int64_t mem_84067_cached_sizze_85636 = 0;
    unsigned char *mem_84067 = NULL;
    int64_t mem_84080_cached_sizze_85637 = 0;
    unsigned char *mem_84080 = NULL;
    int64_t mem_84081_cached_sizze_85638 = 0;
    unsigned char *mem_84081 = NULL;
    int64_t mem_84094_cached_sizze_85639 = 0;
    unsigned char *mem_84094 = NULL;
    int64_t mem_84095_cached_sizze_85640 = 0;
    unsigned char *mem_84095 = NULL;
    int64_t mem_84116_cached_sizze_85641 = 0;
    unsigned char *mem_84116 = NULL;
    int64_t mem_84123_cached_sizze_85642 = 0;
    unsigned char *mem_84123 = NULL;
    int64_t mem_84128_cached_sizze_85643 = 0;
    unsigned char *mem_84128 = NULL;
    int64_t mem_84139_cached_sizze_85644 = 0;
    unsigned char *mem_84139 = NULL;
    int64_t mem_84144_cached_sizze_85645 = 0;
    unsigned char *mem_84144 = NULL;
    int64_t mem_84155_cached_sizze_85646 = 0;
    unsigned char *mem_84155 = NULL;
    int64_t mem_84156_cached_sizze_85647 = 0;
    unsigned char *mem_84156 = NULL;
    int64_t mem_84165_cached_sizze_85648 = 0;
    unsigned char *mem_84165 = NULL;
    int64_t mem_84166_cached_sizze_85649 = 0;
    unsigned char *mem_84166 = NULL;
    int64_t mem_84187_cached_sizze_85650 = 0;
    unsigned char *mem_84187 = NULL;
    int64_t mem_84192_cached_sizze_85651 = 0;
    unsigned char *mem_84192 = NULL;
    int64_t mem_84203_cached_sizze_85652 = 0;
    unsigned char *mem_84203 = NULL;
    int64_t mem_84208_cached_sizze_85653 = 0;
    unsigned char *mem_84208 = NULL;
    int64_t mem_84219_cached_sizze_85654 = 0;
    unsigned char *mem_84219 = NULL;
    int64_t mem_84226_cached_sizze_85655 = 0;
    unsigned char *mem_84226 = NULL;
    int64_t mem_84233_cached_sizze_85656 = 0;
    unsigned char *mem_84233 = NULL;
    int64_t mem_84243_cached_sizze_85657 = 0;
    unsigned char *mem_84243 = NULL;
    int64_t mem_84248_cached_sizze_85658 = 0;
    unsigned char *mem_84248 = NULL;
    int64_t mem_84259_cached_sizze_85659 = 0;
    unsigned char *mem_84259 = NULL;
    int64_t mem_84260_cached_sizze_85660 = 0;
    unsigned char *mem_84260 = NULL;
    int64_t mem_84269_cached_sizze_85661 = 0;
    unsigned char *mem_84269 = NULL;
    int64_t mem_84270_cached_sizze_85662 = 0;
    unsigned char *mem_84270 = NULL;
    int64_t mem_84291_cached_sizze_85663 = 0;
    unsigned char *mem_84291 = NULL;
    int64_t mem_84292_cached_sizze_85664 = 0;
    unsigned char *mem_84292 = NULL;
    int64_t mem_84303_cached_sizze_85665 = 0;
    unsigned char *mem_84303 = NULL;
    int64_t mem_84304_cached_sizze_85666 = 0;
    unsigned char *mem_84304 = NULL;
    int64_t mem_84313_cached_sizze_85667 = 0;
    unsigned char *mem_84313 = NULL;
    int64_t mem_84320_cached_sizze_85668 = 0;
    unsigned char *mem_84320 = NULL;
    int64_t mem_84345_cached_sizze_85669 = 0;
    unsigned char *mem_84345 = NULL;
    int64_t mem_84346_cached_sizze_85670 = 0;
    unsigned char *mem_84346 = NULL;
    int64_t mem_84357_cached_sizze_85671 = 0;
    unsigned char *mem_84357 = NULL;
    int64_t mem_84358_cached_sizze_85672 = 0;
    unsigned char *mem_84358 = NULL;
    int64_t mem_84367_cached_sizze_85673 = 0;
    unsigned char *mem_84367 = NULL;
    int64_t mem_84374_cached_sizze_85674 = 0;
    unsigned char *mem_84374 = NULL;
    int64_t mem_84381_cached_sizze_85675 = 0;
    unsigned char *mem_84381 = NULL;
    int64_t mem_84388_cached_sizze_85676 = 0;
    unsigned char *mem_84388 = NULL;
    int64_t mem_84413_cached_sizze_85677 = 0;
    unsigned char *mem_84413 = NULL;
    int64_t mem_84414_cached_sizze_85678 = 0;
    unsigned char *mem_84414 = NULL;
    int64_t mem_84425_cached_sizze_85679 = 0;
    unsigned char *mem_84425 = NULL;
    int64_t mem_84426_cached_sizze_85680 = 0;
    unsigned char *mem_84426 = NULL;
    int64_t mem_84435_cached_sizze_85681 = 0;
    unsigned char *mem_84435 = NULL;
    int64_t mem_84442_cached_sizze_85682 = 0;
    unsigned char *mem_84442 = NULL;
    int64_t mem_84467_cached_sizze_85683 = 0;
    unsigned char *mem_84467 = NULL;
    int64_t mem_84472_cached_sizze_85684 = 0;
    unsigned char *mem_84472 = NULL;
    int64_t mem_84483_cached_sizze_85685 = 0;
    unsigned char *mem_84483 = NULL;
    int64_t mem_84489_cached_sizze_85686 = 0;
    unsigned char *mem_84489 = NULL;
    int64_t mem_84494_cached_sizze_85687 = 0;
    unsigned char *mem_84494 = NULL;
    int64_t mem_84510_cached_sizze_85688 = 0;
    unsigned char *mem_84510 = NULL;
    int64_t mem_84516_cached_sizze_85689 = 0;
    unsigned char *mem_84516 = NULL;
    int64_t mem_84521_cached_sizze_85690 = 0;
    unsigned char *mem_84521 = NULL;
    int64_t mem_84537_cached_sizze_85691 = 0;
    unsigned char *mem_84537 = NULL;
    int64_t mem_84538_cached_sizze_85692 = 0;
    unsigned char *mem_84538 = NULL;
    int64_t mem_84549_cached_sizze_85693 = 0;
    unsigned char *mem_84549 = NULL;
    int64_t mem_84550_cached_sizze_85694 = 0;
    unsigned char *mem_84550 = NULL;
    int64_t mem_84559_cached_sizze_85695 = 0;
    unsigned char *mem_84559 = NULL;
    int64_t mem_84560_cached_sizze_85696 = 0;
    unsigned char *mem_84560 = NULL;
    int64_t mem_84591_cached_sizze_85697 = 0;
    unsigned char *mem_84591 = NULL;
    int64_t mem_84592_cached_sizze_85698 = 0;
    unsigned char *mem_84592 = NULL;
    int64_t mem_84593_cached_sizze_85699 = 0;
    unsigned char *mem_84593 = NULL;
    int64_t mem_84606_cached_sizze_85700 = 0;
    unsigned char *mem_84606 = NULL;
    int64_t mem_84607_cached_sizze_85701 = 0;
    unsigned char *mem_84607 = NULL;
    int64_t mem_84608_cached_sizze_85702 = 0;
    unsigned char *mem_84608 = NULL;
    int64_t mem_84639_cached_sizze_85703 = 0;
    unsigned char *mem_84639 = NULL;
    int64_t mem_84640_cached_sizze_85704 = 0;
    unsigned char *mem_84640 = NULL;
    int64_t mem_84641_cached_sizze_85705 = 0;
    unsigned char *mem_84641 = NULL;
    int64_t mem_84642_cached_sizze_85706 = 0;
    unsigned char *mem_84642 = NULL;
    int64_t mem_84659_cached_sizze_85707 = 0;
    unsigned char *mem_84659 = NULL;
    int64_t mem_84660_cached_sizze_85708 = 0;
    unsigned char *mem_84660 = NULL;
    int64_t mem_84661_cached_sizze_85709 = 0;
    unsigned char *mem_84661 = NULL;
    int64_t mem_84662_cached_sizze_85710 = 0;
    unsigned char *mem_84662 = NULL;
    int64_t mem_84703_cached_sizze_85711 = 0;
    unsigned char *mem_84703 = NULL;
    int64_t mem_84710_cached_sizze_85712 = 0;
    unsigned char *mem_84710 = NULL;
    int64_t mem_84717_cached_sizze_85713 = 0;
    unsigned char *mem_84717 = NULL;
    int64_t mem_84727_cached_sizze_85714 = 0;
    unsigned char *mem_84727 = NULL;
    int64_t mem_84732_cached_sizze_85715 = 0;
    unsigned char *mem_84732 = NULL;
    int64_t mem_84743_cached_sizze_85716 = 0;
    unsigned char *mem_84743 = NULL;
    int64_t mem_84750_cached_sizze_85717 = 0;
    unsigned char *mem_84750 = NULL;
    int64_t mem_84757_cached_sizze_85718 = 0;
    unsigned char *mem_84757 = NULL;
    int64_t mem_84767_cached_sizze_85719 = 0;
    unsigned char *mem_84767 = NULL;
    int64_t mem_84772_cached_sizze_85720 = 0;
    unsigned char *mem_84772 = NULL;
    int64_t mem_84783_cached_sizze_85721 = 0;
    unsigned char *mem_84783 = NULL;
    int64_t mem_84784_cached_sizze_85722 = 0;
    unsigned char *mem_84784 = NULL;
    int64_t mem_84793_cached_sizze_85723 = 0;
    unsigned char *mem_84793 = NULL;
    int64_t mem_84794_cached_sizze_85724 = 0;
    unsigned char *mem_84794 = NULL;
    int64_t mem_84815_cached_sizze_85725 = 0;
    unsigned char *mem_84815 = NULL;
    int64_t mem_84820_cached_sizze_85726 = 0;
    unsigned char *mem_84820 = NULL;
    int64_t mem_84831_cached_sizze_85727 = 0;
    unsigned char *mem_84831 = NULL;
    int64_t mem_84832_cached_sizze_85728 = 0;
    unsigned char *mem_84832 = NULL;
    int64_t mem_84841_cached_sizze_85729 = 0;
    unsigned char *mem_84841 = NULL;
    int64_t mem_84842_cached_sizze_85730 = 0;
    unsigned char *mem_84842 = NULL;
    struct memblock mem_param_tmp_85195;
    
    mem_param_tmp_85195.references = NULL;
    
    struct memblock mem_param_tmp_85194;
    
    mem_param_tmp_85194.references = NULL;
    
    struct memblock mem_param_tmp_85193;
    
    mem_param_tmp_85193.references = NULL;
    
    struct memblock mem_param_tmp_85192;
    
    mem_param_tmp_85192.references = NULL;
    
    struct memblock mem_param_tmp_85191;
    
    mem_param_tmp_85191.references = NULL;
    
    struct memblock mem_param_tmp_85190;
    
    mem_param_tmp_85190.references = NULL;
    
    struct memblock mem_param_tmp_85189;
    
    mem_param_tmp_85189.references = NULL;
    
    struct memblock mem_param_tmp_85188;
    
    mem_param_tmp_85188.references = NULL;
    
    struct memblock mem_param_tmp_85187;
    
    mem_param_tmp_85187.references = NULL;
    
    struct memblock mem_param_tmp_85186;
    
    mem_param_tmp_85186.references = NULL;
    
    struct memblock mem_param_tmp_85185;
    
    mem_param_tmp_85185.references = NULL;
    
    struct memblock mem_param_tmp_85184;
    
    mem_param_tmp_85184.references = NULL;
    
    struct memblock mem_param_tmp_85183;
    
    mem_param_tmp_85183.references = NULL;
    
    struct memblock mem_param_tmp_85182;
    
    mem_param_tmp_85182.references = NULL;
    
    struct memblock mem_param_tmp_85181;
    
    mem_param_tmp_85181.references = NULL;
    
    struct memblock mem_param_tmp_85180;
    
    mem_param_tmp_85180.references = NULL;
    
    struct memblock mem_param_tmp_85179;
    
    mem_param_tmp_85179.references = NULL;
    
    struct memblock mem_param_tmp_85178;
    
    mem_param_tmp_85178.references = NULL;
    
    struct memblock mem_param_tmp_85177;
    
    mem_param_tmp_85177.references = NULL;
    
    struct memblock mem_param_tmp_85176;
    
    mem_param_tmp_85176.references = NULL;
    
    struct memblock mem_param_tmp_85175;
    
    mem_param_tmp_85175.references = NULL;
    
    struct memblock mem_param_tmp_85174;
    
    mem_param_tmp_85174.references = NULL;
    
    struct memblock mem_param_tmp_85173;
    
    mem_param_tmp_85173.references = NULL;
    
    struct memblock mem_param_tmp_85172;
    
    mem_param_tmp_85172.references = NULL;
    
    struct memblock mem_param_tmp_85171;
    
    mem_param_tmp_85171.references = NULL;
    
    struct memblock mem_param_tmp_85170;
    
    mem_param_tmp_85170.references = NULL;
    
    struct memblock mem_param_tmp_85169;
    
    mem_param_tmp_85169.references = NULL;
    
    struct memblock ext_mem_84959;
    
    ext_mem_84959.references = NULL;
    
    struct memblock ext_mem_84960;
    
    ext_mem_84960.references = NULL;
    
    struct memblock ext_mem_84961;
    
    ext_mem_84961.references = NULL;
    
    struct memblock mem_84957;
    
    mem_84957.references = NULL;
    
    struct memblock mem_84955;
    
    mem_84955.references = NULL;
    
    struct memblock mem_84953;
    
    mem_84953.references = NULL;
    
    struct memblock mem_84951;
    
    mem_84951.references = NULL;
    
    struct memblock ext_mem_84948;
    
    ext_mem_84948.references = NULL;
    
    struct memblock ext_mem_84949;
    
    ext_mem_84949.references = NULL;
    
    struct memblock ext_mem_84950;
    
    ext_mem_84950.references = NULL;
    
    struct memblock mem_84946;
    
    mem_84946.references = NULL;
    
    struct memblock mem_84944;
    
    mem_84944.references = NULL;
    
    struct memblock mem_84942;
    
    mem_84942.references = NULL;
    
    struct memblock mem_84940;
    
    mem_84940.references = NULL;
    
    struct memblock ext_mem_84937;
    
    ext_mem_84937.references = NULL;
    
    struct memblock ext_mem_84938;
    
    ext_mem_84938.references = NULL;
    
    struct memblock ext_mem_84939;
    
    ext_mem_84939.references = NULL;
    
    struct memblock mem_84935;
    
    mem_84935.references = NULL;
    
    struct memblock mem_84933;
    
    mem_84933.references = NULL;
    
    struct memblock mem_84931;
    
    mem_84931.references = NULL;
    
    struct memblock mem_84929;
    
    mem_84929.references = NULL;
    
    struct memblock ext_mem_84926;
    
    ext_mem_84926.references = NULL;
    
    struct memblock ext_mem_84927;
    
    ext_mem_84927.references = NULL;
    
    struct memblock ext_mem_84928;
    
    ext_mem_84928.references = NULL;
    
    struct memblock mem_84924;
    
    mem_84924.references = NULL;
    
    struct memblock mem_84922;
    
    mem_84922.references = NULL;
    
    struct memblock mem_84920;
    
    mem_84920.references = NULL;
    
    struct memblock mem_84918;
    
    mem_84918.references = NULL;
    
    struct memblock ext_mem_84915;
    
    ext_mem_84915.references = NULL;
    
    struct memblock ext_mem_84916;
    
    ext_mem_84916.references = NULL;
    
    struct memblock ext_mem_84917;
    
    ext_mem_84917.references = NULL;
    
    struct memblock mem_84913;
    
    mem_84913.references = NULL;
    
    struct memblock mem_84911;
    
    mem_84911.references = NULL;
    
    struct memblock mem_84909;
    
    mem_84909.references = NULL;
    
    struct memblock mem_84907;
    
    mem_84907.references = NULL;
    
    struct memblock ext_mem_84904;
    
    ext_mem_84904.references = NULL;
    
    struct memblock ext_mem_84905;
    
    ext_mem_84905.references = NULL;
    
    struct memblock ext_mem_84906;
    
    ext_mem_84906.references = NULL;
    
    struct memblock mem_84902;
    
    mem_84902.references = NULL;
    
    struct memblock mem_84900;
    
    mem_84900.references = NULL;
    
    struct memblock mem_84898;
    
    mem_84898.references = NULL;
    
    struct memblock mem_84896;
    
    mem_84896.references = NULL;
    
    struct memblock ext_mem_84893;
    
    ext_mem_84893.references = NULL;
    
    struct memblock ext_mem_84894;
    
    ext_mem_84894.references = NULL;
    
    struct memblock ext_mem_84895;
    
    ext_mem_84895.references = NULL;
    
    struct memblock mem_84891;
    
    mem_84891.references = NULL;
    
    struct memblock mem_84889;
    
    mem_84889.references = NULL;
    
    struct memblock mem_84887;
    
    mem_84887.references = NULL;
    
    struct memblock mem_84885;
    
    mem_84885.references = NULL;
    
    struct memblock ext_mem_84882;
    
    ext_mem_84882.references = NULL;
    
    struct memblock ext_mem_84883;
    
    ext_mem_84883.references = NULL;
    
    struct memblock ext_mem_84884;
    
    ext_mem_84884.references = NULL;
    
    struct memblock mem_84880;
    
    mem_84880.references = NULL;
    
    struct memblock mem_84878;
    
    mem_84878.references = NULL;
    
    struct memblock mem_84876;
    
    mem_84876.references = NULL;
    
    struct memblock mem_84874;
    
    mem_84874.references = NULL;
    
    struct memblock ext_mem_84871;
    
    ext_mem_84871.references = NULL;
    
    struct memblock ext_mem_84872;
    
    ext_mem_84872.references = NULL;
    
    struct memblock ext_mem_84873;
    
    ext_mem_84873.references = NULL;
    
    struct memblock mem_84869;
    
    mem_84869.references = NULL;
    
    struct memblock mem_84867;
    
    mem_84867.references = NULL;
    
    struct memblock mem_84865;
    
    mem_84865.references = NULL;
    
    struct memblock mem_84863;
    
    mem_84863.references = NULL;
    
    struct memblock mem_param_83480;
    
    mem_param_83480.references = NULL;
    
    struct memblock mem_param_83476;
    
    mem_param_83476.references = NULL;
    
    struct memblock mem_param_83472;
    
    mem_param_83472.references = NULL;
    
    struct memblock mem_param_83468;
    
    mem_param_83468.references = NULL;
    
    struct memblock mem_param_83464;
    
    mem_param_83464.references = NULL;
    
    struct memblock mem_param_83460;
    
    mem_param_83460.references = NULL;
    
    struct memblock mem_param_83456;
    
    mem_param_83456.references = NULL;
    
    struct memblock mem_param_83452;
    
    mem_param_83452.references = NULL;
    
    struct memblock mem_param_83448;
    
    mem_param_83448.references = NULL;
    
    struct memblock mem_param_83444;
    
    mem_param_83444.references = NULL;
    
    struct memblock mem_param_83440;
    
    mem_param_83440.references = NULL;
    
    struct memblock mem_param_83436;
    
    mem_param_83436.references = NULL;
    
    struct memblock mem_param_83432;
    
    mem_param_83432.references = NULL;
    
    struct memblock mem_param_83428;
    
    mem_param_83428.references = NULL;
    
    struct memblock mem_param_83424;
    
    mem_param_83424.references = NULL;
    
    struct memblock mem_param_83420;
    
    mem_param_83420.references = NULL;
    
    struct memblock mem_param_83416;
    
    mem_param_83416.references = NULL;
    
    struct memblock mem_param_83412;
    
    mem_param_83412.references = NULL;
    
    struct memblock mem_param_83408;
    
    mem_param_83408.references = NULL;
    
    struct memblock mem_param_83404;
    
    mem_param_83404.references = NULL;
    
    struct memblock mem_param_83400;
    
    mem_param_83400.references = NULL;
    
    struct memblock mem_param_83396;
    
    mem_param_83396.references = NULL;
    
    struct memblock mem_param_83392;
    
    mem_param_83392.references = NULL;
    
    struct memblock mem_param_83388;
    
    mem_param_83388.references = NULL;
    
    struct memblock mem_param_83384;
    
    mem_param_83384.references = NULL;
    
    struct memblock mem_param_83380;
    
    mem_param_83380.references = NULL;
    
    struct memblock mem_param_83376;
    
    mem_param_83376.references = NULL;
    
    struct memblock ext_mem_85043;
    
    ext_mem_85043.references = NULL;
    
    struct memblock ext_mem_85044;
    
    ext_mem_85044.references = NULL;
    
    struct memblock ext_mem_85045;
    
    ext_mem_85045.references = NULL;
    
    struct memblock ext_mem_85046;
    
    ext_mem_85046.references = NULL;
    
    struct memblock ext_mem_85047;
    
    ext_mem_85047.references = NULL;
    
    struct memblock ext_mem_85048;
    
    ext_mem_85048.references = NULL;
    
    struct memblock ext_mem_85049;
    
    ext_mem_85049.references = NULL;
    
    struct memblock ext_mem_85050;
    
    ext_mem_85050.references = NULL;
    
    struct memblock ext_mem_85051;
    
    ext_mem_85051.references = NULL;
    
    struct memblock ext_mem_85052;
    
    ext_mem_85052.references = NULL;
    
    struct memblock ext_mem_85053;
    
    ext_mem_85053.references = NULL;
    
    struct memblock ext_mem_85054;
    
    ext_mem_85054.references = NULL;
    
    struct memblock ext_mem_85055;
    
    ext_mem_85055.references = NULL;
    
    struct memblock ext_mem_85056;
    
    ext_mem_85056.references = NULL;
    
    struct memblock ext_mem_85057;
    
    ext_mem_85057.references = NULL;
    
    struct memblock ext_mem_85058;
    
    ext_mem_85058.references = NULL;
    
    struct memblock ext_mem_85059;
    
    ext_mem_85059.references = NULL;
    
    struct memblock ext_mem_85060;
    
    ext_mem_85060.references = NULL;
    
    struct memblock ext_mem_85061;
    
    ext_mem_85061.references = NULL;
    
    struct memblock ext_mem_85062;
    
    ext_mem_85062.references = NULL;
    
    struct memblock ext_mem_85063;
    
    ext_mem_85063.references = NULL;
    
    struct memblock ext_mem_85064;
    
    ext_mem_85064.references = NULL;
    
    struct memblock ext_mem_85065;
    
    ext_mem_85065.references = NULL;
    
    struct memblock ext_mem_85066;
    
    ext_mem_85066.references = NULL;
    
    struct memblock ext_mem_85067;
    
    ext_mem_85067.references = NULL;
    
    struct memblock ext_mem_85068;
    
    ext_mem_85068.references = NULL;
    
    struct memblock ext_mem_85069;
    
    ext_mem_85069.references = NULL;
    
    struct memblock mem_out_85168;
    
    mem_out_85168.references = NULL;
    
    struct memblock mem_out_85167;
    
    mem_out_85167.references = NULL;
    
    struct memblock mem_out_85166;
    
    mem_out_85166.references = NULL;
    
    struct memblock mem_out_85165;
    
    mem_out_85165.references = NULL;
    
    struct memblock mem_out_85164;
    
    mem_out_85164.references = NULL;
    
    struct memblock mem_out_85163;
    
    mem_out_85163.references = NULL;
    
    struct memblock mem_out_85162;
    
    mem_out_85162.references = NULL;
    
    struct memblock mem_out_85161;
    
    mem_out_85161.references = NULL;
    
    struct memblock mem_out_85160;
    
    mem_out_85160.references = NULL;
    
    struct memblock mem_out_85159;
    
    mem_out_85159.references = NULL;
    
    struct memblock mem_out_85158;
    
    mem_out_85158.references = NULL;
    
    struct memblock mem_out_85157;
    
    mem_out_85157.references = NULL;
    
    struct memblock mem_out_85156;
    
    mem_out_85156.references = NULL;
    
    struct memblock mem_out_85155;
    
    mem_out_85155.references = NULL;
    
    struct memblock mem_out_85154;
    
    mem_out_85154.references = NULL;
    
    struct memblock mem_out_85153;
    
    mem_out_85153.references = NULL;
    
    struct memblock mem_out_85152;
    
    mem_out_85152.references = NULL;
    
    struct memblock mem_out_85151;
    
    mem_out_85151.references = NULL;
    
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    // futhark/microgpt.fut:448:31-42
    
    int64_t num_steps_76651 = mul64(num_batches_62011, batchsizze_62012);
    bool loop_nonempty_76652 = slt64((int64_t) 0, num_steps_76651);
    
    // futhark/microgpt.fut:452:27-38
    
    bool zzero_76653 = batchsizze_62012 == (int64_t) 0;
    
    // futhark/microgpt.fut:452:27-38
    
    bool nonzzero_76654 = !zzero_76653;
    bool loop_not_taken_76655 = !loop_nonempty_76652;
    bool protect_assert_disj_76656 = nonzzero_76654 || loop_not_taken_76655;
    
    // futhark/microgpt.fut:452:27-38
    
    bool nonzzero_cert_76657;
    
    if (!protect_assert_disj_76656) {
        set_error(ctx, msgprintf("Error: %s\n\nBacktrace:\n%s", "division by zero", "-> #0  futhark/microgpt.fut:452:27-38\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_76662 = sitofp_i64_f64(num_steps_76651);
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83481_cached_sizze_85562 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83481, &mem_83481_cached_sizze_85562, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83482_cached_sizze_85563 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83482, &mem_83482_cached_sizze_85563, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83491_cached_sizze_85564 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83491, &mem_83491_cached_sizze_85564, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83498_cached_sizze_85565 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83498, &mem_83498_cached_sizze_85565, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83513_cached_sizze_85566 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83513, &mem_83513_cached_sizze_85566, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83514_cached_sizze_85567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83514, &mem_83514_cached_sizze_85567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83523_cached_sizze_85568 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83523, &mem_83523_cached_sizze_85568, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83530_cached_sizze_85569 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83530, &mem_83530_cached_sizze_85569, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83545_cached_sizze_85570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83545, &mem_83545_cached_sizze_85570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83546_cached_sizze_85571 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83546, &mem_83546_cached_sizze_85571, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83555_cached_sizze_85572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83555, &mem_83555_cached_sizze_85572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83556_cached_sizze_85573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83556, &mem_83556_cached_sizze_85573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83577_cached_sizze_85574 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83577, &mem_83577_cached_sizze_85574, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83578_cached_sizze_85575 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83578, &mem_83578_cached_sizze_85575, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83579_cached_sizze_85576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83579, &mem_83579_cached_sizze_85576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83591_cached_sizze_85577 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83591, &mem_83591_cached_sizze_85577, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83592_cached_sizze_85578 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83592, &mem_83592_cached_sizze_85578, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83616_cached_sizze_85579 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83616, &mem_83616_cached_sizze_85579, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83617_cached_sizze_85580 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83617, &mem_83617_cached_sizze_85580, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83618_cached_sizze_85581 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83618, &mem_83618_cached_sizze_85581, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83619_cached_sizze_85582 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83619, &mem_83619_cached_sizze_85582, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83620_cached_sizze_85583 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83620, &mem_83620_cached_sizze_85583, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83639_cached_sizze_85584 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83639, &mem_83639_cached_sizze_85584, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83640_cached_sizze_85585 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83640, &mem_83640_cached_sizze_85585, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83641_cached_sizze_85586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83641, &mem_83641_cached_sizze_85586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83678_cached_sizze_85587 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83678, &mem_83678_cached_sizze_85587, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83679_cached_sizze_85588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83679, &mem_83679_cached_sizze_85588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83680_cached_sizze_85589 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83680, &mem_83680_cached_sizze_85589, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83696_cached_sizze_85590 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83696, &mem_83696_cached_sizze_85590, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83697_cached_sizze_85591 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83697, &mem_83697_cached_sizze_85591, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83698_cached_sizze_85592 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83698, &mem_83698_cached_sizze_85592, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83711_cached_sizze_85593 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83711, &mem_83711_cached_sizze_85593, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83712_cached_sizze_85594 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83712, &mem_83712_cached_sizze_85594, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83713_cached_sizze_85595 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83713, &mem_83713_cached_sizze_85595, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83759_cached_sizze_85596 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83759, &mem_83759_cached_sizze_85596, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83760_cached_sizze_85597 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83760, &mem_83760_cached_sizze_85597, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83771_cached_sizze_85598 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83771, &mem_83771_cached_sizze_85598, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83772_cached_sizze_85599 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83772, &mem_83772_cached_sizze_85599, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83781_cached_sizze_85600 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83781, &mem_83781_cached_sizze_85600, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83782_cached_sizze_85601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83782, &mem_83782_cached_sizze_85601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83803_cached_sizze_85602 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83803, &mem_83803_cached_sizze_85602, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83808_cached_sizze_85603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83808, &mem_83808_cached_sizze_85603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83819_cached_sizze_85604 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83819, &mem_83819_cached_sizze_85604, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83824_cached_sizze_85605 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83824, &mem_83824_cached_sizze_85605, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83831_cached_sizze_85606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83831, &mem_83831_cached_sizze_85606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83838_cached_sizze_85607 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83838, &mem_83838_cached_sizze_85607, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83849_cached_sizze_85608 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83849, &mem_83849_cached_sizze_85608, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83854_cached_sizze_85609 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83854, &mem_83854_cached_sizze_85609, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83875_cached_sizze_85610 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83875, &mem_83875_cached_sizze_85610, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83876_cached_sizze_85611 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83876, &mem_83876_cached_sizze_85611, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83884_cached_sizze_85612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83884, &mem_83884_cached_sizze_85612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83898_cached_sizze_85613 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83898, &mem_83898_cached_sizze_85613, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83903_cached_sizze_85614 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83903, &mem_83903_cached_sizze_85614, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83914_cached_sizze_85615 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83914, &mem_83914_cached_sizze_85615, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83919_cached_sizze_85616 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83919, &mem_83919_cached_sizze_85616, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83930_cached_sizze_85617 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83930, &mem_83930_cached_sizze_85617, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83931_cached_sizze_85618 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83931, &mem_83931_cached_sizze_85618, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83940_cached_sizze_85619 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83940, &mem_83940_cached_sizze_85619, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83941_cached_sizze_85620 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83941, &mem_83941_cached_sizze_85620, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83962_cached_sizze_85621 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83962, &mem_83962_cached_sizze_85621, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83963_cached_sizze_85622 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83963, &mem_83963_cached_sizze_85622, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83971_cached_sizze_85623 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83971, &mem_83971_cached_sizze_85623, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83985_cached_sizze_85624 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83985, &mem_83985_cached_sizze_85624, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83986_cached_sizze_85625 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83986, &mem_83986_cached_sizze_85625, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83994_cached_sizze_85626 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83994, &mem_83994_cached_sizze_85626, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84008_cached_sizze_85627 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84008, &mem_84008_cached_sizze_85627, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84013_cached_sizze_85628 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84013, &mem_84013_cached_sizze_85628, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84024_cached_sizze_85629 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84024, &mem_84024_cached_sizze_85629, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84029_cached_sizze_85630 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84029, &mem_84029_cached_sizze_85630, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84040_cached_sizze_85631 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84040, &mem_84040_cached_sizze_85631, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84045_cached_sizze_85632 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84045, &mem_84045_cached_sizze_85632, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84056_cached_sizze_85633 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84056, &mem_84056_cached_sizze_85633, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84057_cached_sizze_85634 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84057, &mem_84057_cached_sizze_85634, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84066_cached_sizze_85635 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84066, &mem_84066_cached_sizze_85635, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84067_cached_sizze_85636 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84067, &mem_84067_cached_sizze_85636, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84080_cached_sizze_85637 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84080, &mem_84080_cached_sizze_85637, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84081_cached_sizze_85638 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84081, &mem_84081_cached_sizze_85638, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84094_cached_sizze_85639 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84094, &mem_84094_cached_sizze_85639, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84095_cached_sizze_85640 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84095, &mem_84095_cached_sizze_85640, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84116_cached_sizze_85641 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84116, &mem_84116_cached_sizze_85641, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84123_cached_sizze_85642 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84123, &mem_84123_cached_sizze_85642, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84128_cached_sizze_85643 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84128, &mem_84128_cached_sizze_85643, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84139_cached_sizze_85644 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84139, &mem_84139_cached_sizze_85644, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84144_cached_sizze_85645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84144, &mem_84144_cached_sizze_85645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84155_cached_sizze_85646 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84155, &mem_84155_cached_sizze_85646, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84156_cached_sizze_85647 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84156, &mem_84156_cached_sizze_85647, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84165_cached_sizze_85648 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84165, &mem_84165_cached_sizze_85648, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84166_cached_sizze_85649 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84166, &mem_84166_cached_sizze_85649, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84187_cached_sizze_85650 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84187, &mem_84187_cached_sizze_85650, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84192_cached_sizze_85651 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84192, &mem_84192_cached_sizze_85651, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84203_cached_sizze_85652 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84203, &mem_84203_cached_sizze_85652, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84208_cached_sizze_85653 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84208, &mem_84208_cached_sizze_85653, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84219_cached_sizze_85654 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84219, &mem_84219_cached_sizze_85654, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84226_cached_sizze_85655 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84226, &mem_84226_cached_sizze_85655, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84233_cached_sizze_85656 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84233, &mem_84233_cached_sizze_85656, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84243_cached_sizze_85657 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84243, &mem_84243_cached_sizze_85657, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84248_cached_sizze_85658 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84248, &mem_84248_cached_sizze_85658, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84259_cached_sizze_85659 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84259, &mem_84259_cached_sizze_85659, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84260_cached_sizze_85660 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84260, &mem_84260_cached_sizze_85660, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84269_cached_sizze_85661 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84269, &mem_84269_cached_sizze_85661, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84270_cached_sizze_85662 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84270, &mem_84270_cached_sizze_85662, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84291_cached_sizze_85663 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84291, &mem_84291_cached_sizze_85663, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84292_cached_sizze_85664 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84292, &mem_84292_cached_sizze_85664, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84303_cached_sizze_85665 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84303, &mem_84303_cached_sizze_85665, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84304_cached_sizze_85666 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84304, &mem_84304_cached_sizze_85666, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84313_cached_sizze_85667 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84313, &mem_84313_cached_sizze_85667, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84320_cached_sizze_85668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84320, &mem_84320_cached_sizze_85668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84345_cached_sizze_85669 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84345, &mem_84345_cached_sizze_85669, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84346_cached_sizze_85670 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84346, &mem_84346_cached_sizze_85670, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84357_cached_sizze_85671 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84357, &mem_84357_cached_sizze_85671, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84358_cached_sizze_85672 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84358, &mem_84358_cached_sizze_85672, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84367_cached_sizze_85673 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84367, &mem_84367_cached_sizze_85673, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84374_cached_sizze_85674 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84374, &mem_84374_cached_sizze_85674, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84381_cached_sizze_85675 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84381, &mem_84381_cached_sizze_85675, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84388_cached_sizze_85676 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84388, &mem_84388_cached_sizze_85676, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84413_cached_sizze_85677 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84413, &mem_84413_cached_sizze_85677, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84414_cached_sizze_85678 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84414, &mem_84414_cached_sizze_85678, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84425_cached_sizze_85679 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84425, &mem_84425_cached_sizze_85679, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84426_cached_sizze_85680 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84426, &mem_84426_cached_sizze_85680, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84435_cached_sizze_85681 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84435, &mem_84435_cached_sizze_85681, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84442_cached_sizze_85682 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84442, &mem_84442_cached_sizze_85682, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84467_cached_sizze_85683 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84467, &mem_84467_cached_sizze_85683, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84472_cached_sizze_85684 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84472, &mem_84472_cached_sizze_85684, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84483_cached_sizze_85685 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84483, &mem_84483_cached_sizze_85685, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84489_cached_sizze_85686 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84489, &mem_84489_cached_sizze_85686, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84494_cached_sizze_85687 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84494, &mem_84494_cached_sizze_85687, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84510_cached_sizze_85688 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84510, &mem_84510_cached_sizze_85688, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84516_cached_sizze_85689 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84516, &mem_84516_cached_sizze_85689, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84521_cached_sizze_85690 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84521, &mem_84521_cached_sizze_85690, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84537_cached_sizze_85691 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84537, &mem_84537_cached_sizze_85691, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84538_cached_sizze_85692 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84538, &mem_84538_cached_sizze_85692, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84549_cached_sizze_85693 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84549, &mem_84549_cached_sizze_85693, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84550_cached_sizze_85694 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84550, &mem_84550_cached_sizze_85694, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84559_cached_sizze_85695 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84559, &mem_84559_cached_sizze_85695, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84560_cached_sizze_85696 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84560, &mem_84560_cached_sizze_85696, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84591_cached_sizze_85697 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84591, &mem_84591_cached_sizze_85697, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84592_cached_sizze_85698 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84592, &mem_84592_cached_sizze_85698, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84593_cached_sizze_85699 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84593, &mem_84593_cached_sizze_85699, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84606_cached_sizze_85700 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84606, &mem_84606_cached_sizze_85700, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84607_cached_sizze_85701 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84607, &mem_84607_cached_sizze_85701, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84608_cached_sizze_85702 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84608, &mem_84608_cached_sizze_85702, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84639_cached_sizze_85703 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84639, &mem_84639_cached_sizze_85703, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84640_cached_sizze_85704 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84640, &mem_84640_cached_sizze_85704, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84641_cached_sizze_85705 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84641, &mem_84641_cached_sizze_85705, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84642_cached_sizze_85706 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84642, &mem_84642_cached_sizze_85706, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84659_cached_sizze_85707 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84659, &mem_84659_cached_sizze_85707, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84660_cached_sizze_85708 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84660, &mem_84660_cached_sizze_85708, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84661_cached_sizze_85709 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84661, &mem_84661_cached_sizze_85709, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84662_cached_sizze_85710 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84662, &mem_84662_cached_sizze_85710, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84703_cached_sizze_85711 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84703, &mem_84703_cached_sizze_85711, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84710_cached_sizze_85712 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84710, &mem_84710_cached_sizze_85712, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84717_cached_sizze_85713 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84717, &mem_84717_cached_sizze_85713, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84727_cached_sizze_85714 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84727, &mem_84727_cached_sizze_85714, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84732_cached_sizze_85715 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84732, &mem_84732_cached_sizze_85715, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84743_cached_sizze_85716 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84743, &mem_84743_cached_sizze_85716, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84750_cached_sizze_85717 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84750, &mem_84750_cached_sizze_85717, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84757_cached_sizze_85718 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84757, &mem_84757_cached_sizze_85718, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84767_cached_sizze_85719 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84767, &mem_84767_cached_sizze_85719, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84772_cached_sizze_85720 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84772, &mem_84772_cached_sizze_85720, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84783_cached_sizze_85721 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84783, &mem_84783_cached_sizze_85721, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84784_cached_sizze_85722 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84784, &mem_84784_cached_sizze_85722, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84793_cached_sizze_85723 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84793, &mem_84793_cached_sizze_85723, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84794_cached_sizze_85724 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84794, &mem_84794_cached_sizze_85724, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84815_cached_sizze_85725 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84815, &mem_84815_cached_sizze_85725, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84820_cached_sizze_85726 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84820, &mem_84820_cached_sizze_85726, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84831_cached_sizze_85727 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84831, &mem_84831_cached_sizze_85727, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84832_cached_sizze_85728 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84832, &mem_84832_cached_sizze_85728, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84841_cached_sizze_85729 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84841, &mem_84841_cached_sizze_85729, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84842_cached_sizze_85730 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84842, &mem_84842_cached_sizze_85730, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:450:5-454:55
    if (memblock_set(ctx, &mem_param_83376, &wdown_mem_83344, "wdown_mem_83344") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83380, &wkey_mem_83345, "wkey_mem_83345") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83384, &wout_mem_83346, "wout_mem_83346") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83388, &wpe_mem_83347, "wpe_mem_83347") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83392, &wqry_mem_83348, "wqry_mem_83348") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83396, &wte_mem_83349, "wte_mem_83349") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83400, &wup_mem_83350, "wup_mem_83350") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83404, &wval_mem_83351, "wval_mem_83351") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83408, &wvoc_mem_83352, "wvoc_mem_83352") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83412, &wdown_mem_83353, "wdown_mem_83353") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83416, &wkey_mem_83354, "wkey_mem_83354") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83420, &wout_mem_83355, "wout_mem_83355") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83424, &wpe_mem_83356, "wpe_mem_83356") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83428, &wqry_mem_83357, "wqry_mem_83357") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83432, &wte_mem_83358, "wte_mem_83358") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83436, &wup_mem_83359, "wup_mem_83359") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83440, &wval_mem_83360, "wval_mem_83360") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83444, &wvoc_mem_83361, "wvoc_mem_83361") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83448, &wdown_mem_83362, "wdown_mem_83362") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83452, &wkey_mem_83363, "wkey_mem_83363") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83456, &wout_mem_83364, "wout_mem_83364") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83460, &wpe_mem_83365, "wpe_mem_83365") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83464, &wqry_mem_83366, "wqry_mem_83366") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83468, &wte_mem_83367, "wte_mem_83367") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83472, &wup_mem_83368, "wup_mem_83368") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83476, &wval_mem_83369, "wval_mem_83369") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83480, &wvoc_mem_83370, "wvoc_mem_83370") != 0)
        return 1;
    for (int64_t step_76690 = 0; step_76690 < num_steps_76651; step_76690++) {
        // futhark/microgpt.fut:452:27-38
        
        int64_t seq_76718 = sdiv64(step_76690, batchsizze_62012);
        
        // futhark/microgpt.fut:452:17-39
        
        bool x_76719 = sle64((int64_t) 0, seq_76718);
        
        // futhark/microgpt.fut:452:17-39
        
        bool y_76720 = slt64(seq_76718, num_batches_62011);
        
        // futhark/microgpt.fut:452:17-39
        
        bool bounds_check_76721 = x_76719 && y_76720;
        
        // futhark/microgpt.fut:452:17-39
        
        bool index_certs_76722;
        
        if (!bounds_check_76721) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seq_76718, "] out of bounds for array of shape [", (long long) num_batches_62011, "].", "-> #0  futhark/microgpt.fut:452:17-39\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:452:45-56
        
        int64_t seq_76723 = smod64(step_76690, batchsizze_62012);
        
        // futhark/microgpt.fut:452:17-57
        
        bool x_76724 = sle64((int64_t) 0, seq_76723);
        
        // futhark/microgpt.fut:452:17-57
        
        bool y_76725 = slt64(seq_76723, batchsizze_62012);
        
        // futhark/microgpt.fut:452:17-57
        
        bool bounds_check_76726 = x_76724 && y_76725;
        
        // futhark/microgpt.fut:452:17-57
        
        bool index_certs_76727;
        
        if (!bounds_check_76726) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seq_76723, "] out of bounds for array of shape [", (long long) batchsizze_62012, "].", "-> #0  futhark/microgpt.fut:452:17-57\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82497 = 0; i_82497 < (int64_t) 16; i_82497++) {
            // futhark/microgpt.fut:352:25-76
            
            bool cond_78516 = slt64(i_82497, (int64_t) 15);
            
            // futhark/microgpt.fut:352:51-54
            
            int64_t zeze_lhs_78517 = add64((int64_t) 1, i_82497);
            
            // futhark/microgpt.fut:352:42-55
            
            bool x_78518 = sle64((int64_t) 0, zeze_lhs_78517);
            
            // futhark/microgpt.fut:352:42-55
            
            bool y_78519 = slt64(zeze_lhs_78517, (int64_t) 16);
            
            // futhark/microgpt.fut:352:42-55
            
            bool bounds_check_78520 = x_78518 && y_78519;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_78521 = !cond_78516;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_78522 = bounds_check_78520 || loop_not_taken_78521;
            
            // futhark/microgpt.fut:352:42-55
            
            bool index_certs_78523;
            
            if (!protect_assert_disj_78522) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_78517, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:352:42-55\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:352:3-78\n   #6  futhark/microgpt.fut:368:18-35\n   #7  futhark/microgpt.fut:420:33-426:28\n   #8  futhark/microgpt.fut:454:11-54\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_78538 = ((int64_t *) seqs_mem_83372.mem)[seq_76718 * ((int64_t) 16 * batchsizze_62012) + seq_76723 * (int64_t) 16 + i_82497];
            
            // futhark/microgpt.fut:370:37-51
            
            bool x_78539 = sle64((int64_t) 0, tmp_78538);
            
            // futhark/microgpt.fut:370:37-51
            
            bool y_78540 = slt64(tmp_78538, (int64_t) 27);
            
            // futhark/microgpt.fut:370:37-51
            
            bool bounds_check_78541 = x_78539 && y_78540;
            
            // futhark/microgpt.fut:370:37-51
            
            bool index_certs_78542;
            
            if (!bounds_check_78541) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_78538, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:370:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:370:16-55\n   #6  futhark/microgpt.fut:420:33-426:28\n   #7  futhark/microgpt.fut:454:11-54\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:352:42-55
            
            int64_t zeze_lhs_78524;
            
            if (cond_78516) {
                int64_t x_82306 = ((int64_t *) seqs_mem_83372.mem)[seq_76718 * ((int64_t) 16 * batchsizze_62012) + seq_76723 * (int64_t) 16 + zeze_lhs_78517];
                
                zeze_lhs_78524 = x_82306;
            } else {
                zeze_lhs_78524 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82487 = 0; i_82487 < (int64_t) 27; i_82487++) {
                // futhark/microgpt.fut:352:56-60
                
                bool cond_t_res_78528 = zeze_lhs_78524 == i_82487;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_78529 = cond_78516 && cond_t_res_78528;
                
                // futhark/microgpt.fut:352:25-76
                
                double lifted_lambda_res_78530;
                
                if (x_78529) {
                    lifted_lambda_res_78530 = 1.0;
                } else {
                    lifted_lambda_res_78530 = 0.0;
                }
                ((double *) mem_83491)[i_82487] = lifted_lambda_res_78530;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82491 = 0; i_82491 < (int64_t) 16; i_82491++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_78549 = ((double *) mem_param_83396.mem)[tmp_78538 * (int64_t) 16 + i_82491];
                
                ((double *) mem_83498)[i_82491] = lifted_lambda_res_78549;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83481, i_82497 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83498, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83482, i_82497 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83491, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82512 = 0; i_82512 < (int64_t) 16; i_82512++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82502 = 0; i_82502 < (int64_t) 16; i_82502++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78574 = ((double *) mem_param_83388.mem)[i_82512 * (int64_t) 16 + i_82502];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_78575 = ((double *) mem_83481)[i_82512 * (int64_t) 16 + i_82502];
                
                // futhark/microgpt.fut:211:35-63
                
                double zp_res_78576 = zp_lhs_78574 + zp_rhs_78575;
                
                ((double *) mem_83523)[i_82502] = zp_res_78576;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82506 = 0; i_82506 < (int64_t) 27; i_82506++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78590 = ((double *) mem_83482)[i_82512 * (int64_t) 27 + i_82506];
                
                // futhark/microgpt.fut:243:51-87
                
                double zt_res_78591 = -6.25e-2 * zt_rhs_78590;
                
                ((double *) mem_83530)[i_82506] = zt_res_78591;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83513, i_82512 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83530, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83514, i_82512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83523, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82526 = 0; i_82526 < (int64_t) 16; i_82526++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78610;
            double r_78612 = 0.0;
            
            for (int64_t i_78611 = 0; i_78611 < (int64_t) 16; i_78611++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78613 = ((double *) mem_83514)[i_82526 * (int64_t) 16 + i_78611];
                
                // futhark/microgpt.fut:212:58-83
                
                double zt_res_78614 = zt_lhs_78613 * zt_lhs_78613;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78615 = r_78612 + zt_res_78614;
                double r_tmp_85233 = zp_res_78615;
                
                r_78612 = r_tmp_85233;
            }
            defunc_0_lifted_lambda_res_78610 = r_78612;
            // futhark/microgpt.fut:212:40-101
            
            double zs_res_78616 = defunc_0_lifted_lambda_res_78610 / 16.0;
            
            // futhark/microgpt.fut:213:23-53
            
            double zp_res_78617 = 1.0e-5 + zs_res_78616;
            
            // futhark/microgpt.fut:213:15-53
            
            double sqrt_res_78618 = futrts_sqrt64(zp_res_78617);
            
            // futhark/microgpt.fut:214:39-49
            
            double zs_res_78619 = 1.0 / sqrt_res_78618;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82519 = 0; i_82519 < (int64_t) 16; i_82519++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80615 = ((double *) mem_83514)[i_82526 * (int64_t) 16 + i_82519];
                
                // futhark/microgpt.fut:214:23-49
                
                double zt_res_80616 = zs_res_78619 * zt_lhs_80615;
                
                // futhark/microgpt.fut:286:53-86
                
                double zt_res_80624 = zt_lhs_80615 * zt_lhs_80615;
                
                ((double *) mem_83555)[i_82519] = zt_res_80624;
                ((double *) mem_83556)[i_82519] = zt_res_80616;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83545, i_82526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83555, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83546, i_82526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83556, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82542 = 0; i_82542 < (int64_t) 16; i_82542++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78718;
            double r_78720 = 0.0;
            
            for (int64_t i_78719 = 0; i_78719 < (int64_t) 16; i_78719++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78721 = ((double *) mem_83546)[i_82542 * (int64_t) 16 + i_78719];
                
                // futhark/microgpt.fut:215:61-90
                
                double zt_res_78722 = zt_lhs_78721 * zt_lhs_78721;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78723 = r_78720 + zt_res_78722;
                double r_tmp_85239 = zp_res_78723;
                
                r_78720 = r_tmp_85239;
            }
            defunc_0_lifted_lambda_res_78718 = r_78720;
            // futhark/microgpt.fut:215:42-108
            
            double zs_res_78724 = defunc_0_lifted_lambda_res_78718 / 16.0;
            
            // futhark/microgpt.fut:216:24-55
            
            double zp_res_78725 = 1.0e-5 + zs_res_78724;
            
            // futhark/microgpt.fut:216:16-55
            
            double sqrt_res_78726 = futrts_sqrt64(zp_res_78725);
            
            // futhark/microgpt.fut:217:42-53
            
            double zs_res_78727 = 1.0 / sqrt_res_78726;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82533 = 0; i_82533 < (int64_t) 16; i_82533++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80644 = ((double *) mem_83546)[i_82542 * (int64_t) 16 + i_82533];
                
                // futhark/microgpt.fut:217:24-53
                
                double zt_res_80645 = zs_res_78727 * zt_lhs_80644;
                
                // futhark/microgpt.fut:279:53-86
                
                double zt_res_80653 = zt_lhs_80644 * zt_lhs_80644;
                
                ((double *) mem_83591)[i_82533] = zt_res_80653;
                ((double *) mem_83592)[i_82533] = zt_res_80645;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78761;
            double r_78763 = 0.0;
            
            for (int64_t i_78762 = 0; i_78762 < (int64_t) 16; i_78762++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_78764 = ((double *) mem_83545)[i_82542 * (int64_t) 16 + i_78762];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78765 = r_78763 + lifted_lambda_res_78764;
                double r_tmp_85242 = zp_res_78765;
                
                r_78763 = r_tmp_85242;
            }
            defunc_0_lifted_lambda_res_78761 = r_78763;
            // futhark/microgpt.fut:287:34-86
            
            double zs_res_78766 = defunc_0_lifted_lambda_res_78761 / 16.0;
            
            ((double *) mem_83577)[i_82542] = zs_res_78766;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83578, i_82542 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83591, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83579, i_82542 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83592, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82566 = 0; i_82566 < (int64_t) 16; i_82566++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82552 = 0; i_82552 < (int64_t) 16; i_82552++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80716;
                double r_80718 = 0.0;
                
                for (int64_t i_80717 = 0; i_80717 < (int64_t) 16; i_80717++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80719 = ((double *) mem_param_83392.mem)[i_82552 * (int64_t) 16 + i_80717];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80720 = ((double *) mem_83579)[i_82566 * (int64_t) 16 + i_80717];
                    
                    // futhark/microgpt.fut:218:69-100
                    
                    double zt_res_80721 = zt_lhs_80719 * zt_rhs_80720;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80722 = r_80718 + zt_res_80721;
                    double r_tmp_85251 = zp_res_80722;
                    
                    r_80718 = r_tmp_85251;
                }
                defunc_0_lifted_lambda_res_80716 = r_80718;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80729;
                double r_80731 = 0.0;
                
                for (int64_t i_80730 = 0; i_80730 < (int64_t) 16; i_80730++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80732 = ((double *) mem_param_83380.mem)[i_82552 * (int64_t) 16 + i_80730];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80733 = ((double *) mem_83579)[i_82566 * (int64_t) 16 + i_80730];
                    
                    // futhark/microgpt.fut:219:69-100
                    
                    double zt_res_80734 = zt_lhs_80732 * zt_rhs_80733;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80735 = r_80731 + zt_res_80734;
                    double r_tmp_85252 = zp_res_80735;
                    
                    r_80731 = r_tmp_85252;
                }
                defunc_0_lifted_lambda_res_80729 = r_80731;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80745;
                double r_80747 = 0.0;
                
                for (int64_t i_80746 = 0; i_80746 < (int64_t) 16; i_80746++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80748 = ((double *) mem_param_83404.mem)[i_82552 * (int64_t) 16 + i_80746];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80749 = ((double *) mem_83579)[i_82566 * (int64_t) 16 + i_80746];
                    
                    // futhark/microgpt.fut:220:69-100
                    
                    double zt_res_80750 = zt_lhs_80748 * zt_rhs_80749;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80751 = r_80747 + zt_res_80750;
                    double r_tmp_85253 = zp_res_80751;
                    
                    r_80747 = r_tmp_85253;
                }
                defunc_0_lifted_lambda_res_80745 = r_80747;
                ((double *) mem_83639)[i_82552] = defunc_0_lifted_lambda_res_80745;
                ((double *) mem_83640)[i_82552] = defunc_0_lifted_lambda_res_80729;
                ((double *) mem_83641)[i_82552] = defunc_0_lifted_lambda_res_80716;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79108;
            double r_79110 = 0.0;
            
            for (int64_t i_79109 = 0; i_79109 < (int64_t) 16; i_79109++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79111 = ((double *) mem_83578)[i_82566 * (int64_t) 16 + i_79109];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79112 = r_79110 + lifted_lambda_res_79111;
                double r_tmp_85254 = zp_res_79112;
                
                r_79110 = r_tmp_85254;
            }
            defunc_0_lifted_lambda_res_79108 = r_79110;
            // futhark/microgpt.fut:280:34-86
            
            double zs_res_79113 = defunc_0_lifted_lambda_res_79108 / 16.0;
            
            // futhark/microgpt.fut:288:41-51
            
            double zp_lhs_79127 = ((double *) mem_83577)[i_82566];
            
            // futhark/microgpt.fut:288:41-79
            
            double zp_res_79128 = 1.0e-5 + zp_lhs_79127;
            
            // futhark/microgpt.fut:288:33-79
            
            double sqrt_res_79129 = futrts_sqrt64(zp_res_79128);
            
            ((double *) mem_83616)[i_82566] = sqrt_res_79129;
            ((double *) mem_83617)[i_82566] = zs_res_79113;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83618, i_82566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83639, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83619, i_82566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83640, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83620, i_82566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83641, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82598 = 0; i_82598 < (int64_t) 4; i_82598++) {
            // futhark/microgpt.fut:221:81-84
            
            int64_t zp_lhs_79201 = mul64((int64_t) 4, i_82598);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82588 = 0; i_82588 < (int64_t) 16; i_82588++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82578 = 0; i_82578 < (int64_t) 4; i_82578++) {
                    // futhark/microgpt.fut:221:86-91
                    
                    int64_t tmp_80909 = add64(zp_lhs_79201, i_82578);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool x_80910 = sle64((int64_t) 0, tmp_80909);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool y_80911 = slt64(tmp_80909, (int64_t) 16);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool bounds_check_80912 = x_80910 && y_80911;
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool index_certs_80913;
                    
                    if (!bounds_check_80912) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_80909, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:221:66-93\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:221:49-94\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:221:30-96\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:221:12-98\n   #10 futhark/microgpt.fut:373:5-76\n   #11 futhark/microgpt.fut:420:33-426:28\n   #12 futhark/microgpt.fut:454:11-54\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80914 = ((double *) mem_83620)[i_82588 * (int64_t) 16 + tmp_80909];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80922 = ((double *) mem_83619)[i_82588 * (int64_t) 16 + tmp_80909];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80933 = ((double *) mem_83618)[i_82588 * (int64_t) 16 + tmp_80909];
                    
                    ((double *) mem_83711)[i_82578] = lifted_lambda_res_80933;
                    ((double *) mem_83712)[i_82578] = lifted_lambda_res_80922;
                    ((double *) mem_83713)[i_82578] = lifted_lambda_res_80914;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83696, i_82588 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83711, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83697, i_82588 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83712, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83698, i_82588 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83713, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83678, i_82598 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83696, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83679, i_82598 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83697, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83680, i_82598 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83698, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82654 = 0; i_82654 < (int64_t) 4; i_82654++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82613 = 0; i_82613 < (int64_t) 16; i_82613++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82606 = 0; i_82606 < (int64_t) 16; i_82606++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81012;
                    double r_81014 = 0.0;
                    
                    for (int64_t i_81013 = 0; i_81013 < (int64_t) 4; i_81013++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81015 = ((double *) mem_83680)[i_82654 * (int64_t) 64 + i_82613 * (int64_t) 4 + i_81013];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81016 = ((double *) mem_83679)[i_82654 * (int64_t) 64 + i_82606 * (int64_t) 4 + i_81013];
                        
                        // futhark/microgpt.fut:224:97-138
                        
                        double zt_res_81017 = zt_lhs_81015 * zt_rhs_81016;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81018 = r_81014 + zt_res_81017;
                        double r_tmp_85270 = zp_res_81018;
                        
                        r_81014 = r_tmp_85270;
                    }
                    defunc_0_lifted_lambda_res_81012 = r_81014;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81025;
                    double r_81027 = 0.0;
                    
                    for (int64_t i_81026 = 0; i_81026 < (int64_t) 4; i_81026++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81028 = ((double *) mem_83680)[i_82654 * (int64_t) 64 + i_82613 * (int64_t) 4 + i_81026];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81029 = ((double *) mem_83679)[i_82654 * (int64_t) 64 + i_82606 * (int64_t) 4 + i_81026];
                        
                        // futhark/microgpt.fut:263:91-138
                        
                        double zt_res_81030 = zt_lhs_81028 * zt_rhs_81029;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81031 = r_81027 + zt_res_81030;
                        double r_tmp_85271 = zp_res_81031;
                        
                        r_81027 = r_tmp_85271;
                    }
                    defunc_0_lifted_lambda_res_81025 = r_81027;
                    ((double *) mem_83781)[i_82606] = defunc_0_lifted_lambda_res_81025;
                    ((double *) mem_83782)[i_82606] = defunc_0_lifted_lambda_res_81012;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83771, i_82613 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83772, i_82613 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83782, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82622 = 0; i_82622 < (int64_t) 16; i_82622++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82618 = 0; i_82618 < (int64_t) 16; i_82618++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_79310 = ((double *) mem_83772)[i_82622 * (int64_t) 16 + i_82618];
                    
                    // futhark/microgpt.fut:225:43-70
                    
                    double zs_res_79311 = zs_lhs_79310 / 2.0;
                    double zp_rhs_79312 = ((double *) masks_mem_83371.mem)[seq_76718 * ((int64_t) 256 * batchsizze_62012) + seq_76723 * (int64_t) 256 + i_82622 * (int64_t) 16 + i_82618];
                    
                    // futhark/microgpt.fut:225:57-90
                    
                    double zp_res_79313 = zs_res_79311 + zp_rhs_79312;
                    
                    ((double *) mem_83808)[i_82618] = zp_res_79313;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83803, i_82622 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83808, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82640 = 0; i_82640 < (int64_t) 16; i_82640++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_82327;
                double redout_82624 = -INFINITY;
                
                for (int64_t i_82625 = 0; i_82625 < (int64_t) 16; i_82625++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81049 = ((double *) mem_83803)[i_82640 * (int64_t) 16 + i_82625];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_79334 = fmax64(lifted_lambda_res_81049, redout_82624);
                    double redout_tmp_85275 = max_res_79334;
                    
                    redout_82624 = redout_tmp_85275;
                }
                defunc_0_reduce_res_82327 = redout_82624;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_79335 = -defunc_0_reduce_res_82327;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82628 = 0; i_82628 < (int64_t) 16; i_82628++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_79342 = ((double *) mem_83803)[i_82640 * (int64_t) 16 + i_82628];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_79343 = neg_res_79335 + lifted_lambda_res_79342;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_79344 = futrts_exp64(zp_res_79343);
                    
                    ((double *) mem_83824)[i_82628] = exp_res_79344;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79346;
                double r_79348 = 0.0;
                
                for (int64_t i_79347 = 0; i_79347 < (int64_t) 16; i_79347++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_79349 = ((double *) mem_83824)[i_79347];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79350 = r_79348 + lifted_lambda_res_79349;
                    double r_tmp_85277 = zp_res_79350;
                    
                    r_79348 = r_tmp_85277;
                }
                defunc_0_lifted_lambda_res_79346 = r_79348;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82632 = 0; i_82632 < (int64_t) 16; i_82632++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_79357 = ((double *) mem_83824)[i_82632];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_79358 = zs_lhs_79357 / defunc_0_lifted_lambda_res_79346;
                    
                    ((double *) mem_83831)[i_82632] = zs_res_79358;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82636 = 0; i_82636 < (int64_t) 16; i_82636++) {
                    // futhark/microgpt.fut:227:23-31
                    
                    double lifted_lambda_res_79366 = ((double *) mem_83831)[i_82636];
                    
                    ((double *) mem_83838)[i_82636] = lifted_lambda_res_79366;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83819, i_82640 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82648 = 0; i_82648 < (int64_t) 16; i_82648++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82644 = 0; i_82644 < (int64_t) 4; i_82644++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_79381;
                    double r_79383 = 0.0;
                    
                    for (int64_t i_79382 = 0; i_79382 < (int64_t) 16; i_79382++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_79384 = ((double *) mem_83819)[i_82648 * (int64_t) 16 + i_79382];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_79385 = ((double *) mem_83678)[i_82654 * (int64_t) 64 + i_79382 * (int64_t) 4 + i_82644];
                        
                        // futhark/microgpt.fut:228:61-97
                        
                        double zt_res_79386 = zt_lhs_79384 * zt_rhs_79385;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_79387 = r_79383 + zt_res_79386;
                        double r_tmp_85282 = zp_res_79387;
                        
                        r_79383 = r_tmp_85282;
                    }
                    defunc_0_lifted_lambda_res_79381 = r_79383;
                    ((double *) mem_83854)[i_82644] = defunc_0_lifted_lambda_res_79381;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83849, i_82648 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83854, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83759, i_82654 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_83771, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83760, i_82654 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83849, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82665 = 0; i_82665 < (int64_t) 16; i_82665++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82659 = 0; i_82659 < (int64_t) 16; i_82659++) {
                // futhark/microgpt.fut:229:58-61
                
                int64_t tmp_79436 = sdiv64(i_82659, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-63
                
                bool x_79437 = sle64((int64_t) 0, tmp_79436);
                
                // futhark/microgpt.fut:229:49-63
                
                bool y_79438 = slt64(tmp_79436, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-63
                
                bool bounds_check_79439 = x_79437 && y_79438;
                
                // futhark/microgpt.fut:229:49-63
                
                bool index_certs_79440;
                
                if (!bounds_check_79439) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79436, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:229:49-63\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:229:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:229:12-82\n   #7  futhark/microgpt.fut:373:5-76\n   #8  futhark/microgpt.fut:420:33-426:28\n   #9  futhark/microgpt.fut:454:11-54\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:229:74-77
                
                int64_t tmp_79441 = smod64(i_82659, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-79
                
                bool x_79442 = sle64((int64_t) 0, tmp_79441);
                
                // futhark/microgpt.fut:229:49-79
                
                bool y_79443 = slt64(tmp_79441, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-79
                
                bool bounds_check_79444 = x_79442 && y_79443;
                
                // futhark/microgpt.fut:229:49-79
                
                bool index_certs_79445;
                
                if (!bounds_check_79444) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79441, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:229:49-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:229:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:229:12-82\n   #7  futhark/microgpt.fut:373:5-76\n   #8  futhark/microgpt.fut:420:33-426:28\n   #9  futhark/microgpt.fut:454:11-54\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79446 = ((double *) mem_83760)[tmp_79436 * (int64_t) 64 + i_82665 * (int64_t) 4 + tmp_79441];
                
                ((double *) mem_83884)[i_82659] = lifted_lambda_res_79446;
            }
            // futhark/microgpt.fut:281:41-51
            
            double zp_lhs_79454 = ((double *) mem_83617)[i_82665];
            
            // futhark/microgpt.fut:281:41-79
            
            double zp_res_79455 = 1.0e-5 + zp_lhs_79454;
            
            // futhark/microgpt.fut:281:33-79
            
            double sqrt_res_79456 = futrts_sqrt64(zp_res_79455);
            
            ((double *) mem_83875)[i_82665] = sqrt_res_79456;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83876, i_82665 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83884, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82674 = 0; i_82674 < (int64_t) 16; i_82674++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82670 = 0; i_82670 < (int64_t) 16; i_82670++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77112;
                double r_77114 = 0.0;
                
                for (int64_t i_77113 = 0; i_77113 < (int64_t) 16; i_77113++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77115 = ((double *) mem_param_83384.mem)[i_82670 * (int64_t) 16 + i_77113];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77116 = ((double *) mem_83876)[i_82674 * (int64_t) 16 + i_77113];
                    
                    // futhark/microgpt.fut:230:69-101
                    
                    double zt_res_77117 = zt_lhs_77115 * zt_rhs_77116;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77118 = r_77114 + zt_res_77117;
                    double r_tmp_85288 = zp_res_77118;
                    
                    r_77114 = r_tmp_85288;
                }
                defunc_0_lifted_lambda_res_77112 = r_77114;
                ((double *) mem_83903)[i_82670] = defunc_0_lifted_lambda_res_77112;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83898, i_82674 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83903, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82682 = 0; i_82682 < (int64_t) 16; i_82682++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82678 = 0; i_82678 < (int64_t) 16; i_82678++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77133 = ((double *) mem_83898)[i_82682 * (int64_t) 16 + i_82678];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77134 = ((double *) mem_83546)[i_82682 * (int64_t) 16 + i_82678];
                
                // futhark/microgpt.fut:231:38-68
                
                double zp_res_77135 = zp_lhs_77133 + zp_rhs_77134;
                
                ((double *) mem_83919)[i_82678] = zp_res_77135;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83914, i_82682 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83919, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82695 = 0; i_82695 < (int64_t) 16; i_82695++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79474;
            double r_79476 = 0.0;
            
            for (int64_t i_79475 = 0; i_79475 < (int64_t) 16; i_79475++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_79477 = ((double *) mem_83914)[i_82695 * (int64_t) 16 + i_79475];
                
                // futhark/microgpt.fut:232:62-93
                
                double zt_res_79478 = zt_lhs_79477 * zt_lhs_79477;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79479 = r_79476 + zt_res_79478;
                double r_tmp_85293 = zp_res_79479;
                
                r_79476 = r_tmp_85293;
            }
            defunc_0_lifted_lambda_res_79474 = r_79476;
            // futhark/microgpt.fut:232:43-111
            
            double zs_res_79480 = defunc_0_lifted_lambda_res_79474 / 16.0;
            
            // futhark/microgpt.fut:233:24-55
            
            double zp_res_79481 = 1.0e-5 + zs_res_79480;
            
            // futhark/microgpt.fut:233:16-55
            
            double sqrt_res_79482 = futrts_sqrt64(zp_res_79481);
            
            // futhark/microgpt.fut:234:43-54
            
            double zs_res_79483 = 1.0 / sqrt_res_79482;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82688 = 0; i_82688 < (int64_t) 16; i_82688++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81090 = ((double *) mem_83914)[i_82695 * (int64_t) 16 + i_82688];
                
                // futhark/microgpt.fut:234:24-54
                
                double zt_res_81091 = zs_res_79483 * zt_lhs_81090;
                
                // futhark/microgpt.fut:254:53-88
                
                double zt_res_81099 = zt_lhs_81090 * zt_lhs_81090;
                
                ((double *) mem_83940)[i_82688] = zt_res_81099;
                ((double *) mem_83941)[i_82688] = zt_res_81091;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83930, i_82695 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83940, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83931, i_82695 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83941, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82706 = 0; i_82706 < (int64_t) 16; i_82706++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82700 = 0; i_82700 < (int64_t) 64; i_82700++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79531;
                double r_79533 = 0.0;
                
                for (int64_t i_79532 = 0; i_79532 < (int64_t) 16; i_79532++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_79534 = ((double *) mem_param_83400.mem)[i_82700 * (int64_t) 16 + i_79532];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_79535 = ((double *) mem_83931)[i_82706 * (int64_t) 16 + i_79532];
                    
                    // futhark/microgpt.fut:235:69-100
                    
                    double zt_res_79536 = zt_lhs_79534 * zt_rhs_79535;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79537 = r_79533 + zt_res_79536;
                    double r_tmp_85299 = zp_res_79537;
                    
                    r_79533 = r_tmp_85299;
                }
                defunc_0_lifted_lambda_res_79531 = r_79533;
                ((double *) mem_83971)[i_82700] = defunc_0_lifted_lambda_res_79531;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79545;
            double r_79547 = 0.0;
            
            for (int64_t i_79546 = 0; i_79546 < (int64_t) 16; i_79546++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79548 = ((double *) mem_83930)[i_82706 * (int64_t) 16 + i_79546];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79549 = r_79547 + lifted_lambda_res_79548;
                double r_tmp_85300 = zp_res_79549;
                
                r_79547 = r_tmp_85300;
            }
            defunc_0_lifted_lambda_res_79545 = r_79547;
            // futhark/microgpt.fut:255:34-86
            
            double zs_res_79550 = defunc_0_lifted_lambda_res_79545 / 16.0;
            
            ((double *) mem_83962)[i_82706] = zs_res_79550;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83963, i_82706 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83971, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82717 = 0; i_82717 < (int64_t) 16; i_82717++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82711 = 0; i_82711 < (int64_t) 64; i_82711++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_79574 = ((double *) mem_83963)[i_82717 * (int64_t) 64 + i_82711];
                
                // futhark/microgpt.fut:236:38-62
                
                double max_res_79575 = fmax64(0.0, max_arg0_79574);
                
                ((double *) mem_83994)[i_82711] = max_res_79575;
            }
            // futhark/microgpt.fut:256:41-51
            
            double zp_lhs_79583 = ((double *) mem_83962)[i_82717];
            
            // futhark/microgpt.fut:256:41-79
            
            double zp_res_79584 = 1.0e-5 + zp_lhs_79583;
            
            // futhark/microgpt.fut:256:33-79
            
            double sqrt_res_79585 = futrts_sqrt64(zp_res_79584);
            
            ((double *) mem_83985)[i_82717] = sqrt_res_79585;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83986, i_82717 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83994, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82726 = 0; i_82726 < (int64_t) 16; i_82726++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82722 = 0; i_82722 < (int64_t) 16; i_82722++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77213;
                double r_77215 = 0.0;
                
                for (int64_t i_77214 = 0; i_77214 < (int64_t) 64; i_77214++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77216 = ((double *) mem_param_83376.mem)[i_82722 * (int64_t) 64 + i_77214];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77217 = ((double *) mem_83986)[i_82726 * (int64_t) 64 + i_77214];
                    
                    // futhark/microgpt.fut:237:69-102
                    
                    double zt_res_77218 = zt_lhs_77216 * zt_rhs_77217;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77219 = r_77215 + zt_res_77218;
                    double r_tmp_85306 = zp_res_77219;
                    
                    r_77215 = r_tmp_85306;
                }
                defunc_0_lifted_lambda_res_77213 = r_77215;
                ((double *) mem_84013)[i_82722] = defunc_0_lifted_lambda_res_77213;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84008, i_82726 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84013, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82734 = 0; i_82734 < (int64_t) 16; i_82734++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82730 = 0; i_82730 < (int64_t) 16; i_82730++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77234 = ((double *) mem_84008)[i_82734 * (int64_t) 16 + i_82730];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77235 = ((double *) mem_83914)[i_82734 * (int64_t) 16 + i_82730];
                
                // futhark/microgpt.fut:238:38-69
                
                double zp_res_77236 = zp_lhs_77234 + zp_rhs_77235;
                
                ((double *) mem_84029)[i_82730] = zp_res_77236;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84024, i_82734 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84029, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82742 = 0; i_82742 < (int64_t) 16; i_82742++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82738 = 0; i_82738 < (int64_t) 27; i_82738++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77251;
                double r_77253 = 0.0;
                
                for (int64_t i_77252 = 0; i_77252 < (int64_t) 16; i_77252++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77254 = ((double *) mem_param_83408.mem)[i_82738 * (int64_t) 16 + i_77252];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77255 = ((double *) mem_84024)[i_82742 * (int64_t) 16 + i_77252];
                    
                    // futhark/microgpt.fut:239:69-101
                    
                    double zt_res_77256 = zt_lhs_77254 * zt_rhs_77255;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77257 = r_77253 + zt_res_77256;
                    double r_tmp_85311 = zp_res_77257;
                    
                    r_77253 = r_tmp_85311;
                }
                defunc_0_lifted_lambda_res_77251 = r_77253;
                ((double *) mem_84045)[i_82738] = defunc_0_lifted_lambda_res_77251;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84040, i_82742 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84045, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82772 = 0; i_82772 < (int64_t) 16; i_82772++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_82347;
            double defunc_0_reduce_res_82348;
            double redout_82744;
            double redout_82745;
            
            redout_82744 = -INFINITY;
            redout_82745 = -INFINITY;
            for (int64_t i_82746 = 0; i_82746 < (int64_t) 27; i_82746++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81167 = ((double *) mem_84040)[i_82772 * (int64_t) 27 + i_82746];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79615 = fmax64(lifted_lambda_res_81167, redout_82744);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79667 = fmax64(lifted_lambda_res_81167, redout_82745);
                double redout_tmp_85314 = max_res_79615;
                double redout_tmp_85315 = max_res_79667;
                
                redout_82744 = redout_tmp_85314;
                redout_82745 = redout_tmp_85315;
            }
            defunc_0_reduce_res_82347 = redout_82744;
            defunc_0_reduce_res_82348 = redout_82745;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79616 = -defunc_0_reduce_res_82347;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79668 = -defunc_0_reduce_res_82348;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82751 = 0; i_82751 < (int64_t) 27; i_82751++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_81206 = ((double *) mem_84040)[i_82772 * (int64_t) 27 + i_82751];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81207 = neg_res_79616 + lifted_lambda_res_81206;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81208 = futrts_exp64(zp_res_81207);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81216 = neg_res_79668 + lifted_lambda_res_81206;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81217 = futrts_exp64(zp_res_81216);
                
                ((double *) mem_84066)[i_82751] = exp_res_81217;
                ((double *) mem_84067)[i_82751] = exp_res_81208;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79627;
            double r_79629 = 0.0;
            
            for (int64_t i_79628 = 0; i_79628 < (int64_t) 27; i_79628++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79630 = ((double *) mem_84067)[i_79628];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79631 = r_79629 + lifted_lambda_res_79630;
                double r_tmp_85318 = zp_res_79631;
                
                r_79629 = r_tmp_85318;
            }
            defunc_0_lifted_lambda_res_79627 = r_79629;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79679;
            double r_79681 = 0.0;
            
            for (int64_t i_79680 = 0; i_79680 < (int64_t) 27; i_79680++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79682 = ((double *) mem_84066)[i_79680];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79683 = r_79681 + lifted_lambda_res_79682;
                double r_tmp_85319 = zp_res_79683;
                
                r_79681 = r_tmp_85319;
            }
            defunc_0_lifted_lambda_res_79679 = r_79681;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82758 = 0; i_82758 < (int64_t) 27; i_82758++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81235 = ((double *) mem_84067)[i_82758];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81236 = zs_lhs_81235 / defunc_0_lifted_lambda_res_79627;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81243 = ((double *) mem_84066)[i_82758];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81244 = zs_lhs_81243 / defunc_0_lifted_lambda_res_79679;
                
                ((double *) mem_84080)[i_82758] = zs_res_81244;
                ((double *) mem_84081)[i_82758] = zs_res_81236;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82765 = 0; i_82765 < (int64_t) 27; i_82765++) {
                // futhark/microgpt.fut:245:24-34
                
                double lifted_lambda_res_81262 = ((double *) mem_84081)[i_82765];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81269 = ((double *) mem_83513)[i_82772 * (int64_t) 27 + i_82765];
                
                // futhark/microgpt.fut:247:4-14
                
                double zs_rhs_81270 = ((double *) mem_84080)[i_82765];
                
                // futhark/microgpt.fut:246:74-247:14
                
                double zs_res_81271 = 1.0 / zs_rhs_81270;
                
                // futhark/microgpt.fut:246:53-247:14
                
                double zt_res_81272 = zt_lhs_81269 * zs_res_81271;
                
                ((double *) mem_84094)[i_82765] = zt_res_81272;
                ((double *) mem_84095)[i_82765] = lifted_lambda_res_81262;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84056, i_82772 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84094, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84057, i_82772 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84095, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82777 = 0; i_82777 < (int64_t) 16; i_82777++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77391;
            double r_77393 = 0.0;
            
            for (int64_t i_77392 = 0; i_77392 < (int64_t) 27; i_77392++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77394 = ((double *) mem_84056)[i_82777 * (int64_t) 27 + i_77392];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77395 = ((double *) mem_84057)[i_82777 * (int64_t) 27 + i_77392];
                
                // futhark/microgpt.fut:248:53-90
                
                double zt_res_77396 = zt_lhs_77394 * zt_rhs_77395;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77397 = r_77393 + zt_res_77396;
                double r_tmp_85325 = zp_res_77397;
                
                r_77393 = r_tmp_85325;
            }
            defunc_0_lifted_lambda_res_77391 = r_77393;
            ((double *) mem_84116)[i_82777] = defunc_0_lifted_lambda_res_77391;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82785 = 0; i_82785 < (int64_t) 16; i_82785++) {
            // futhark/microgpt.fut:249:103-113
            
            double neg_arg0_77405 = ((double *) mem_84116)[i_82785];
            
            // futhark/microgpt.fut:249:97-113
            
            double neg_res_77406 = -neg_arg0_77405;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82781 = 0; i_82781 < (int64_t) 27; i_82781++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77413 = ((double *) mem_84057)[i_82785 * (int64_t) 27 + i_82781];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77414 = ((double *) mem_84056)[i_82785 * (int64_t) 27 + i_82781];
                
                // futhark/microgpt.fut:249:75-113
                
                double zp_res_77415 = neg_res_77406 + zp_lhs_77414;
                
                // futhark/microgpt.fut:249:53-113
                
                double zt_res_77416 = zt_lhs_77413 * zp_res_77415;
                
                ((double *) mem_84128)[i_82781] = zt_res_77416;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84123, i_82785 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84128, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82793 = 0; i_82793 < (int64_t) 16; i_82793++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82789 = 0; i_82789 < (int64_t) 16; i_82789++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77431;
                double r_77433 = 0.0;
                
                for (int64_t i_77432 = 0; i_77432 < (int64_t) 27; i_77432++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77434 = ((double *) mem_param_83408.mem)[i_77432 * (int64_t) 16 + i_82789];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77435 = ((double *) mem_84123)[i_82793 * (int64_t) 27 + i_77432];
                    
                    // futhark/microgpt.fut:250:73-110
                    
                    double zt_res_77436 = zt_lhs_77434 * zt_rhs_77435;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77437 = r_77433 + zt_res_77436;
                    double r_tmp_85330 = zp_res_77437;
                    
                    r_77433 = r_tmp_85330;
                }
                defunc_0_lifted_lambda_res_77431 = r_77433;
                ((double *) mem_84144)[i_82789] = defunc_0_lifted_lambda_res_77431;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84139, i_82793 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84144, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82806 = 0; i_82806 < (int64_t) 16; i_82806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82799 = 0; i_82799 < (int64_t) 64; i_82799++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81300;
                double r_81302 = 0.0;
                
                for (int64_t i_81301 = 0; i_81301 < (int64_t) 16; i_81301++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81303 = ((double *) mem_param_83376.mem)[i_81301 * (int64_t) 64 + i_82799];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81304 = ((double *) mem_84139)[i_82806 * (int64_t) 16 + i_81301];
                    
                    // futhark/microgpt.fut:251:73-111
                    
                    double zt_res_81305 = zt_lhs_81303 * zt_rhs_81304;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81306 = r_81302 + zt_res_81305;
                    double r_tmp_85335 = zp_res_81306;
                    
                    r_81302 = r_tmp_85335;
                }
                defunc_0_lifted_lambda_res_81300 = r_81302;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81313;
                double r_81315 = 0.0;
                
                for (int64_t i_81314 = 0; i_81314 < (int64_t) 16; i_81314++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81316 = ((double *) mem_84139)[i_81314 * (int64_t) 16 + i_82806];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81317 = ((double *) mem_83986)[i_81314 * (int64_t) 64 + i_82799];
                    
                    // futhark/microgpt.fut:301:75-111
                    
                    double zt_res_81318 = zt_lhs_81316 * zt_rhs_81317;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81319 = r_81315 + zt_res_81318;
                    double r_tmp_85336 = zp_res_81319;
                    
                    r_81315 = r_tmp_85336;
                }
                defunc_0_lifted_lambda_res_81313 = r_81315;
                ((double *) mem_84165)[i_82799] = defunc_0_lifted_lambda_res_81313;
                ((double *) mem_84166)[i_82799] = defunc_0_lifted_lambda_res_81300;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84155, i_82806 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84165, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84156, i_82806 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84166, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82815 = 0; i_82815 < (int64_t) 16; i_82815++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82811 = 0; i_82811 < (int64_t) 64; i_82811++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_77473 = ((double *) mem_83963)[i_82815 * (int64_t) 64 + i_82811];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_77474 = fmax64(0.0, indicatorp_arg0_77473);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_77475 = fsignum64(max_res_77474);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77476 = ((double *) mem_84156)[i_82815 * (int64_t) 64 + i_82811];
                
                // futhark/microgpt.fut:252:42-90
                
                double zt_res_77477 = sgn_res_77475 * zt_rhs_77476;
                
                ((double *) mem_84192)[i_82811] = zt_res_77477;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84187, i_82815 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82823 = 0; i_82823 < (int64_t) 16; i_82823++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82819 = 0; i_82819 < (int64_t) 16; i_82819++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77492;
                double r_77494 = 0.0;
                
                for (int64_t i_77493 = 0; i_77493 < (int64_t) 64; i_77493++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77495 = ((double *) mem_param_83400.mem)[i_77493 * (int64_t) 16 + i_82819];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77496 = ((double *) mem_84187)[i_82823 * (int64_t) 64 + i_77493];
                    
                    // futhark/microgpt.fut:253:73-109
                    
                    double zt_res_77497 = zt_lhs_77495 * zt_rhs_77496;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77498 = r_77494 + zt_res_77497;
                    double r_tmp_85341 = zp_res_77498;
                    
                    r_77494 = r_tmp_85341;
                }
                defunc_0_lifted_lambda_res_77492 = r_77494;
                ((double *) mem_84208)[i_82819] = defunc_0_lifted_lambda_res_77492;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84203, i_82823 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82827 = 0; i_82827 < (int64_t) 16; i_82827++) {
            // futhark/microgpt.fut:257:49-59
            
            double zs_rhs_77546 = ((double *) mem_83985)[i_82827];
            
            // futhark/microgpt.fut:257:41-59
            
            double zs_res_77547 = 1.0 / zs_rhs_77546;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77548;
            double r_77550 = 0.0;
            
            for (int64_t i_77549 = 0; i_77549 < (int64_t) 16; i_77549++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77551 = ((double *) mem_83914)[i_82827 * (int64_t) 16 + i_77549];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77552 = ((double *) mem_84203)[i_82827 * (int64_t) 16 + i_77549];
                
                // futhark/microgpt.fut:257:87-123
                
                double zt_res_77553 = zt_lhs_77551 * zt_rhs_77552;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77554 = r_77550 + zt_res_77553;
                double r_tmp_85343 = zp_res_77554;
                
                r_77550 = r_tmp_85343;
            }
            defunc_0_lifted_lambda_res_77548 = r_77550;
            // futhark/microgpt.fut:257:67-150
            
            double zt_res_77555 = zs_res_77547 * defunc_0_lifted_lambda_res_77548;
            
            // futhark/microgpt.fut:257:45-150
            
            double zt_res_77556 = zs_res_77547 * zt_res_77555;
            
            // futhark/microgpt.fut:257:33-150
            
            double neg_res_77557 = -zt_res_77556;
            
            ((double *) mem_84219)[i_82827] = neg_res_77557;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82831 = 0; i_82831 < (int64_t) 16; i_82831++) {
            // futhark/microgpt.fut:258:33-43
            
            double zt_lhs_77565 = ((double *) mem_84219)[i_82831];
            
            // futhark/microgpt.fut:258:85-95
            
            double zp_lhs_77566 = ((double *) mem_83962)[i_82831];
            
            // futhark/microgpt.fut:258:85-123
            
            double zp_res_77567 = 1.0e-5 + zp_lhs_77566;
            
            // futhark/microgpt.fut:258:77-123
            
            double sqrt_res_77568 = futrts_sqrt64(zp_res_77567);
            
            // futhark/microgpt.fut:258:63-125
            
            double zt_res_77569 = 2.0 * sqrt_res_77568;
            
            // futhark/microgpt.fut:258:49-125
            
            double zs_res_77570 = 1.0 / zt_res_77569;
            
            // futhark/microgpt.fut:258:33-125
            
            double zt_res_77571 = zt_lhs_77565 * zs_res_77570;
            
            ((double *) mem_84226)[i_82831] = zt_res_77571;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82835 = 0; i_82835 < (int64_t) 16; i_82835++) {
            // futhark/microgpt.fut:259:53-63
            
            double zs_lhs_77579 = ((double *) mem_84226)[i_82835];
            
            // futhark/microgpt.fut:259:53-78
            
            double zs_res_77580 = zs_lhs_77579 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85346 = 0; nest_i_85346 < (int64_t) 16; nest_i_85346++) {
                ((double *) mem_84233)[i_82835 * (int64_t) 16 + nest_i_85346] = zs_res_77580;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82843 = 0; i_82843 < (int64_t) 16; i_82843++) {
            // futhark/microgpt.fut:260:107-117
            
            double zs_rhs_77589 = ((double *) mem_83985)[i_82843];
            
            // futhark/microgpt.fut:260:99-117
            
            double zs_res_77590 = 1.0 / zs_rhs_77589;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82839 = 0; i_82839 < (int64_t) 16; i_82839++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77597 = ((double *) mem_84139)[i_82843 * (int64_t) 16 + i_82839];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77598 = ((double *) mem_84203)[i_82843 * (int64_t) 16 + i_82839];
                
                // futhark/microgpt.fut:260:77-117
                
                double zt_res_77599 = zs_res_77590 * zt_lhs_77598;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77600 = ((double *) mem_83914)[i_82843 * (int64_t) 16 + i_82839];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77601 = ((double *) mem_84233)[i_82843 * (int64_t) 16 + i_82839];
                
                // futhark/microgpt.fut:260:125-161
                
                double zt_res_77602 = zt_lhs_77600 * zt_rhs_77601;
                
                // futhark/microgpt.fut:260:94-161
                
                double zp_res_77603 = zt_res_77599 + zt_res_77602;
                
                // futhark/microgpt.fut:260:120-205
                
                double zp_res_77604 = zt_res_77602 + zp_res_77603;
                
                // futhark/microgpt.fut:260:53-205
                
                double zp_res_77605 = zp_lhs_77597 + zp_res_77604;
                
                ((double *) mem_84248)[i_82839] = zp_res_77605;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84243, i_82843 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84248, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82856 = 0; i_82856 < (int64_t) 16; i_82856++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82849 = 0; i_82849 < (int64_t) 16; i_82849++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81342;
                double r_81344 = 0.0;
                
                for (int64_t i_81343 = 0; i_81343 < (int64_t) 16; i_81343++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81345 = ((double *) mem_param_83384.mem)[i_81343 * (int64_t) 16 + i_82849];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81346 = ((double *) mem_84243)[i_82856 * (int64_t) 16 + i_81343];
                    
                    // futhark/microgpt.fut:261:73-110
                    
                    double zt_res_81347 = zt_lhs_81345 * zt_rhs_81346;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81348 = r_81344 + zt_res_81347;
                    double r_tmp_85353 = zp_res_81348;
                    
                    r_81344 = r_tmp_85353;
                }
                defunc_0_lifted_lambda_res_81342 = r_81344;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81355;
                double r_81357 = 0.0;
                
                for (int64_t i_81356 = 0; i_81356 < (int64_t) 16; i_81356++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81358 = ((double *) mem_84243)[i_81356 * (int64_t) 16 + i_82856];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81359 = ((double *) mem_83876)[i_81356 * (int64_t) 16 + i_82849];
                    
                    // futhark/microgpt.fut:299:74-110
                    
                    double zt_res_81360 = zt_lhs_81358 * zt_rhs_81359;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81361 = r_81357 + zt_res_81360;
                    double r_tmp_85354 = zp_res_81361;
                    
                    r_81357 = r_tmp_85354;
                }
                defunc_0_lifted_lambda_res_81355 = r_81357;
                ((double *) mem_84269)[i_82849] = defunc_0_lifted_lambda_res_81355;
                ((double *) mem_84270)[i_82849] = defunc_0_lifted_lambda_res_81342;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84259, i_82856 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84269, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84260, i_82856 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84270, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82878 = 0; i_82878 < (int64_t) 4; i_82878++) {
            // futhark/microgpt.fut:262:88-91
            
            int64_t zp_lhs_79819 = mul64((int64_t) 4, i_82878);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82871 = 0; i_82871 < (int64_t) 16; i_82871++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82861 = 0; i_82861 < (int64_t) 4; i_82861++) {
                    // futhark/microgpt.fut:262:93-99
                    
                    int64_t tmp_81383 = add64(zp_lhs_79819, i_82861);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool x_81384 = sle64((int64_t) 0, tmp_81383);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool y_81385 = slt64(tmp_81383, (int64_t) 16);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool bounds_check_81386 = x_81384 && y_81385;
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool index_certs_81387;
                    
                    if (!bounds_check_81386) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81383, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:262:70-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:262:52-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:262:32-104\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:262:13-106\n   #10 futhark/microgpt.fut:373:5-76\n   #11 futhark/microgpt.fut:420:33-426:28\n   #12 futhark/microgpt.fut:454:11-54\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81388 = ((double *) mem_84260)[i_82871 * (int64_t) 16 + tmp_81383];
                    
                    ((double *) mem_84313)[i_82861] = lifted_lambda_res_81388;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82865 = 0; i_82865 < (int64_t) 16; i_82865++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_81402 = ((double *) mem_83759)[i_82878 * (int64_t) 256 + i_82871 * (int64_t) 16 + i_82865];
                    
                    // futhark/microgpt.fut:264:61-97
                    
                    double zs_res_81403 = zs_lhs_81402 / 2.0;
                    double zp_rhs_81404 = ((double *) masks_mem_83371.mem)[seq_76718 * ((int64_t) 256 * batchsizze_62012) + seq_76723 * (int64_t) 256 + i_82871 * (int64_t) 16 + i_82865];
                    
                    // futhark/microgpt.fut:264:84-119
                    
                    double zp_res_81405 = zs_res_81403 + zp_rhs_81404;
                    
                    ((double *) mem_84320)[i_82865] = zp_res_81405;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84303, i_82871 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84304, i_82871 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84313, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84291, i_82878 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84303, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84292, i_82878 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84304, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82909 = 0; i_82909 < (int64_t) 4; i_82909++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82902 = 0; i_82902 < (int64_t) 16; i_82902++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_82368;
                double redout_82882 = -INFINITY;
                
                for (int64_t i_82884 = 0; i_82884 < (int64_t) 16; i_82884++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81531 = ((double *) mem_84291)[i_82909 * (int64_t) 256 + i_82902 * (int64_t) 16 + i_82884];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81542;
                    double r_81544 = 0.0;
                    
                    for (int64_t i_81543 = 0; i_81543 < (int64_t) 4; i_81543++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81545 = ((double *) mem_84292)[i_82909 * (int64_t) 64 + i_82902 * (int64_t) 4 + i_81543];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81546 = ((double *) mem_83678)[i_82909 * (int64_t) 64 + i_82884 * (int64_t) 4 + i_81543];
                        
                        // futhark/microgpt.fut:267:91-139
                        
                        double zt_res_81547 = zt_lhs_81545 * zt_rhs_81546;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81548 = r_81544 + zt_res_81547;
                        double r_tmp_85367 = zp_res_81548;
                        
                        r_81544 = r_tmp_85367;
                    }
                    defunc_0_lifted_lambda_res_81542 = r_81544;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_81442 = fmax64(lifted_lambda_res_81531, redout_82882);
                    
                    ((double *) mem_84367)[i_82884] = defunc_0_lifted_lambda_res_81542;
                    
                    double redout_tmp_85365 = max_res_81442;
                    
                    redout_82882 = redout_tmp_85365;
                }
                defunc_0_reduce_res_82368 = redout_82882;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_81443 = -defunc_0_reduce_res_82368;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82888 = 0; i_82888 < (int64_t) 16; i_82888++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_81450 = ((double *) mem_84291)[i_82909 * (int64_t) 256 + i_82902 * (int64_t) 16 + i_82888];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_81451 = neg_res_81443 + lifted_lambda_res_81450;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_81452 = futrts_exp64(zp_res_81451);
                    
                    ((double *) mem_84374)[i_82888] = exp_res_81452;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81454;
                double r_81456 = 0.0;
                
                for (int64_t i_81455 = 0; i_81455 < (int64_t) 16; i_81455++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_81457 = ((double *) mem_84374)[i_81455];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81458 = r_81456 + lifted_lambda_res_81457;
                    double r_tmp_85369 = zp_res_81458;
                    
                    r_81456 = r_tmp_85369;
                }
                defunc_0_lifted_lambda_res_81454 = r_81456;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82892 = 0; i_82892 < (int64_t) 16; i_82892++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_81465 = ((double *) mem_84374)[i_82892];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_81466 = zs_lhs_81465 / defunc_0_lifted_lambda_res_81454;
                    
                    ((double *) mem_84381)[i_82892] = zs_res_81466;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82896 = 0; i_82896 < (int64_t) 16; i_82896++) {
                    // futhark/microgpt.fut:266:24-34
                    
                    double lifted_lambda_res_81474 = ((double *) mem_84381)[i_82896];
                    
                    ((double *) mem_84388)[i_82896] = lifted_lambda_res_81474;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84357, i_82902 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84367, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84358, i_82902 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84388, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84345, i_82909 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84357, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84346, i_82909 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84358, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82931 = 0; i_82931 < (int64_t) 4; i_82931++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82924 = 0; i_82924 < (int64_t) 16; i_82924++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82914 = 0; i_82914 < (int64_t) 16; i_82914++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81584 = ((double *) mem_84345)[i_82931 * (int64_t) 256 + i_82924 * (int64_t) 16 + i_82914];
                    
                    ((double *) mem_84435)[i_82914] = lifted_lambda_res_81584;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82918 = 0; i_82918 < (int64_t) 4; i_82918++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81598;
                    double r_81600 = 0.0;
                    
                    for (int64_t i_81599 = 0; i_81599 < (int64_t) 16; i_81599++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81601 = ((double *) mem_84346)[i_82931 * (int64_t) 256 + i_81599 * (int64_t) 16 + i_82924];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81602 = ((double *) mem_84292)[i_82931 * (int64_t) 64 + i_81599 * (int64_t) 4 + i_82918];
                        
                        // futhark/microgpt.fut:272:91-140
                        
                        double zt_res_81603 = zt_lhs_81601 * zt_rhs_81602;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81604 = r_81600 + zt_res_81603;
                        double r_tmp_85378 = zp_res_81604;
                        
                        r_81600 = r_tmp_85378;
                    }
                    defunc_0_lifted_lambda_res_81598 = r_81600;
                    ((double *) mem_84442)[i_82918] = defunc_0_lifted_lambda_res_81598;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84425, i_82924 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84442, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84426, i_82924 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84435, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84413, i_82931 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84425, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84414, i_82931 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84426, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82940 = 0; i_82940 < (int64_t) 4; i_82940++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82936 = 0; i_82936 < (int64_t) 16; i_82936++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77824;
                double r_77826 = 0.0;
                
                for (int64_t i_77825 = 0; i_77825 < (int64_t) 16; i_77825++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77827 = ((double *) mem_84414)[i_82940 * (int64_t) 256 + i_82936 * (int64_t) 16 + i_77825];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77828 = ((double *) mem_84346)[i_82940 * (int64_t) 256 + i_82936 * (int64_t) 16 + i_77825];
                    
                    // futhark/microgpt.fut:269:72-121
                    
                    double zt_res_77829 = zt_lhs_77827 * zt_rhs_77828;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77830 = r_77826 + zt_res_77829;
                    double r_tmp_85381 = zp_res_77830;
                    
                    r_77826 = r_tmp_85381;
                }
                defunc_0_lifted_lambda_res_77824 = r_77826;
                ((double *) mem_84472)[i_82936] = defunc_0_lifted_lambda_res_77824;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84467, i_82940 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84472, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82952 = 0; i_82952 < (int64_t) 4; i_82952++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82948 = 0; i_82948 < (int64_t) 16; i_82948++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_77845 = ((double *) mem_84467)[i_82952 * (int64_t) 16 + i_82948];
                
                // futhark/microgpt.fut:270:128-150
                
                double neg_res_77846 = -neg_arg0_77845;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82944 = 0; i_82944 < (int64_t) 16; i_82944++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_77853 = ((double *) mem_84346)[i_82952 * (int64_t) 256 + i_82948 * (int64_t) 16 + i_82944];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_77854 = ((double *) mem_84414)[i_82952 * (int64_t) 256 + i_82948 * (int64_t) 16 + i_82944];
                    
                    // futhark/microgpt.fut:270:100-150
                    
                    double zp_res_77855 = neg_res_77846 + zp_lhs_77854;
                    
                    // futhark/microgpt.fut:270:72-150
                    
                    double zt_res_77856 = zt_lhs_77853 * zp_res_77855;
                    
                    ((double *) mem_84494)[i_82944] = zt_res_77856;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84489, i_82948 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84494, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84483, i_82952 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84489, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82964 = 0; i_82964 < (int64_t) 4; i_82964++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82960 = 0; i_82960 < (int64_t) 16; i_82960++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82956 = 0; i_82956 < (int64_t) 16; i_82956++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_77878 = ((double *) mem_84483)[i_82964 * (int64_t) 256 + i_82960 * (int64_t) 16 + i_82956];
                    
                    // futhark/microgpt.fut:271:60-96
                    
                    double zs_res_77879 = zs_lhs_77878 / 2.0;
                    
                    ((double *) mem_84521)[i_82956] = zs_res_77879;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84516, i_82960 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84521, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84510, i_82964 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84516, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82984 = 0; i_82984 < (int64_t) 4; i_82984++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82977 = 0; i_82977 < (int64_t) 16; i_82977++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82970 = 0; i_82970 < (int64_t) 4; i_82970++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81685;
                    double r_81687 = 0.0;
                    
                    for (int64_t i_81686 = 0; i_81686 < (int64_t) 16; i_81686++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81688 = ((double *) mem_83680)[i_82984 * (int64_t) 64 + i_81686 * (int64_t) 4 + i_82970];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81689 = ((double *) mem_84510)[i_82984 * (int64_t) 256 + i_81686 * (int64_t) 16 + i_82977];
                        
                        // futhark/microgpt.fut:273:91-139
                        
                        double zt_res_81690 = zt_lhs_81688 * zt_rhs_81689;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81691 = r_81687 + zt_res_81690;
                        double r_tmp_85394 = zp_res_81691;
                        
                        r_81687 = r_tmp_85394;
                    }
                    defunc_0_lifted_lambda_res_81685 = r_81687;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81698;
                    double r_81700 = 0.0;
                    
                    for (int64_t i_81699 = 0; i_81699 < (int64_t) 16; i_81699++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81701 = ((double *) mem_84510)[i_82984 * (int64_t) 256 + i_82977 * (int64_t) 16 + i_81699];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81702 = ((double *) mem_83679)[i_82984 * (int64_t) 64 + i_81699 * (int64_t) 4 + i_82970];
                        
                        // futhark/microgpt.fut:274:91-139
                        
                        double zt_res_81703 = zt_lhs_81701 * zt_rhs_81702;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81704 = r_81700 + zt_res_81703;
                        double r_tmp_85395 = zp_res_81704;
                        
                        r_81700 = r_tmp_85395;
                    }
                    defunc_0_lifted_lambda_res_81698 = r_81700;
                    ((double *) mem_84559)[i_82970] = defunc_0_lifted_lambda_res_81698;
                    ((double *) mem_84560)[i_82970] = defunc_0_lifted_lambda_res_81685;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84549, i_82977 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84559, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84550, i_82977 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84560, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84537, i_82984 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84549, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84538, i_82984 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84550, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83003 = 0; i_83003 < (int64_t) 16; i_83003++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82993 = 0; i_82993 < (int64_t) 16; i_82993++) {
                // futhark/microgpt.fut:275:63-66
                
                int64_t tmp_81767 = sdiv64(i_82993, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-68
                
                bool x_81768 = sle64((int64_t) 0, tmp_81767);
                
                // futhark/microgpt.fut:275:52-68
                
                bool y_81769 = slt64(tmp_81767, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-68
                
                bool bounds_check_81770 = x_81768 && y_81769;
                
                // futhark/microgpt.fut:275:52-68
                
                bool index_certs_81771;
                
                if (!bounds_check_81770) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81767, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:275:52-68\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:275:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:275:13-89\n   #7  futhark/microgpt.fut:373:5-76\n   #8  futhark/microgpt.fut:420:33-426:28\n   #9  futhark/microgpt.fut:454:11-54\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:275:81-84
                
                int64_t tmp_81772 = smod64(i_82993, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-86
                
                bool x_81773 = sle64((int64_t) 0, tmp_81772);
                
                // futhark/microgpt.fut:275:52-86
                
                bool y_81774 = slt64(tmp_81772, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-86
                
                bool bounds_check_81775 = x_81773 && y_81774;
                
                // futhark/microgpt.fut:275:52-86
                
                bool index_certs_81776;
                
                if (!bounds_check_81775) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81772, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:275:52-86\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:275:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:275:13-89\n   #7  futhark/microgpt.fut:373:5-76\n   #8  futhark/microgpt.fut:420:33-426:28\n   #9  futhark/microgpt.fut:454:11-54\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81777 = ((double *) mem_84413)[tmp_81767 * (int64_t) 64 + i_83003 * (int64_t) 4 + tmp_81772];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81790 = ((double *) mem_84538)[tmp_81767 * (int64_t) 64 + i_83003 * (int64_t) 4 + tmp_81772];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81806 = ((double *) mem_84537)[tmp_81767 * (int64_t) 64 + i_83003 * (int64_t) 4 + tmp_81772];
                
                ((double *) mem_84606)[i_82993] = lifted_lambda_res_81806;
                ((double *) mem_84607)[i_82993] = lifted_lambda_res_81790;
                ((double *) mem_84608)[i_82993] = lifted_lambda_res_81777;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84591, i_83003 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84606, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84592, i_83003 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84607, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84593, i_83003 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84608, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83028 = 0; i_83028 < (int64_t) 16; i_83028++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83015 = 0; i_83015 < (int64_t) 16; i_83015++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81969;
                double r_81971 = 0.0;
                
                for (int64_t i_81970 = 0; i_81970 < (int64_t) 16; i_81970++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81972 = ((double *) mem_param_83404.mem)[i_81970 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81973 = ((double *) mem_84593)[i_83028 * (int64_t) 16 + i_81970];
                    
                    // futhark/microgpt.fut:278:75-112
                    
                    double zt_res_81974 = zt_lhs_81972 * zt_rhs_81973;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81975 = r_81971 + zt_res_81974;
                    double r_tmp_85410 = zp_res_81975;
                    
                    r_81971 = r_tmp_85410;
                }
                defunc_0_lifted_lambda_res_81969 = r_81971;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81976;
                double r_81978 = 0.0;
                
                for (int64_t i_81977 = 0; i_81977 < (int64_t) 16; i_81977++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81979 = ((double *) mem_param_83380.mem)[i_81977 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81980 = ((double *) mem_84592)[i_83028 * (int64_t) 16 + i_81977];
                    
                    // futhark/microgpt.fut:278:141-178
                    
                    double zt_res_81981 = zt_lhs_81979 * zt_rhs_81980;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81982 = r_81978 + zt_res_81981;
                    double r_tmp_85411 = zp_res_81982;
                    
                    r_81978 = r_tmp_85411;
                }
                defunc_0_lifted_lambda_res_81976 = r_81978;
                // futhark/microgpt.fut:278:55-180
                
                double zp_res_81983 = defunc_0_lifted_lambda_res_81969 + defunc_0_lifted_lambda_res_81976;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81984;
                double r_81986 = 0.0;
                
                for (int64_t i_81985 = 0; i_81985 < (int64_t) 16; i_81985++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81987 = ((double *) mem_param_83392.mem)[i_81985 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81988 = ((double *) mem_84591)[i_83028 * (int64_t) 16 + i_81985];
                    
                    // futhark/microgpt.fut:278:208-245
                    
                    double zt_res_81989 = zt_lhs_81987 * zt_rhs_81988;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81990 = r_81986 + zt_res_81989;
                    double r_tmp_85412 = zp_res_81990;
                    
                    r_81986 = r_tmp_85412;
                }
                defunc_0_lifted_lambda_res_81984 = r_81986;
                // futhark/microgpt.fut:278:116-247
                
                double zp_res_81991 = zp_res_81983 + defunc_0_lifted_lambda_res_81984;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81998;
                double r_82000 = 0.0;
                
                for (int64_t i_81999 = 0; i_81999 < (int64_t) 16; i_81999++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82001 = ((double *) mem_84591)[i_81999 * (int64_t) 16 + i_83028];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82002 = ((double *) mem_83579)[i_81999 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:296:74-109
                    
                    double zt_res_82003 = zt_lhs_82001 * zt_rhs_82002;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82004 = r_82000 + zt_res_82003;
                    double r_tmp_85413 = zp_res_82004;
                    
                    r_82000 = r_tmp_85413;
                }
                defunc_0_lifted_lambda_res_81998 = r_82000;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82014;
                double r_82016 = 0.0;
                
                for (int64_t i_82015 = 0; i_82015 < (int64_t) 16; i_82015++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82017 = ((double *) mem_84592)[i_82015 * (int64_t) 16 + i_83028];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82018 = ((double *) mem_83579)[i_82015 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:297:74-109
                    
                    double zt_res_82019 = zt_lhs_82017 * zt_rhs_82018;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82020 = r_82016 + zt_res_82019;
                    double r_tmp_85414 = zp_res_82020;
                    
                    r_82016 = r_tmp_85414;
                }
                defunc_0_lifted_lambda_res_82014 = r_82016;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82032;
                double r_82034 = 0.0;
                
                for (int64_t i_82033 = 0; i_82033 < (int64_t) 16; i_82033++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82035 = ((double *) mem_84593)[i_82033 * (int64_t) 16 + i_83028];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82036 = ((double *) mem_83579)[i_82033 * (int64_t) 16 + i_83015];
                    
                    // futhark/microgpt.fut:298:74-109
                    
                    double zt_res_82037 = zt_lhs_82035 * zt_rhs_82036;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82038 = r_82034 + zt_res_82037;
                    double r_tmp_85415 = zp_res_82038;
                    
                    r_82034 = r_tmp_85415;
                }
                defunc_0_lifted_lambda_res_82032 = r_82034;
                ((double *) mem_84659)[i_83015] = defunc_0_lifted_lambda_res_82032;
                ((double *) mem_84660)[i_83015] = defunc_0_lifted_lambda_res_82014;
                ((double *) mem_84661)[i_83015] = defunc_0_lifted_lambda_res_81998;
                ((double *) mem_84662)[i_83015] = zp_res_81991;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84639, i_83028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84659, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84640, i_83028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84660, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84641, i_83028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84661, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84642, i_83028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84662, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83035 = 0; i_83035 < (int64_t) 16; i_83035++) {
            // futhark/microgpt.fut:282:49-59
            
            double zs_rhs_78112 = ((double *) mem_83875)[i_83035];
            
            // futhark/microgpt.fut:282:41-59
            
            double zs_res_78113 = 1.0 / zs_rhs_78112;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78114;
            double r_78116 = 0.0;
            
            for (int64_t i_78115 = 0; i_78115 < (int64_t) 16; i_78115++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78117 = ((double *) mem_83546)[i_83035 * (int64_t) 16 + i_78115];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78118 = ((double *) mem_84642)[i_83035 * (int64_t) 16 + i_78115];
                
                // futhark/microgpt.fut:282:87-122
                
                double zt_res_78119 = zt_lhs_78117 * zt_rhs_78118;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78120 = r_78116 + zt_res_78119;
                double r_tmp_85417 = zp_res_78120;
                
                r_78116 = r_tmp_85417;
            }
            defunc_0_lifted_lambda_res_78114 = r_78116;
            // futhark/microgpt.fut:282:67-149
            
            double zt_res_78121 = zs_res_78113 * defunc_0_lifted_lambda_res_78114;
            
            // futhark/microgpt.fut:282:45-149
            
            double zt_res_78122 = zs_res_78113 * zt_res_78121;
            
            // futhark/microgpt.fut:282:33-149
            
            double neg_res_78123 = -zt_res_78122;
            
            ((double *) mem_84703)[i_83035] = neg_res_78123;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83039 = 0; i_83039 < (int64_t) 16; i_83039++) {
            // futhark/microgpt.fut:283:33-43
            
            double zt_lhs_78131 = ((double *) mem_84703)[i_83039];
            
            // futhark/microgpt.fut:283:85-95
            
            double zp_lhs_78132 = ((double *) mem_83617)[i_83039];
            
            // futhark/microgpt.fut:283:85-123
            
            double zp_res_78133 = 1.0e-5 + zp_lhs_78132;
            
            // futhark/microgpt.fut:283:77-123
            
            double sqrt_res_78134 = futrts_sqrt64(zp_res_78133);
            
            // futhark/microgpt.fut:283:63-125
            
            double zt_res_78135 = 2.0 * sqrt_res_78134;
            
            // futhark/microgpt.fut:283:49-125
            
            double zs_res_78136 = 1.0 / zt_res_78135;
            
            // futhark/microgpt.fut:283:33-125
            
            double zt_res_78137 = zt_lhs_78131 * zs_res_78136;
            
            ((double *) mem_84710)[i_83039] = zt_res_78137;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83043 = 0; i_83043 < (int64_t) 16; i_83043++) {
            // futhark/microgpt.fut:284:53-63
            
            double zs_lhs_78145 = ((double *) mem_84710)[i_83043];
            
            // futhark/microgpt.fut:284:53-78
            
            double zs_res_78146 = zs_lhs_78145 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85420 = 0; nest_i_85420 < (int64_t) 16; nest_i_85420++) {
                ((double *) mem_84717)[i_83043 * (int64_t) 16 + nest_i_85420] = zs_res_78146;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83051 = 0; i_83051 < (int64_t) 16; i_83051++) {
            // futhark/microgpt.fut:285:107-117
            
            double zs_rhs_78155 = ((double *) mem_83875)[i_83051];
            
            // futhark/microgpt.fut:285:99-117
            
            double zs_res_78156 = 1.0 / zs_rhs_78155;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83047 = 0; i_83047 < (int64_t) 16; i_83047++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78163 = ((double *) mem_84243)[i_83051 * (int64_t) 16 + i_83047];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78164 = ((double *) mem_84642)[i_83051 * (int64_t) 16 + i_83047];
                
                // futhark/microgpt.fut:285:77-117
                
                double zt_res_78165 = zs_res_78156 * zt_lhs_78164;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78166 = ((double *) mem_83546)[i_83051 * (int64_t) 16 + i_83047];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78167 = ((double *) mem_84717)[i_83051 * (int64_t) 16 + i_83047];
                
                // futhark/microgpt.fut:285:125-160
                
                double zt_res_78168 = zt_lhs_78166 * zt_rhs_78167;
                
                // futhark/microgpt.fut:285:94-160
                
                double zp_res_78169 = zt_res_78165 + zt_res_78168;
                
                // futhark/microgpt.fut:285:120-203
                
                double zp_res_78170 = zt_res_78168 + zp_res_78169;
                
                // futhark/microgpt.fut:285:53-203
                
                double zp_res_78171 = zp_lhs_78163 + zp_res_78170;
                
                ((double *) mem_84732)[i_83047] = zp_res_78171;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84727, i_83051 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84732, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83055 = 0; i_83055 < (int64_t) 16; i_83055++) {
            // futhark/microgpt.fut:289:49-59
            
            double zs_rhs_78219 = ((double *) mem_83616)[i_83055];
            
            // futhark/microgpt.fut:289:41-59
            
            double zs_res_78220 = 1.0 / zs_rhs_78219;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78221;
            double r_78223 = 0.0;
            
            for (int64_t i_78222 = 0; i_78222 < (int64_t) 16; i_78222++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78224 = ((double *) mem_83514)[i_83055 * (int64_t) 16 + i_78222];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78225 = ((double *) mem_84727)[i_83055 * (int64_t) 16 + i_78222];
                
                // futhark/microgpt.fut:289:87-122
                
                double zt_res_78226 = zt_lhs_78224 * zt_rhs_78225;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78227 = r_78223 + zt_res_78226;
                double r_tmp_85424 = zp_res_78227;
                
                r_78223 = r_tmp_85424;
            }
            defunc_0_lifted_lambda_res_78221 = r_78223;
            // futhark/microgpt.fut:289:67-149
            
            double zt_res_78228 = zs_res_78220 * defunc_0_lifted_lambda_res_78221;
            
            // futhark/microgpt.fut:289:45-149
            
            double zt_res_78229 = zs_res_78220 * zt_res_78228;
            
            // futhark/microgpt.fut:289:33-149
            
            double neg_res_78230 = -zt_res_78229;
            
            ((double *) mem_84743)[i_83055] = neg_res_78230;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83059 = 0; i_83059 < (int64_t) 16; i_83059++) {
            // futhark/microgpt.fut:290:33-43
            
            double zt_lhs_78238 = ((double *) mem_84743)[i_83059];
            
            // futhark/microgpt.fut:290:85-95
            
            double zp_lhs_78239 = ((double *) mem_83577)[i_83059];
            
            // futhark/microgpt.fut:290:85-123
            
            double zp_res_78240 = 1.0e-5 + zp_lhs_78239;
            
            // futhark/microgpt.fut:290:77-123
            
            double sqrt_res_78241 = futrts_sqrt64(zp_res_78240);
            
            // futhark/microgpt.fut:290:63-125
            
            double zt_res_78242 = 2.0 * sqrt_res_78241;
            
            // futhark/microgpt.fut:290:49-125
            
            double zs_res_78243 = 1.0 / zt_res_78242;
            
            // futhark/microgpt.fut:290:33-125
            
            double zt_res_78244 = zt_lhs_78238 * zs_res_78243;
            
            ((double *) mem_84750)[i_83059] = zt_res_78244;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83063 = 0; i_83063 < (int64_t) 16; i_83063++) {
            // futhark/microgpt.fut:291:53-63
            
            double zs_lhs_78252 = ((double *) mem_84750)[i_83063];
            
            // futhark/microgpt.fut:291:53-78
            
            double zs_res_78253 = zs_lhs_78252 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85427 = 0; nest_i_85427 < (int64_t) 16; nest_i_85427++) {
                ((double *) mem_84757)[i_83063 * (int64_t) 16 + nest_i_85427] = zs_res_78253;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83071 = 0; i_83071 < (int64_t) 16; i_83071++) {
            // futhark/microgpt.fut:292:85-95
            
            double zs_rhs_78262 = ((double *) mem_83616)[i_83071];
            
            // futhark/microgpt.fut:292:77-95
            
            double zs_res_78263 = 1.0 / zs_rhs_78262;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83067 = 0; i_83067 < (int64_t) 16; i_83067++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78270 = ((double *) mem_84727)[i_83071 * (int64_t) 16 + i_83067];
                
                // futhark/microgpt.fut:292:55-95
                
                double zt_res_78271 = zs_res_78263 * zt_lhs_78270;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78272 = ((double *) mem_83514)[i_83071 * (int64_t) 16 + i_83067];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78273 = ((double *) mem_84757)[i_83071 * (int64_t) 16 + i_83067];
                
                // futhark/microgpt.fut:292:103-138
                
                double zt_res_78274 = zt_lhs_78272 * zt_rhs_78273;
                
                // futhark/microgpt.fut:292:72-138
                
                double zp_res_78275 = zt_res_78271 + zt_res_78274;
                
                // futhark/microgpt.fut:292:98-181
                
                double zp_res_78276 = zt_res_78274 + zp_res_78275;
                
                ((double *) mem_84772)[i_83067] = zp_res_78276;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84767, i_83071 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84772, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83084 = 0; i_83084 < (int64_t) 16; i_83084++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83077 = 0; i_83077 < (int64_t) 16; i_83077++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82064 = ((double *) mem_84767)[i_83084 * (int64_t) 16 + i_83077];
                
                ((double *) mem_84793)[i_83077] = lifted_lambda_res_82064;
                ((double *) mem_84794)[i_83077] = lifted_lambda_res_82064;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84783, i_83084 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84793, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84784, i_83084 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84794, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83093 = 0; i_83093 < (int64_t) 64; i_83093++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83089 = 0; i_83089 < (int64_t) 16; i_83089++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_78390;
                double r_78392 = 0.0;
                
                for (int64_t i_78391 = 0; i_78391 < (int64_t) 16; i_78391++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_78393 = ((double *) mem_84187)[i_78391 * (int64_t) 64 + i_83093];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_78394 = ((double *) mem_83931)[i_78391 * (int64_t) 16 + i_83089];
                    
                    // futhark/microgpt.fut:300:73-109
                    
                    double zt_res_78395 = zt_lhs_78393 * zt_rhs_78394;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_78396 = r_78392 + zt_res_78395;
                    double r_tmp_85436 = zp_res_78396;
                    
                    r_78392 = r_tmp_85436;
                }
                defunc_0_lifted_lambda_res_78390 = r_78392;
                ((double *) mem_84820)[i_83089] = defunc_0_lifted_lambda_res_78390;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84815, i_83093 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84820, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83106 = 0; i_83106 < (int64_t) 27; i_83106++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83099 = 0; i_83099 < (int64_t) 16; i_83099++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82092;
                double r_82094 = 0.0;
                
                for (int64_t i_82093 = 0; i_82093 < (int64_t) 16; i_82093++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82095 = ((double *) mem_84123)[i_82093 * (int64_t) 27 + i_83106];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82096 = ((double *) mem_84024)[i_82093 * (int64_t) 16 + i_83099];
                    
                    // futhark/microgpt.fut:302:74-110
                    
                    double zt_res_82097 = zt_lhs_82095 * zt_rhs_82096;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82098 = r_82094 + zt_res_82097;
                    double r_tmp_85441 = zp_res_82098;
                    
                    r_82094 = r_tmp_85441;
                }
                defunc_0_lifted_lambda_res_82092 = r_82094;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82101;
                double r_82103 = 0.0;
                
                for (int64_t i_82102 = 0; i_82102 < (int64_t) 16; i_82102++) {
                    int64_t zeze_lhs_82104 = ((int64_t *) seqs_mem_83372.mem)[seq_76718 * ((int64_t) 16 * batchsizze_62012) + seq_76723 * (int64_t) 16 + i_82102];
                    
                    // futhark/microgpt.fut:375:26-77
                    
                    bool cond_82105 = zeze_lhs_82104 == i_83106;
                    
                    // futhark/microgpt.fut:375:26-77
                    
                    double lifted_lambda_res_82106;
                    
                    if (cond_82105) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_82404 = ((double *) mem_84783)[i_82102 * (int64_t) 16 + i_83099];
                        
                        lifted_lambda_res_82106 = lifted_lambda_res_t_res_82404;
                    } else {
                        lifted_lambda_res_82106 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82112 = r_82103 + lifted_lambda_res_82106;
                    double r_tmp_85442 = zp_res_82112;
                    
                    r_82103 = r_tmp_85442;
                }
                defunc_0_lifted_lambda_res_82101 = r_82103;
                ((double *) mem_84841)[i_83099] = defunc_0_lifted_lambda_res_82101;
                ((double *) mem_84842)[i_83099] = defunc_0_lifted_lambda_res_82092;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84831, i_83106 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84841, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84832, i_83106 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84842, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_78474 = sitofp_i64_f64(step_76690);
        
        // futhark/microgpt.fut:396:46-57
        
        double zm_rhs_78475 = i64_res_78474 / i64_res_76662;
        
        // futhark/microgpt.fut:396:24-57
        
        double zt_rhs_78476 = 1.0 - zm_rhs_78475;
        
        // futhark/microgpt.fut:396:19-57
        
        double lt_r_78477 = 1.0e-2 * zt_rhs_78476;
        
        // futhark/microgpt.fut:398:5-52
        if (memblock_alloc(ctx, &mem_84863, (int64_t) 3456, "mem_84863")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:398:5-52
        // futhark/microgpt.fut:398:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84863.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83396.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:398:5-52
        if (memblock_alloc(ctx, &mem_84865, (int64_t) 3456, "mem_84865")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:398:5-52
        // futhark/microgpt.fut:398:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84865.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83432.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:398:5-52
        if (memblock_alloc(ctx, &mem_84867, (int64_t) 3456, "mem_84867")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:398:5-52
        // futhark/microgpt.fut:398:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84867.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83468.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:398:5-52
        if (memblock_alloc(ctx, &mem_84869, (int64_t) 3456, "mem_84869")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:398:5-52
        // futhark/microgpt.fut:398:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84869.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84831, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:398:5-52
        if (futrts_adam_opt_w_10353(ctx, &ext_mem_84873, &ext_mem_84872, &ext_mem_84871, mem_84863, mem_84865, mem_84867, mem_84869, (int64_t) 27, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84863, "mem_84863") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84865, "mem_84865") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84867, "mem_84867") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84869, "mem_84869") != 0)
            return 1;
        // futhark/microgpt.fut:400:5-52
        if (memblock_alloc(ctx, &mem_84874, (int64_t) 2048, "mem_84874")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:400:5-52
        // futhark/microgpt.fut:400:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84874.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83388.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:400:5-52
        if (memblock_alloc(ctx, &mem_84876, (int64_t) 2048, "mem_84876")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:400:5-52
        // futhark/microgpt.fut:400:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84876.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83424.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:400:5-52
        if (memblock_alloc(ctx, &mem_84878, (int64_t) 2048, "mem_84878")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:400:5-52
        // futhark/microgpt.fut:400:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84878.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83460.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:400:5-52
        if (memblock_alloc(ctx, &mem_84880, (int64_t) 2048, "mem_84880")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:400:5-52
        // futhark/microgpt.fut:400:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84880.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84784, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:400:5-52
        if (futrts_adam_opt_w_10354(ctx, &ext_mem_84884, &ext_mem_84883, &ext_mem_84882, mem_84874, mem_84876, mem_84878, mem_84880, (int64_t) 16, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84874, "mem_84874") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84876, "mem_84876") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84878, "mem_84878") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84880, "mem_84880") != 0)
            return 1;
        // futhark/microgpt.fut:402:5-56
        if (memblock_alloc(ctx, &mem_84885, (int64_t) 2048, "mem_84885")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:402:5-56
        // futhark/microgpt.fut:402:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84885.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83392.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:402:5-56
        if (memblock_alloc(ctx, &mem_84887, (int64_t) 2048, "mem_84887")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:402:5-56
        // futhark/microgpt.fut:402:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84887.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83428.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:402:5-56
        if (memblock_alloc(ctx, &mem_84889, (int64_t) 2048, "mem_84889")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:402:5-56
        // futhark/microgpt.fut:402:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84889.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83464.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:402:5-56
        if (memblock_alloc(ctx, &mem_84891, (int64_t) 2048, "mem_84891")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:402:5-56
        // futhark/microgpt.fut:402:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84891.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84641, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:402:5-56
        if (futrts_adam_opt_w_10354(ctx, &ext_mem_84895, &ext_mem_84894, &ext_mem_84893, mem_84885, mem_84887, mem_84889, mem_84891, (int64_t) 16, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84885, "mem_84885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84887, "mem_84887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84889, "mem_84889") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84891, "mem_84891") != 0)
            return 1;
        // futhark/microgpt.fut:404:5-56
        if (memblock_alloc(ctx, &mem_84896, (int64_t) 2048, "mem_84896")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:404:5-56
        // futhark/microgpt.fut:404:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84896.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83380.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:404:5-56
        if (memblock_alloc(ctx, &mem_84898, (int64_t) 2048, "mem_84898")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:404:5-56
        // futhark/microgpt.fut:404:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83416.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:404:5-56
        if (memblock_alloc(ctx, &mem_84900, (int64_t) 2048, "mem_84900")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:404:5-56
        // futhark/microgpt.fut:404:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84900.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83452.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:404:5-56
        if (memblock_alloc(ctx, &mem_84902, (int64_t) 2048, "mem_84902")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:404:5-56
        // futhark/microgpt.fut:404:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84902.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84640, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:404:5-56
        if (futrts_adam_opt_w_10354(ctx, &ext_mem_84906, &ext_mem_84905, &ext_mem_84904, mem_84896, mem_84898, mem_84900, mem_84902, (int64_t) 16, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84896, "mem_84896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84898, "mem_84898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84900, "mem_84900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84902, "mem_84902") != 0)
            return 1;
        // futhark/microgpt.fut:406:5-56
        if (memblock_alloc(ctx, &mem_84907, (int64_t) 2048, "mem_84907")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:406:5-56
        // futhark/microgpt.fut:406:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84907.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83404.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:406:5-56
        if (memblock_alloc(ctx, &mem_84909, (int64_t) 2048, "mem_84909")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:406:5-56
        // futhark/microgpt.fut:406:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84909.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83440.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:406:5-56
        if (memblock_alloc(ctx, &mem_84911, (int64_t) 2048, "mem_84911")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:406:5-56
        // futhark/microgpt.fut:406:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84911.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83476.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:406:5-56
        if (memblock_alloc(ctx, &mem_84913, (int64_t) 2048, "mem_84913")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:406:5-56
        // futhark/microgpt.fut:406:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84913.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84639, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:406:5-56
        if (futrts_adam_opt_w_10354(ctx, &ext_mem_84917, &ext_mem_84916, &ext_mem_84915, mem_84907, mem_84909, mem_84911, mem_84913, (int64_t) 16, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84907, "mem_84907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84909, "mem_84909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84911, "mem_84911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84913, "mem_84913") != 0)
            return 1;
        // futhark/microgpt.fut:408:5-56
        if (memblock_alloc(ctx, &mem_84918, (int64_t) 2048, "mem_84918")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-56
        // futhark/microgpt.fut:408:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84918.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83384.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:408:5-56
        if (memblock_alloc(ctx, &mem_84920, (int64_t) 2048, "mem_84920")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-56
        // futhark/microgpt.fut:408:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84920.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83420.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:408:5-56
        if (memblock_alloc(ctx, &mem_84922, (int64_t) 2048, "mem_84922")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-56
        // futhark/microgpt.fut:408:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84922.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83456.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:408:5-56
        if (memblock_alloc(ctx, &mem_84924, (int64_t) 2048, "mem_84924")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:408:5-56
        // futhark/microgpt.fut:408:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84924.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84259, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:408:5-56
        if (futrts_adam_opt_w_10354(ctx, &ext_mem_84928, &ext_mem_84927, &ext_mem_84926, mem_84918, mem_84920, mem_84922, mem_84924, (int64_t) 16, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84918, "mem_84918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84920, "mem_84920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84922, "mem_84922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84924, "mem_84924") != 0)
            return 1;
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_84929, (int64_t) 8192, "mem_84929")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84929.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83400.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_84931, (int64_t) 8192, "mem_84931")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84931.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83436.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_84933, (int64_t) 8192, "mem_84933")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84933.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83472.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (memblock_alloc(ctx, &mem_84935, (int64_t) 8192, "mem_84935")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:410:5-52
        // futhark/microgpt.fut:410:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84935.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84815, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:410:5-52
        if (futrts_adam_opt_w_10353(ctx, &ext_mem_84939, &ext_mem_84938, &ext_mem_84937, mem_84929, mem_84931, mem_84933, mem_84935, (int64_t) 64, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84929, "mem_84929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84931, "mem_84931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84933, "mem_84933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84935, "mem_84935") != 0)
            return 1;
        // futhark/microgpt.fut:412:5-60
        if (memblock_alloc(ctx, &mem_84940, (int64_t) 8192, "mem_84940")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-60
        // futhark/microgpt.fut:412:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84940.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83376.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:412:5-60
        if (memblock_alloc(ctx, &mem_84942, (int64_t) 8192, "mem_84942")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-60
        // futhark/microgpt.fut:412:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84942.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83412.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:412:5-60
        if (memblock_alloc(ctx, &mem_84944, (int64_t) 8192, "mem_84944")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-60
        // futhark/microgpt.fut:412:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84944.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83448.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:412:5-60
        if (memblock_alloc(ctx, &mem_84946, (int64_t) 8192, "mem_84946")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-60
        // futhark/microgpt.fut:412:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84946.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_84155, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:412:5-60
        if (futrts_adam_opt_w_10353(ctx, &ext_mem_84950, &ext_mem_84949, &ext_mem_84948, mem_84940, mem_84942, mem_84944, mem_84946, (int64_t) 16, (int64_t) 64, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84940, "mem_84940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84942, "mem_84942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84944, "mem_84944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84946, "mem_84946") != 0)
            return 1;
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_84951, (int64_t) 3456, "mem_84951")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84951.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83408.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_84953, (int64_t) 3456, "mem_84953")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84953.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83444.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_84955, (int64_t) 3456, "mem_84955")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84955.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83480.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (memblock_alloc(ctx, &mem_84957, (int64_t) 3456, "mem_84957")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-56
        // futhark/microgpt.fut:414:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84957.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84832, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:414:5-56
        if (futrts_adam_opt_w_10353(ctx, &ext_mem_84961, &ext_mem_84960, &ext_mem_84959, mem_84951, mem_84953, mem_84955, mem_84957, (int64_t) 27, (int64_t) 16, step_76690, lt_r_78477) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84951, "mem_84951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84953, "mem_84953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84955, "mem_84955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84957, "mem_84957") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85169, &ext_mem_84950, "ext_mem_84950") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85170, &ext_mem_84906, "ext_mem_84906") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85171, &ext_mem_84928, "ext_mem_84928") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85172, &ext_mem_84884, "ext_mem_84884") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85173, &ext_mem_84895, "ext_mem_84895") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85174, &ext_mem_84873, "ext_mem_84873") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85175, &ext_mem_84939, "ext_mem_84939") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85176, &ext_mem_84917, "ext_mem_84917") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85177, &ext_mem_84961, "ext_mem_84961") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85178, &ext_mem_84949, "ext_mem_84949") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85179, &ext_mem_84905, "ext_mem_84905") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85180, &ext_mem_84927, "ext_mem_84927") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85181, &ext_mem_84883, "ext_mem_84883") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85182, &ext_mem_84894, "ext_mem_84894") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85183, &ext_mem_84872, "ext_mem_84872") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85184, &ext_mem_84938, "ext_mem_84938") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85185, &ext_mem_84916, "ext_mem_84916") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85186, &ext_mem_84960, "ext_mem_84960") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85187, &ext_mem_84948, "ext_mem_84948") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85188, &ext_mem_84904, "ext_mem_84904") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85189, &ext_mem_84926, "ext_mem_84926") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85190, &ext_mem_84882, "ext_mem_84882") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85191, &ext_mem_84893, "ext_mem_84893") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85192, &ext_mem_84871, "ext_mem_84871") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85193, &ext_mem_84937, "ext_mem_84937") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85194, &ext_mem_84915, "ext_mem_84915") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85195, &ext_mem_84959, "ext_mem_84959") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83376, &mem_param_tmp_85169, "mem_param_tmp_85169") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83380, &mem_param_tmp_85170, "mem_param_tmp_85170") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83384, &mem_param_tmp_85171, "mem_param_tmp_85171") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83388, &mem_param_tmp_85172, "mem_param_tmp_85172") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83392, &mem_param_tmp_85173, "mem_param_tmp_85173") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83396, &mem_param_tmp_85174, "mem_param_tmp_85174") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83400, &mem_param_tmp_85175, "mem_param_tmp_85175") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83404, &mem_param_tmp_85176, "mem_param_tmp_85176") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83408, &mem_param_tmp_85177, "mem_param_tmp_85177") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83412, &mem_param_tmp_85178, "mem_param_tmp_85178") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83416, &mem_param_tmp_85179, "mem_param_tmp_85179") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83420, &mem_param_tmp_85180, "mem_param_tmp_85180") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83424, &mem_param_tmp_85181, "mem_param_tmp_85181") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83428, &mem_param_tmp_85182, "mem_param_tmp_85182") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83432, &mem_param_tmp_85183, "mem_param_tmp_85183") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83436, &mem_param_tmp_85184, "mem_param_tmp_85184") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83440, &mem_param_tmp_85185, "mem_param_tmp_85185") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83444, &mem_param_tmp_85186, "mem_param_tmp_85186") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83448, &mem_param_tmp_85187, "mem_param_tmp_85187") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83452, &mem_param_tmp_85188, "mem_param_tmp_85188") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83456, &mem_param_tmp_85189, "mem_param_tmp_85189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83460, &mem_param_tmp_85190, "mem_param_tmp_85190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83464, &mem_param_tmp_85191, "mem_param_tmp_85191") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83468, &mem_param_tmp_85192, "mem_param_tmp_85192") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83472, &mem_param_tmp_85193, "mem_param_tmp_85193") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83476, &mem_param_tmp_85194, "mem_param_tmp_85194") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83480, &mem_param_tmp_85195, "mem_param_tmp_85195") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_85069, &mem_param_83376, "mem_param_83376") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85068, &mem_param_83380, "mem_param_83380") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85067, &mem_param_83384, "mem_param_83384") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85066, &mem_param_83388, "mem_param_83388") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85065, &mem_param_83392, "mem_param_83392") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85064, &mem_param_83396, "mem_param_83396") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85063, &mem_param_83400, "mem_param_83400") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85062, &mem_param_83404, "mem_param_83404") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85061, &mem_param_83408, "mem_param_83408") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85060, &mem_param_83412, "mem_param_83412") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85059, &mem_param_83416, "mem_param_83416") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85058, &mem_param_83420, "mem_param_83420") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85057, &mem_param_83424, "mem_param_83424") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85056, &mem_param_83428, "mem_param_83428") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85055, &mem_param_83432, "mem_param_83432") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85054, &mem_param_83436, "mem_param_83436") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85053, &mem_param_83440, "mem_param_83440") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85052, &mem_param_83444, "mem_param_83444") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85051, &mem_param_83448, "mem_param_83448") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85050, &mem_param_83452, "mem_param_83452") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85049, &mem_param_83456, "mem_param_83456") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85048, &mem_param_83460, "mem_param_83460") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85047, &mem_param_83464, "mem_param_83464") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85046, &mem_param_83468, "mem_param_83468") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85045, &mem_param_83472, "mem_param_83472") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85044, &mem_param_83476, "mem_param_83476") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85043, &mem_param_83480, "mem_param_83480") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85142, &ext_mem_85064, "ext_mem_85064") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &ext_mem_85066, "ext_mem_85066") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &ext_mem_85065, "ext_mem_85065") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85145, &ext_mem_85068, "ext_mem_85068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85146, &ext_mem_85062, "ext_mem_85062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85147, &ext_mem_85067, "ext_mem_85067") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85148, &ext_mem_85063, "ext_mem_85063") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85149, &ext_mem_85069, "ext_mem_85069") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85150, &ext_mem_85061, "ext_mem_85061") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85151, &ext_mem_85055, "ext_mem_85055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85152, &ext_mem_85057, "ext_mem_85057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85153, &ext_mem_85056, "ext_mem_85056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85154, &ext_mem_85059, "ext_mem_85059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85155, &ext_mem_85053, "ext_mem_85053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85156, &ext_mem_85058, "ext_mem_85058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85157, &ext_mem_85054, "ext_mem_85054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85158, &ext_mem_85060, "ext_mem_85060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85159, &ext_mem_85052, "ext_mem_85052") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85160, &ext_mem_85046, "ext_mem_85046") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85161, &ext_mem_85048, "ext_mem_85048") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85162, &ext_mem_85047, "ext_mem_85047") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85163, &ext_mem_85050, "ext_mem_85050") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85164, &ext_mem_85044, "ext_mem_85044") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85165, &ext_mem_85049, "ext_mem_85049") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85166, &ext_mem_85045, "ext_mem_85045") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85167, &ext_mem_85051, "ext_mem_85051") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85168, &ext_mem_85043, "ext_mem_85043") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85535, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85536, &mem_out_85143, "mem_out_85143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85537, &mem_out_85144, "mem_out_85144") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85538, &mem_out_85145, "mem_out_85145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85539, &mem_out_85146, "mem_out_85146") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85540, &mem_out_85147, "mem_out_85147") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85541, &mem_out_85148, "mem_out_85148") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85542, &mem_out_85149, "mem_out_85149") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85543, &mem_out_85150, "mem_out_85150") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85544, &mem_out_85151, "mem_out_85151") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85545, &mem_out_85152, "mem_out_85152") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85546, &mem_out_85153, "mem_out_85153") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85547, &mem_out_85154, "mem_out_85154") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85548, &mem_out_85155, "mem_out_85155") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85549, &mem_out_85156, "mem_out_85156") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85550, &mem_out_85157, "mem_out_85157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85551, &mem_out_85158, "mem_out_85158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85552, &mem_out_85159, "mem_out_85159") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85553, &mem_out_85160, "mem_out_85160") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85554, &mem_out_85161, "mem_out_85161") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85555, &mem_out_85162, "mem_out_85162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85556, &mem_out_85163, "mem_out_85163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85557, &mem_out_85164, "mem_out_85164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85558, &mem_out_85165, "mem_out_85165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85559, &mem_out_85166, "mem_out_85166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85560, &mem_out_85167, "mem_out_85167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85561, &mem_out_85168, "mem_out_85168") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83481);
        free(mem_83482);
        free(mem_83491);
        free(mem_83498);
        free(mem_83513);
        free(mem_83514);
        free(mem_83523);
        free(mem_83530);
        free(mem_83545);
        free(mem_83546);
        free(mem_83555);
        free(mem_83556);
        free(mem_83577);
        free(mem_83578);
        free(mem_83579);
        free(mem_83591);
        free(mem_83592);
        free(mem_83616);
        free(mem_83617);
        free(mem_83618);
        free(mem_83619);
        free(mem_83620);
        free(mem_83639);
        free(mem_83640);
        free(mem_83641);
        free(mem_83678);
        free(mem_83679);
        free(mem_83680);
        free(mem_83696);
        free(mem_83697);
        free(mem_83698);
        free(mem_83711);
        free(mem_83712);
        free(mem_83713);
        free(mem_83759);
        free(mem_83760);
        free(mem_83771);
        free(mem_83772);
        free(mem_83781);
        free(mem_83782);
        free(mem_83803);
        free(mem_83808);
        free(mem_83819);
        free(mem_83824);
        free(mem_83831);
        free(mem_83838);
        free(mem_83849);
        free(mem_83854);
        free(mem_83875);
        free(mem_83876);
        free(mem_83884);
        free(mem_83898);
        free(mem_83903);
        free(mem_83914);
        free(mem_83919);
        free(mem_83930);
        free(mem_83931);
        free(mem_83940);
        free(mem_83941);
        free(mem_83962);
        free(mem_83963);
        free(mem_83971);
        free(mem_83985);
        free(mem_83986);
        free(mem_83994);
        free(mem_84008);
        free(mem_84013);
        free(mem_84024);
        free(mem_84029);
        free(mem_84040);
        free(mem_84045);
        free(mem_84056);
        free(mem_84057);
        free(mem_84066);
        free(mem_84067);
        free(mem_84080);
        free(mem_84081);
        free(mem_84094);
        free(mem_84095);
        free(mem_84116);
        free(mem_84123);
        free(mem_84128);
        free(mem_84139);
        free(mem_84144);
        free(mem_84155);
        free(mem_84156);
        free(mem_84165);
        free(mem_84166);
        free(mem_84187);
        free(mem_84192);
        free(mem_84203);
        free(mem_84208);
        free(mem_84219);
        free(mem_84226);
        free(mem_84233);
        free(mem_84243);
        free(mem_84248);
        free(mem_84259);
        free(mem_84260);
        free(mem_84269);
        free(mem_84270);
        free(mem_84291);
        free(mem_84292);
        free(mem_84303);
        free(mem_84304);
        free(mem_84313);
        free(mem_84320);
        free(mem_84345);
        free(mem_84346);
        free(mem_84357);
        free(mem_84358);
        free(mem_84367);
        free(mem_84374);
        free(mem_84381);
        free(mem_84388);
        free(mem_84413);
        free(mem_84414);
        free(mem_84425);
        free(mem_84426);
        free(mem_84435);
        free(mem_84442);
        free(mem_84467);
        free(mem_84472);
        free(mem_84483);
        free(mem_84489);
        free(mem_84494);
        free(mem_84510);
        free(mem_84516);
        free(mem_84521);
        free(mem_84537);
        free(mem_84538);
        free(mem_84549);
        free(mem_84550);
        free(mem_84559);
        free(mem_84560);
        free(mem_84591);
        free(mem_84592);
        free(mem_84593);
        free(mem_84606);
        free(mem_84607);
        free(mem_84608);
        free(mem_84639);
        free(mem_84640);
        free(mem_84641);
        free(mem_84642);
        free(mem_84659);
        free(mem_84660);
        free(mem_84661);
        free(mem_84662);
        free(mem_84703);
        free(mem_84710);
        free(mem_84717);
        free(mem_84727);
        free(mem_84732);
        free(mem_84743);
        free(mem_84750);
        free(mem_84757);
        free(mem_84767);
        free(mem_84772);
        free(mem_84783);
        free(mem_84784);
        free(mem_84793);
        free(mem_84794);
        free(mem_84815);
        free(mem_84820);
        free(mem_84831);
        free(mem_84832);
        free(mem_84841);
        free(mem_84842);
        if (memblock_unref(ctx, &mem_param_tmp_85195, "mem_param_tmp_85195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85194, "mem_param_tmp_85194") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85193, "mem_param_tmp_85193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85192, "mem_param_tmp_85192") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85191, "mem_param_tmp_85191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85190, "mem_param_tmp_85190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85189, "mem_param_tmp_85189") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85188, "mem_param_tmp_85188") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85187, "mem_param_tmp_85187") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85186, "mem_param_tmp_85186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85185, "mem_param_tmp_85185") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85184, "mem_param_tmp_85184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85183, "mem_param_tmp_85183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85182, "mem_param_tmp_85182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85181, "mem_param_tmp_85181") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85180, "mem_param_tmp_85180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85179, "mem_param_tmp_85179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85178, "mem_param_tmp_85178") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85177, "mem_param_tmp_85177") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85176, "mem_param_tmp_85176") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85175, "mem_param_tmp_85175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85174, "mem_param_tmp_85174") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85173, "mem_param_tmp_85173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85172, "mem_param_tmp_85172") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85171, "mem_param_tmp_85171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85170, "mem_param_tmp_85170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85169, "mem_param_tmp_85169") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84959, "ext_mem_84959") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84960, "ext_mem_84960") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84961, "ext_mem_84961") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84957, "mem_84957") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84955, "mem_84955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84953, "mem_84953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84951, "mem_84951") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84948, "ext_mem_84948") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84949, "ext_mem_84949") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84950, "ext_mem_84950") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84946, "mem_84946") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84944, "mem_84944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84942, "mem_84942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84940, "mem_84940") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84937, "ext_mem_84937") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84938, "ext_mem_84938") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84939, "ext_mem_84939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84935, "mem_84935") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84933, "mem_84933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84931, "mem_84931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84929, "mem_84929") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84926, "ext_mem_84926") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84927, "ext_mem_84927") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84928, "ext_mem_84928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84924, "mem_84924") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84922, "mem_84922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84920, "mem_84920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84918, "mem_84918") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84915, "ext_mem_84915") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84916, "ext_mem_84916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84917, "ext_mem_84917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84913, "mem_84913") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84911, "mem_84911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84909, "mem_84909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84907, "mem_84907") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84904, "ext_mem_84904") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84905, "ext_mem_84905") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84906, "ext_mem_84906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84902, "mem_84902") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84900, "mem_84900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84898, "mem_84898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84896, "mem_84896") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84893, "ext_mem_84893") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84894, "ext_mem_84894") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84895, "ext_mem_84895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84891, "mem_84891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84889, "mem_84889") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84887, "mem_84887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84885, "mem_84885") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84882, "ext_mem_84882") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84883, "ext_mem_84883") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84884, "ext_mem_84884") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84880, "mem_84880") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84878, "mem_84878") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84876, "mem_84876") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84874, "mem_84874") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84871, "ext_mem_84871") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84872, "ext_mem_84872") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84873, "ext_mem_84873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84869, "mem_84869") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84867, "mem_84867") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84865, "mem_84865") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84863, "mem_84863") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83480, "mem_param_83480") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83476, "mem_param_83476") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83472, "mem_param_83472") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83468, "mem_param_83468") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83464, "mem_param_83464") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83460, "mem_param_83460") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83456, "mem_param_83456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83452, "mem_param_83452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83448, "mem_param_83448") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83444, "mem_param_83444") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83440, "mem_param_83440") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83436, "mem_param_83436") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83432, "mem_param_83432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83428, "mem_param_83428") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83424, "mem_param_83424") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83420, "mem_param_83420") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83416, "mem_param_83416") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83412, "mem_param_83412") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83408, "mem_param_83408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83404, "mem_param_83404") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83400, "mem_param_83400") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83396, "mem_param_83396") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83392, "mem_param_83392") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83388, "mem_param_83388") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83384, "mem_param_83384") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83380, "mem_param_83380") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83376, "mem_param_83376") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85043, "ext_mem_85043") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85044, "ext_mem_85044") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85045, "ext_mem_85045") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85046, "ext_mem_85046") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85047, "ext_mem_85047") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85048, "ext_mem_85048") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85049, "ext_mem_85049") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85050, "ext_mem_85050") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85051, "ext_mem_85051") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85052, "ext_mem_85052") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85053, "ext_mem_85053") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85054, "ext_mem_85054") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85055, "ext_mem_85055") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85056, "ext_mem_85056") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85057, "ext_mem_85057") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85058, "ext_mem_85058") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85059, "ext_mem_85059") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85060, "ext_mem_85060") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85061, "ext_mem_85061") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85062, "ext_mem_85062") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85063, "ext_mem_85063") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85064, "ext_mem_85064") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85065, "ext_mem_85065") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85066, "ext_mem_85066") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85067, "ext_mem_85067") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85068, "ext_mem_85068") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85069, "ext_mem_85069") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85168, "mem_out_85168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85167, "mem_out_85167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85166, "mem_out_85166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85165, "mem_out_85165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85164, "mem_out_85164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85163, "mem_out_85163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85162, "mem_out_85162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85161, "mem_out_85161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85160, "mem_out_85160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85159, "mem_out_85159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85158, "mem_out_85158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85157, "mem_out_85157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85156, "mem_out_85156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85155, "mem_out_85155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85154, "mem_out_85154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85153, "mem_out_85153") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85152, "mem_out_85152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85151, "mem_out_85151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85150, "mem_out_85150") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85149, "mem_out_85149") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85148, "mem_out_85148") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85147, "mem_out_85147") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85146, "mem_out_85146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85145, "mem_out_85145") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85144, "mem_out_85144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85143, "mem_out_85143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85731, struct memblock *mem_out_p_85732, struct memblock *mem_out_p_85733, struct memblock *mem_out_p_85734, struct memblock *mem_out_p_85735, struct memblock *mem_out_p_85736, struct memblock *mem_out_p_85737, struct memblock *mem_out_p_85738, struct memblock *mem_out_p_85739)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mem_83335 = ctx->constants->mem_83335;
    struct memblock mem_83336 = ctx->constants->mem_83336;
    struct memblock mem_83337 = ctx->constants->mem_83337;
    struct memblock mem_83338 = ctx->constants->mem_83338;
    struct memblock mem_83339 = ctx->constants->mem_83339;
    struct memblock mem_83340 = ctx->constants->mem_83340;
    struct memblock mem_83341 = ctx->constants->mem_83341;
    struct memblock mem_83342 = ctx->constants->mem_83342;
    struct memblock mem_83343 = ctx->constants->mem_83343;
    
    if (memblock_set(ctx, &mem_out_85142, &mem_83342, "mem_83342") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &mem_83338, "mem_83338") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &mem_83340, "mem_83340") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85145, &mem_83336, "mem_83336") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85146, &mem_83337, "mem_83337") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85147, &mem_83335, "mem_83335") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85148, &mem_83341, "mem_83341") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85149, &mem_83339, "mem_83339") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85150, &mem_83343, "mem_83343") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85731, &mem_out_85142, "mem_out_85142") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85732, &mem_out_85143, "mem_out_85143") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85733, &mem_out_85144, "mem_out_85144") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85734, &mem_out_85145, "mem_out_85145") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85735, &mem_out_85146, "mem_out_85146") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85736, &mem_out_85147, "mem_out_85147") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85737, &mem_out_85148, "mem_out_85148") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85738, &mem_out_85149, "mem_out_85149") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85739, &mem_out_85150, "mem_out_85150") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85150, "mem_out_85150") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85149, "mem_out_85149") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85148, "mem_out_85148") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85147, "mem_out_85147") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85146, "mem_out_85146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85145, "mem_out_85145") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85144, "mem_out_85144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85143, "mem_out_85143") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85142, "mem_out_85142") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock mask_mem_83354;
    
    mask_mem_83354.references = NULL;
    
    struct memblock tokens_mem_83353;
    
    tokens_mem_83353.references = NULL;
    
    struct memblock wvoc_mem_83352;
    
    wvoc_mem_83352.references = NULL;
    
    struct memblock wval_mem_83351;
    
    wval_mem_83351.references = NULL;
    
    struct memblock wup_mem_83350;
    
    wup_mem_83350.references = NULL;
    
    struct memblock wte_mem_83349;
    
    wte_mem_83349.references = NULL;
    
    struct memblock wqry_mem_83348;
    
    wqry_mem_83348.references = NULL;
    
    struct memblock wpe_mem_83347;
    
    wpe_mem_83347.references = NULL;
    
    struct memblock wout_mem_83346;
    
    wout_mem_83346.references = NULL;
    
    struct memblock wkey_mem_83345;
    
    wkey_mem_83345.references = NULL;
    
    struct memblock wdown_mem_83344;
    
    wdown_mem_83344.references = NULL;
    wdown_mem_83344 = in0->v0->mem;
    wkey_mem_83345 = in0->v1->mem;
    wout_mem_83346 = in0->v2->mem;
    wpe_mem_83347 = in0->v3->mem;
    wqry_mem_83348 = in0->v4->mem;
    wte_mem_83349 = in0->v5->mem;
    wup_mem_83350 = in0->v6->mem;
    wval_mem_83351 = in0->v7->mem;
    wvoc_mem_83352 = in0->v8->mem;
    tokens_mem_83353 = in1->mem;
    mask_mem_83354 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_85142, wdown_mem_83344, wkey_mem_83345, wout_mem_83346, wpe_mem_83347, wqry_mem_83348, wte_mem_83349, wup_mem_83350, wval_mem_83351, wvoc_mem_83352, tokens_mem_83353, mask_mem_83354);
        if (ret == 0) {
            struct memblock mem_83335 = ctx->constants->mem_83335;
            struct memblock mem_83336 = ctx->constants->mem_83336;
            struct memblock mem_83337 = ctx->constants->mem_83337;
            struct memblock mem_83338 = ctx->constants->mem_83338;
            struct memblock mem_83339 = ctx->constants->mem_83339;
            struct memblock mem_83340 = ctx->constants->mem_83340;
            struct memblock mem_83341 = ctx->constants->mem_83341;
            struct memblock mem_83342 = ctx->constants->mem_83342;
            struct memblock mem_83343 = ctx->constants->mem_83343;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_85142;
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
    
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock wvoc_mem_83352;
    
    wvoc_mem_83352.references = NULL;
    
    struct memblock wdown_mem_83351;
    
    wdown_mem_83351.references = NULL;
    
    struct memblock wup_mem_83350;
    
    wup_mem_83350.references = NULL;
    
    struct memblock wout_mem_83349;
    
    wout_mem_83349.references = NULL;
    
    struct memblock wval_mem_83348;
    
    wval_mem_83348.references = NULL;
    
    struct memblock wkey_mem_83347;
    
    wkey_mem_83347.references = NULL;
    
    struct memblock wqry_mem_83346;
    
    wqry_mem_83346.references = NULL;
    
    struct memblock wpe_mem_83345;
    
    wpe_mem_83345.references = NULL;
    
    struct memblock wte_mem_83344;
    
    wte_mem_83344.references = NULL;
    wte_mem_83344 = in0->mem;
    wpe_mem_83345 = in1->mem;
    wqry_mem_83346 = in2->mem;
    wkey_mem_83347 = in3->mem;
    wval_mem_83348 = in4->mem;
    wout_mem_83349 = in5->mem;
    wup_mem_83350 = in6->mem;
    wdown_mem_83351 = in7->mem;
    wvoc_mem_83352 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_85142, &mem_out_85143, &mem_out_85144, &mem_out_85145, &mem_out_85146, &mem_out_85147, &mem_out_85148, &mem_out_85149, &mem_out_85150, wte_mem_83344, wpe_mem_83345, wqry_mem_83346, wkey_mem_83347, wval_mem_83348, wout_mem_83349, wup_mem_83350, wdown_mem_83351, wvoc_mem_83352);
        if (ret == 0) {
            struct memblock mem_83335 = ctx->constants->mem_83335;
            struct memblock mem_83336 = ctx->constants->mem_83336;
            struct memblock mem_83337 = ctx->constants->mem_83337;
            struct memblock mem_83338 = ctx->constants->mem_83338;
            struct memblock mem_83339 = ctx->constants->mem_83339;
            struct memblock mem_83340 = ctx->constants->mem_83340;
            struct memblock mem_83341 = ctx->constants->mem_83341;
            struct memblock mem_83342 = ctx->constants->mem_83342;
            struct memblock mem_83343 = ctx->constants->mem_83343;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85142;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85143;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85144;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85145;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85146;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85147;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85148;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85149;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85150;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f64_4d *in3, const struct futhark_i64_3d *in4)
{
    int64_t num_batches_62011 = (int64_t) 0;
    int64_t batchsizze_62012 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_85168;
    
    mem_out_85168.references = NULL;
    
    struct memblock mem_out_85167;
    
    mem_out_85167.references = NULL;
    
    struct memblock mem_out_85166;
    
    mem_out_85166.references = NULL;
    
    struct memblock mem_out_85165;
    
    mem_out_85165.references = NULL;
    
    struct memblock mem_out_85164;
    
    mem_out_85164.references = NULL;
    
    struct memblock mem_out_85163;
    
    mem_out_85163.references = NULL;
    
    struct memblock mem_out_85162;
    
    mem_out_85162.references = NULL;
    
    struct memblock mem_out_85161;
    
    mem_out_85161.references = NULL;
    
    struct memblock mem_out_85160;
    
    mem_out_85160.references = NULL;
    
    struct memblock mem_out_85159;
    
    mem_out_85159.references = NULL;
    
    struct memblock mem_out_85158;
    
    mem_out_85158.references = NULL;
    
    struct memblock mem_out_85157;
    
    mem_out_85157.references = NULL;
    
    struct memblock mem_out_85156;
    
    mem_out_85156.references = NULL;
    
    struct memblock mem_out_85155;
    
    mem_out_85155.references = NULL;
    
    struct memblock mem_out_85154;
    
    mem_out_85154.references = NULL;
    
    struct memblock mem_out_85153;
    
    mem_out_85153.references = NULL;
    
    struct memblock mem_out_85152;
    
    mem_out_85152.references = NULL;
    
    struct memblock mem_out_85151;
    
    mem_out_85151.references = NULL;
    
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    
    struct memblock seqs_mem_83372;
    
    seqs_mem_83372.references = NULL;
    
    struct memblock masks_mem_83371;
    
    masks_mem_83371.references = NULL;
    
    struct memblock wvoc_mem_83370;
    
    wvoc_mem_83370.references = NULL;
    
    struct memblock wval_mem_83369;
    
    wval_mem_83369.references = NULL;
    
    struct memblock wup_mem_83368;
    
    wup_mem_83368.references = NULL;
    
    struct memblock wte_mem_83367;
    
    wte_mem_83367.references = NULL;
    
    struct memblock wqry_mem_83366;
    
    wqry_mem_83366.references = NULL;
    
    struct memblock wpe_mem_83365;
    
    wpe_mem_83365.references = NULL;
    
    struct memblock wout_mem_83364;
    
    wout_mem_83364.references = NULL;
    
    struct memblock wkey_mem_83363;
    
    wkey_mem_83363.references = NULL;
    
    struct memblock wdown_mem_83362;
    
    wdown_mem_83362.references = NULL;
    
    struct memblock wvoc_mem_83361;
    
    wvoc_mem_83361.references = NULL;
    
    struct memblock wval_mem_83360;
    
    wval_mem_83360.references = NULL;
    
    struct memblock wup_mem_83359;
    
    wup_mem_83359.references = NULL;
    
    struct memblock wte_mem_83358;
    
    wte_mem_83358.references = NULL;
    
    struct memblock wqry_mem_83357;
    
    wqry_mem_83357.references = NULL;
    
    struct memblock wpe_mem_83356;
    
    wpe_mem_83356.references = NULL;
    
    struct memblock wout_mem_83355;
    
    wout_mem_83355.references = NULL;
    
    struct memblock wkey_mem_83354;
    
    wkey_mem_83354.references = NULL;
    
    struct memblock wdown_mem_83353;
    
    wdown_mem_83353.references = NULL;
    
    struct memblock wvoc_mem_83352;
    
    wvoc_mem_83352.references = NULL;
    
    struct memblock wval_mem_83351;
    
    wval_mem_83351.references = NULL;
    
    struct memblock wup_mem_83350;
    
    wup_mem_83350.references = NULL;
    
    struct memblock wte_mem_83349;
    
    wte_mem_83349.references = NULL;
    
    struct memblock wqry_mem_83348;
    
    wqry_mem_83348.references = NULL;
    
    struct memblock wpe_mem_83347;
    
    wpe_mem_83347.references = NULL;
    
    struct memblock wout_mem_83346;
    
    wout_mem_83346.references = NULL;
    
    struct memblock wkey_mem_83345;
    
    wkey_mem_83345.references = NULL;
    
    struct memblock wdown_mem_83344;
    
    wdown_mem_83344.references = NULL;
    wdown_mem_83344 = in0->v0->mem;
    wkey_mem_83345 = in0->v1->mem;
    wout_mem_83346 = in0->v2->mem;
    wpe_mem_83347 = in0->v3->mem;
    wqry_mem_83348 = in0->v4->mem;
    wte_mem_83349 = in0->v5->mem;
    wup_mem_83350 = in0->v6->mem;
    wval_mem_83351 = in0->v7->mem;
    wvoc_mem_83352 = in0->v8->mem;
    wdown_mem_83353 = in1->v0->mem;
    wkey_mem_83354 = in1->v1->mem;
    wout_mem_83355 = in1->v2->mem;
    wpe_mem_83356 = in1->v3->mem;
    wqry_mem_83357 = in1->v4->mem;
    wte_mem_83358 = in1->v5->mem;
    wup_mem_83359 = in1->v6->mem;
    wval_mem_83360 = in1->v7->mem;
    wvoc_mem_83361 = in1->v8->mem;
    wdown_mem_83362 = in2->v0->mem;
    wkey_mem_83363 = in2->v1->mem;
    wout_mem_83364 = in2->v2->mem;
    wpe_mem_83365 = in2->v3->mem;
    wqry_mem_83366 = in2->v4->mem;
    wte_mem_83367 = in2->v5->mem;
    wup_mem_83368 = in2->v6->mem;
    wval_mem_83369 = in2->v7->mem;
    wvoc_mem_83370 = in2->v8->mem;
    masks_mem_83371 = in3->mem;
    num_batches_62011 = in3->shape[0];
    batchsizze_62012 = in3->shape[1];
    seqs_mem_83372 = in4->mem;
    num_batches_62011 = in4->shape[0];
    batchsizze_62012 = in4->shape[1];
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && ((num_batches_62011 == in3->shape[0] && (batchsizze_62012 == in3->shape[1] && ((int64_t) 16 == in3->shape[2] && (int64_t) 16 == in3->shape[3]))) && (num_batches_62011 == in4->shape[0] && (batchsizze_62012 == in4->shape[1] && (int64_t) 16 == in4->shape[2]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_85142, &mem_out_85143, &mem_out_85144, &mem_out_85145, &mem_out_85146, &mem_out_85147, &mem_out_85148, &mem_out_85149, &mem_out_85150, &mem_out_85151, &mem_out_85152, &mem_out_85153, &mem_out_85154, &mem_out_85155, &mem_out_85156, &mem_out_85157, &mem_out_85158, &mem_out_85159, &mem_out_85160, &mem_out_85161, &mem_out_85162, &mem_out_85163, &mem_out_85164, &mem_out_85165, &mem_out_85166, &mem_out_85167, &mem_out_85168, wdown_mem_83344, wkey_mem_83345, wout_mem_83346, wpe_mem_83347, wqry_mem_83348, wte_mem_83349, wup_mem_83350, wval_mem_83351, wvoc_mem_83352, wdown_mem_83353, wkey_mem_83354, wout_mem_83355, wpe_mem_83356, wqry_mem_83357, wte_mem_83358, wup_mem_83359, wval_mem_83360, wvoc_mem_83361, wdown_mem_83362, wkey_mem_83363, wout_mem_83364, wpe_mem_83365, wqry_mem_83366, wte_mem_83367, wup_mem_83368, wval_mem_83369, wvoc_mem_83370, masks_mem_83371, seqs_mem_83372, num_batches_62011, batchsizze_62012);
        if (ret == 0) {
            struct memblock mem_83335 = ctx->constants->mem_83335;
            struct memblock mem_83336 = ctx->constants->mem_83336;
            struct memblock mem_83337 = ctx->constants->mem_83337;
            struct memblock mem_83338 = ctx->constants->mem_83338;
            struct memblock mem_83339 = ctx->constants->mem_83339;
            struct memblock mem_83340 = ctx->constants->mem_83340;
            struct memblock mem_83341 = ctx->constants->mem_83341;
            struct memblock mem_83342 = ctx->constants->mem_83342;
            struct memblock mem_83343 = ctx->constants->mem_83343;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85142;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85143;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85144;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85145;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85146;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85147;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85148;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85149;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85150;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_85151;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_85152;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_85153;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_85154;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_85155;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_85156;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_85157;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_85158;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_85159;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_85160;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_85161;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_85162;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_85163;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_85164;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_85165;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_85166;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_85167;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_85168;
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
    
    struct memblock mem_out_85150;
    
    mem_out_85150.references = NULL;
    
    struct memblock mem_out_85149;
    
    mem_out_85149.references = NULL;
    
    struct memblock mem_out_85148;
    
    mem_out_85148.references = NULL;
    
    struct memblock mem_out_85147;
    
    mem_out_85147.references = NULL;
    
    struct memblock mem_out_85146;
    
    mem_out_85146.references = NULL;
    
    struct memblock mem_out_85145;
    
    mem_out_85145.references = NULL;
    
    struct memblock mem_out_85144;
    
    mem_out_85144.references = NULL;
    
    struct memblock mem_out_85143;
    
    mem_out_85143.references = NULL;
    
    struct memblock mem_out_85142;
    
    mem_out_85142.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_85142, &mem_out_85143, &mem_out_85144, &mem_out_85145, &mem_out_85146, &mem_out_85147, &mem_out_85148, &mem_out_85149, &mem_out_85150);
        if (ret == 0) {
            struct memblock mem_83335 = ctx->constants->mem_83335;
            struct memblock mem_83336 = ctx->constants->mem_83336;
            struct memblock mem_83337 = ctx->constants->mem_83337;
            struct memblock mem_83338 = ctx->constants->mem_83338;
            struct memblock mem_83339 = ctx->constants->mem_83339;
            struct memblock mem_83340 = ctx->constants->mem_83340;
            struct memblock mem_83341 = ctx->constants->mem_83341;
            struct memblock mem_83342 = ctx->constants->mem_83342;
            struct memblock mem_83343 = ctx->constants->mem_83343;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85142;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85143;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85144;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85145;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85146;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85147;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85148;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85149;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85150;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
