
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
const struct type type_ZMZNZMZNZMZNf64;
const struct type type_ZMZNZMZNf64;
const struct type type_ZMZNZMZNi64;
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
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, &type_ZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZR, &type_ZMZNZMZNZMZNf64, &type_ZMZNZMZNf64, &type_ZMZNZMZNi64, &type_ZMZNi64, &type_params, NULL};
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
    struct memblock mem_83318;
    struct memblock mem_83319;
    struct memblock mem_83320;
    struct memblock mem_83321;
    struct memblock mem_83322;
    struct memblock mem_83323;
    struct memblock mem_83324;
    struct memblock mem_83325;
    struct memblock mem_83326;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10345(struct futhark_context *ctx, struct memblock *mem_out_p_85445, struct memblock *mem_out_p_85446, struct memblock *mem_out_p_85447, struct memblock w_mem_83327, struct memblock mw_mem_83328, struct memblock vw_mem_83329, struct memblock dw_mem_83330, int64_t n_60372, int64_t m_60373, int64_t step_60378, double lt_r_60379);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10346(struct futhark_context *ctx, struct memblock *mem_out_p_85450, struct memblock *mem_out_p_85451, struct memblock *mem_out_p_85452, struct memblock w_mem_83327, struct memblock mw_mem_83328, struct memblock vw_mem_83329, struct memblock dw_mem_83330, int64_t n_61405, int64_t m_61406, int64_t step_61411, double lt_r_61412);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85455, struct memblock wdown_mem_83327, struct memblock wkey_mem_83328, struct memblock wout_mem_83329, struct memblock wpe_mem_83330, struct memblock wqry_mem_83331, struct memblock wte_mem_83332, struct memblock wup_mem_83333, struct memblock wval_mem_83334, struct memblock wvoc_mem_83335, struct memblock tokens_mem_83336, struct memblock mask_mem_83337);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85510, struct memblock *mem_out_p_85511, struct memblock *mem_out_p_85512, struct memblock *mem_out_p_85513, struct memblock *mem_out_p_85514, struct memblock *mem_out_p_85515, struct memblock *mem_out_p_85516, struct memblock *mem_out_p_85517, struct memblock *mem_out_p_85518, struct memblock wte_mem_83327, struct memblock wpe_mem_83328, struct memblock wqry_mem_83329, struct memblock wkey_mem_83330, struct memblock wval_mem_83331, struct memblock wout_mem_83332, struct memblock wup_mem_83333, struct memblock wdown_mem_83334, struct memblock wvoc_mem_83335);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85519, struct memblock *mem_out_p_85520, struct memblock *mem_out_p_85521, struct memblock *mem_out_p_85522, struct memblock *mem_out_p_85523, struct memblock *mem_out_p_85524, struct memblock *mem_out_p_85525, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock *mem_out_p_85540, struct memblock *mem_out_p_85541, struct memblock *mem_out_p_85542, struct memblock *mem_out_p_85543, struct memblock *mem_out_p_85544, struct memblock *mem_out_p_85545, struct memblock wdown_mem_83327, struct memblock wkey_mem_83328, struct memblock wout_mem_83329, struct memblock wpe_mem_83330, struct memblock wqry_mem_83331, struct memblock wte_mem_83332, struct memblock wup_mem_83333, struct memblock wval_mem_83334, struct memblock wvoc_mem_83335, struct memblock wdown_mem_83336, struct memblock wkey_mem_83337, struct memblock wout_mem_83338, struct memblock wpe_mem_83339, struct memblock wqry_mem_83340, struct memblock wte_mem_83341, struct memblock wup_mem_83342, struct memblock wval_mem_83343, struct memblock wvoc_mem_83344, struct memblock wdown_mem_83345, struct memblock wkey_mem_83346, struct memblock wout_mem_83347, struct memblock wpe_mem_83348, struct memblock wqry_mem_83349, struct memblock wte_mem_83350, struct memblock wup_mem_83351, struct memblock wval_mem_83352, struct memblock wvoc_mem_83353, struct memblock masks_mem_83354, struct memblock dls_mem_83355, struct memblock seqs_mem_83356, int64_t num_steps_62012);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85715, struct memblock *mem_out_p_85716, struct memblock *mem_out_p_85717, struct memblock *mem_out_p_85718, struct memblock *mem_out_p_85719, struct memblock *mem_out_p_85720, struct memblock *mem_out_p_85721, struct memblock *mem_out_p_85722, struct memblock *mem_out_p_85723);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_83318 (ctx->constants->mem_83318)
    #define mem_83319 (ctx->constants->mem_83319)
    #define mem_83320 (ctx->constants->mem_83320)
    #define mem_83321 (ctx->constants->mem_83321)
    #define mem_83322 (ctx->constants->mem_83322)
    #define mem_83323 (ctx->constants->mem_83323)
    #define mem_83324 (ctx->constants->mem_83324)
    #define mem_83325 (ctx->constants->mem_83325)
    #define mem_83326 (ctx->constants->mem_83326)
    mem_83318.references = NULL;
    mem_83319.references = NULL;
    mem_83320.references = NULL;
    mem_83321.references = NULL;
    mem_83322.references = NULL;
    mem_83323.references = NULL;
    mem_83324.references = NULL;
    mem_83325.references = NULL;
    mem_83326.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83318, (int64_t) 3456, "mem_83318")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85427 = 0; nest_i_85427 < (int64_t) 27; nest_i_85427++) {
        for (int64_t nest_i_85428 = 0; nest_i_85428 < (int64_t) 16; nest_i_85428++) {
            ((double *) mem_83318.mem)[nest_i_85427 * (int64_t) 16 + nest_i_85428] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83319, (int64_t) 2048, "mem_83319")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85429 = 0; nest_i_85429 < (int64_t) 16; nest_i_85429++) {
        for (int64_t nest_i_85430 = 0; nest_i_85430 < (int64_t) 16; nest_i_85430++) {
            ((double *) mem_83319.mem)[nest_i_85429 * (int64_t) 16 + nest_i_85430] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83320, (int64_t) 2048, "mem_83320")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85431 = 0; nest_i_85431 < (int64_t) 16; nest_i_85431++) {
        for (int64_t nest_i_85432 = 0; nest_i_85432 < (int64_t) 16; nest_i_85432++) {
            ((double *) mem_83320.mem)[nest_i_85431 * (int64_t) 16 + nest_i_85432] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83321, (int64_t) 2048, "mem_83321")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85433 = 0; nest_i_85433 < (int64_t) 16; nest_i_85433++) {
        for (int64_t nest_i_85434 = 0; nest_i_85434 < (int64_t) 16; nest_i_85434++) {
            ((double *) mem_83321.mem)[nest_i_85433 * (int64_t) 16 + nest_i_85434] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83322, (int64_t) 2048, "mem_83322")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85435 = 0; nest_i_85435 < (int64_t) 16; nest_i_85435++) {
        for (int64_t nest_i_85436 = 0; nest_i_85436 < (int64_t) 16; nest_i_85436++) {
            ((double *) mem_83322.mem)[nest_i_85435 * (int64_t) 16 + nest_i_85436] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83323, (int64_t) 2048, "mem_83323")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85437 = 0; nest_i_85437 < (int64_t) 16; nest_i_85437++) {
        for (int64_t nest_i_85438 = 0; nest_i_85438 < (int64_t) 16; nest_i_85438++) {
            ((double *) mem_83323.mem)[nest_i_85437 * (int64_t) 16 + nest_i_85438] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83324, (int64_t) 8192, "mem_83324")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85439 = 0; nest_i_85439 < (int64_t) 64; nest_i_85439++) {
        for (int64_t nest_i_85440 = 0; nest_i_85440 < (int64_t) 16; nest_i_85440++) {
            ((double *) mem_83324.mem)[nest_i_85439 * (int64_t) 16 + nest_i_85440] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83325, (int64_t) 8192, "mem_83325")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85441 = 0; nest_i_85441 < (int64_t) 16; nest_i_85441++) {
        for (int64_t nest_i_85442 = 0; nest_i_85442 < (int64_t) 64; nest_i_85442++) {
            ((double *) mem_83325.mem)[nest_i_85441 * (int64_t) 64 + nest_i_85442] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83326, (int64_t) 3456, "mem_83326")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85443 = 0; nest_i_85443 < (int64_t) 27; nest_i_85443++) {
        for (int64_t nest_i_85444 = 0; nest_i_85444 < (int64_t) 16; nest_i_85444++) {
            ((double *) mem_83326.mem)[nest_i_85443 * (int64_t) 16 + nest_i_85444] = 0.0;
        }
    }
    #undef mem_83318
    #undef mem_83319
    #undef mem_83320
    #undef mem_83321
    #undef mem_83322
    #undef mem_83323
    #undef mem_83324
    #undef mem_83325
    #undef mem_83326
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_83318, "ctx->constants->mem_83318") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83319, "ctx->constants->mem_83319") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83320, "ctx->constants->mem_83320") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83321, "ctx->constants->mem_83321") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83322, "ctx->constants->mem_83322") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83323, "ctx->constants->mem_83323") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83324, "ctx->constants->mem_83324") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83325, "ctx->constants->mem_83325") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83326, "ctx->constants->mem_83326") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10345(struct futhark_context *ctx, struct memblock *mem_out_p_85445, struct memblock *mem_out_p_85446, struct memblock *mem_out_p_85447, struct memblock w_mem_83327, struct memblock mw_mem_83328, struct memblock vw_mem_83329, struct memblock dw_mem_83330, int64_t n_60372, int64_t m_60373, int64_t step_60378, double lt_r_60379)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83371_cached_sizze_85448 = 0;
    unsigned char *mem_83371 = NULL;
    int64_t mem_83374_cached_sizze_85449 = 0;
    unsigned char *mem_83374 = NULL;
    struct memblock mem_83409;
    
    mem_83409.references = NULL;
    
    struct memblock mem_83336;
    
    mem_83336.references = NULL;
    
    struct memblock mem_83333;
    
    mem_83333.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83331 = (int64_t) 8 * n_60372;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83332 = m_60373 * binop_x_83331;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83333, bytes_83332, "mem_83333")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83336, bytes_83332, "mem_83336")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82479 = 0; i_82479 < n_60372; i_82479++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82472 = 0; i_82472 < m_60373; i_82472++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78592 = ((double *) mw_mem_83328.mem)[i_82479 * m_60373 + i_82472];
            
            // futhark/microgpt.fut:358:10-20
            
            double zp_lhs_78593 = 0.85 * zt_rhs_78592;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78594 = ((double *) dw_mem_83330.mem)[i_82479 * m_60373 + i_82472];
            
            // futhark/microgpt.fut:358:35-45
            
            double zp_rhs_78595 = 0.15000000000000002 * zt_rhs_78594;
            
            // futhark/microgpt.fut:358:21-45
            
            double lifted_lambda_res_78596 = zp_lhs_78593 + zp_rhs_78595;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78603 = ((double *) vw_mem_83329.mem)[i_82479 * m_60373 + i_82472];
            
            // futhark/microgpt.fut:360:10-20
            
            double zp_lhs_78604 = 0.99 * zt_rhs_78603;
            
            // futhark/microgpt.fut:360:35-45
            
            double zt_lhs_78606 = 1.0000000000000009e-2 * zt_rhs_78594;
            
            // futhark/microgpt.fut:360:46-56
            
            double zp_rhs_78607 = zt_rhs_78594 * zt_lhs_78606;
            
            // futhark/microgpt.fut:360:21-56
            
            double lifted_lambda_res_78608 = zp_lhs_78604 + zp_rhs_78607;
            
            ((double *) mem_83333.mem)[i_82479 * m_60373 + i_82472] = lifted_lambda_res_78608;
            ((double *) mem_83336.mem)[i_82479 * m_60373 + i_82472] = lifted_lambda_res_78596;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65354 = sitofp_i64_f64(step_60378);
    
    // futhark/microgpt.fut:362:54-57
    
    double ztzt_rhs_65355 = 1.0 + i64_res_65354;
    
    // futhark/microgpt.fut:362:30-57
    
    double zm_rhs_65356 = fpow64(0.85, ztzt_rhs_65355);
    
    // futhark/microgpt.fut:362:23-57
    
    double zs_rhs_65357 = 1.0 - zm_rhs_65356;
    
    // futhark/microgpt.fut:364:31-58
    
    double zm_rhs_65395 = fpow64(0.99, ztzt_rhs_65355);
    
    // futhark/microgpt.fut:364:23-58
    
    double zs_rhs_65396 = 1.0 - zm_rhs_65395;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83371_cached_sizze_85448 < bytes_83332) {
        err = lexical_realloc(ctx, &mem_83371, &mem_83371_cached_sizze_85448, bytes_83332);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83374_cached_sizze_85449 < bytes_83332) {
        err = lexical_realloc(ctx, &mem_83374, &mem_83374_cached_sizze_85449, bytes_83332);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82493 = 0; i_82493 < n_60372; i_82493++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82486 = 0; i_82486 < m_60373; i_82486++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78628 = ((double *) mem_83336.mem)[i_82493 * m_60373 + i_82486];
            
            // futhark/microgpt.fut:362:18-57
            
            double lifted_lambda_res_78629 = zs_lhs_78628 / zs_rhs_65357;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78636 = ((double *) mem_83333.mem)[i_82493 * m_60373 + i_82486];
            
            // futhark/microgpt.fut:364:18-58
            
            double lifted_lambda_res_78637 = zs_lhs_78636 / zs_rhs_65396;
            
            ((double *) mem_83371)[i_82493 * m_60373 + i_82486] = lifted_lambda_res_78637;
            ((double *) mem_83374)[i_82493 * m_60373 + i_82486] = lifted_lambda_res_78629;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83409, bytes_83332, "mem_83409")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82502 = 0; i_82502 < n_60372; i_82502++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82498 = 0; i_82498 < m_60373; i_82498++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64518 = ((double *) w_mem_83327.mem)[i_82502 * m_60373 + i_82498];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64519 = ((double *) mem_83374)[i_82502 * m_60373 + i_82498];
            
            // futhark/microgpt.fut:366:21-34
            
            double zs_lhs_64520 = lt_r_60379 * zt_rhs_64519;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64521 = ((double *) mem_83371)[i_82502 * m_60373 + i_82498];
            
            // futhark/microgpt.fut:366:51-57
            
            double zp_lhs_64522 = fpow64(ztzt_lhs_64521, 0.5);
            
            // futhark/microgpt.fut:366:59-71
            
            double zs_rhs_64523 = 1.0e-8 + zp_lhs_64522;
            
            // futhark/microgpt.fut:366:35-71
            
            double zm_rhs_64524 = zs_lhs_64520 / zs_rhs_64523;
            
            // futhark/microgpt.fut:366:13-71
            
            double lifted_lambda_res_64525 = zm_lhs_64518 - zm_rhs_64524;
            
            ((double *) mem_83409.mem)[i_82502 * m_60373 + i_82498] = lifted_lambda_res_64525;
        }
    }
    if (memblock_set(ctx, &mem_out_85126, &mem_83409, "mem_83409") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &mem_83336, "mem_83336") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &mem_83333, "mem_83333") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85445, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85446, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85447, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83371);
        free(mem_83374);
        if (memblock_unref(ctx, &mem_83409, "mem_83409") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83336, "mem_83336") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83333, "mem_83333") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10346(struct futhark_context *ctx, struct memblock *mem_out_p_85450, struct memblock *mem_out_p_85451, struct memblock *mem_out_p_85452, struct memblock w_mem_83327, struct memblock mw_mem_83328, struct memblock vw_mem_83329, struct memblock dw_mem_83330, int64_t n_61405, int64_t m_61406, int64_t step_61411, double lt_r_61412)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83371_cached_sizze_85453 = 0;
    unsigned char *mem_83371 = NULL;
    int64_t mem_83374_cached_sizze_85454 = 0;
    unsigned char *mem_83374 = NULL;
    struct memblock mem_83409;
    
    mem_83409.references = NULL;
    
    struct memblock mem_83336;
    
    mem_83336.references = NULL;
    
    struct memblock mem_83333;
    
    mem_83333.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83331 = (int64_t) 8 * n_61405;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83332 = m_61406 * binop_x_83331;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83333, bytes_83332, "mem_83333")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83336, bytes_83332, "mem_83336")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82479 = 0; i_82479 < n_61405; i_82479++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82472 = 0; i_82472 < m_61406; i_82472++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78592 = ((double *) mw_mem_83328.mem)[i_82479 * m_61406 + i_82472];
            
            // futhark/microgpt.fut:358:10-20
            
            double zp_lhs_78593 = 0.85 * zt_rhs_78592;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78594 = ((double *) dw_mem_83330.mem)[i_82479 * m_61406 + i_82472];
            
            // futhark/microgpt.fut:358:35-45
            
            double zp_rhs_78595 = 0.15000000000000002 * zt_rhs_78594;
            
            // futhark/microgpt.fut:358:21-45
            
            double lifted_lambda_res_78596 = zp_lhs_78593 + zp_rhs_78595;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78603 = ((double *) vw_mem_83329.mem)[i_82479 * m_61406 + i_82472];
            
            // futhark/microgpt.fut:360:10-20
            
            double zp_lhs_78604 = 0.99 * zt_rhs_78603;
            
            // futhark/microgpt.fut:360:35-45
            
            double zt_lhs_78606 = 1.0000000000000009e-2 * zt_rhs_78594;
            
            // futhark/microgpt.fut:360:46-56
            
            double zp_rhs_78607 = zt_rhs_78594 * zt_lhs_78606;
            
            // futhark/microgpt.fut:360:21-56
            
            double lifted_lambda_res_78608 = zp_lhs_78604 + zp_rhs_78607;
            
            ((double *) mem_83333.mem)[i_82479 * m_61406 + i_82472] = lifted_lambda_res_78608;
            ((double *) mem_83336.mem)[i_82479 * m_61406 + i_82472] = lifted_lambda_res_78596;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65354 = sitofp_i64_f64(step_61411);
    
    // futhark/microgpt.fut:362:54-57
    
    double ztzt_rhs_65355 = 1.0 + i64_res_65354;
    
    // futhark/microgpt.fut:362:30-57
    
    double zm_rhs_65356 = fpow64(0.85, ztzt_rhs_65355);
    
    // futhark/microgpt.fut:362:23-57
    
    double zs_rhs_65357 = 1.0 - zm_rhs_65356;
    
    // futhark/microgpt.fut:364:31-58
    
    double zm_rhs_65395 = fpow64(0.99, ztzt_rhs_65355);
    
    // futhark/microgpt.fut:364:23-58
    
    double zs_rhs_65396 = 1.0 - zm_rhs_65395;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83371_cached_sizze_85453 < bytes_83332) {
        err = lexical_realloc(ctx, &mem_83371, &mem_83371_cached_sizze_85453, bytes_83332);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83374_cached_sizze_85454 < bytes_83332) {
        err = lexical_realloc(ctx, &mem_83374, &mem_83374_cached_sizze_85454, bytes_83332);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82493 = 0; i_82493 < n_61405; i_82493++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82486 = 0; i_82486 < m_61406; i_82486++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78628 = ((double *) mem_83336.mem)[i_82493 * m_61406 + i_82486];
            
            // futhark/microgpt.fut:362:18-57
            
            double lifted_lambda_res_78629 = zs_lhs_78628 / zs_rhs_65357;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78636 = ((double *) mem_83333.mem)[i_82493 * m_61406 + i_82486];
            
            // futhark/microgpt.fut:364:18-58
            
            double lifted_lambda_res_78637 = zs_lhs_78636 / zs_rhs_65396;
            
            ((double *) mem_83371)[i_82493 * m_61406 + i_82486] = lifted_lambda_res_78637;
            ((double *) mem_83374)[i_82493 * m_61406 + i_82486] = lifted_lambda_res_78629;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83409, bytes_83332, "mem_83409")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82502 = 0; i_82502 < n_61405; i_82502++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82498 = 0; i_82498 < m_61406; i_82498++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64518 = ((double *) w_mem_83327.mem)[i_82502 * m_61406 + i_82498];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64519 = ((double *) mem_83374)[i_82502 * m_61406 + i_82498];
            
            // futhark/microgpt.fut:366:21-34
            
            double zs_lhs_64520 = lt_r_61412 * zt_rhs_64519;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64521 = ((double *) mem_83371)[i_82502 * m_61406 + i_82498];
            
            // futhark/microgpt.fut:366:51-57
            
            double zp_lhs_64522 = fpow64(ztzt_lhs_64521, 0.5);
            
            // futhark/microgpt.fut:366:59-71
            
            double zs_rhs_64523 = 1.0e-8 + zp_lhs_64522;
            
            // futhark/microgpt.fut:366:35-71
            
            double zm_rhs_64524 = zs_lhs_64520 / zs_rhs_64523;
            
            // futhark/microgpt.fut:366:13-71
            
            double lifted_lambda_res_64525 = zm_lhs_64518 - zm_rhs_64524;
            
            ((double *) mem_83409.mem)[i_82502 * m_61406 + i_82498] = lifted_lambda_res_64525;
        }
    }
    if (memblock_set(ctx, &mem_out_85126, &mem_83409, "mem_83409") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &mem_83336, "mem_83336") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &mem_83333, "mem_83333") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85450, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85451, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85452, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83371);
        free(mem_83374);
        if (memblock_unref(ctx, &mem_83409, "mem_83409") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83336, "mem_83336") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83333, "mem_83333") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85455, struct memblock wdown_mem_83327, struct memblock wkey_mem_83328, struct memblock wout_mem_83329, struct memblock wpe_mem_83330, struct memblock wqry_mem_83331, struct memblock wte_mem_83332, struct memblock wup_mem_83333, struct memblock wval_mem_83334, struct memblock wvoc_mem_83335, struct memblock tokens_mem_83336, struct memblock mask_mem_83337)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83338_cached_sizze_85456 = 0;
    unsigned char *mem_83338 = NULL;
    int64_t mem_83343_cached_sizze_85457 = 0;
    unsigned char *mem_83343 = NULL;
    int64_t mem_83354_cached_sizze_85458 = 0;
    unsigned char *mem_83354 = NULL;
    int64_t mem_83359_cached_sizze_85459 = 0;
    unsigned char *mem_83359 = NULL;
    int64_t mem_83370_cached_sizze_85460 = 0;
    unsigned char *mem_83370 = NULL;
    int64_t mem_83375_cached_sizze_85461 = 0;
    unsigned char *mem_83375 = NULL;
    int64_t mem_83382_cached_sizze_85462 = 0;
    unsigned char *mem_83382 = NULL;
    int64_t mem_83393_cached_sizze_85463 = 0;
    unsigned char *mem_83393 = NULL;
    int64_t mem_83398_cached_sizze_85464 = 0;
    unsigned char *mem_83398 = NULL;
    int64_t mem_83405_cached_sizze_85465 = 0;
    unsigned char *mem_83405 = NULL;
    int64_t mem_83416_cached_sizze_85466 = 0;
    unsigned char *mem_83416 = NULL;
    int64_t mem_83417_cached_sizze_85467 = 0;
    unsigned char *mem_83417 = NULL;
    int64_t mem_83418_cached_sizze_85468 = 0;
    unsigned char *mem_83418 = NULL;
    int64_t mem_83431_cached_sizze_85469 = 0;
    unsigned char *mem_83431 = NULL;
    int64_t mem_83432_cached_sizze_85470 = 0;
    unsigned char *mem_83432 = NULL;
    int64_t mem_83433_cached_sizze_85471 = 0;
    unsigned char *mem_83433 = NULL;
    int64_t mem_83464_cached_sizze_85472 = 0;
    unsigned char *mem_83464 = NULL;
    int64_t mem_83465_cached_sizze_85473 = 0;
    unsigned char *mem_83465 = NULL;
    int64_t mem_83466_cached_sizze_85474 = 0;
    unsigned char *mem_83466 = NULL;
    int64_t mem_83482_cached_sizze_85475 = 0;
    unsigned char *mem_83482 = NULL;
    int64_t mem_83483_cached_sizze_85476 = 0;
    unsigned char *mem_83483 = NULL;
    int64_t mem_83484_cached_sizze_85477 = 0;
    unsigned char *mem_83484 = NULL;
    int64_t mem_83497_cached_sizze_85478 = 0;
    unsigned char *mem_83497 = NULL;
    int64_t mem_83498_cached_sizze_85479 = 0;
    unsigned char *mem_83498 = NULL;
    int64_t mem_83499_cached_sizze_85480 = 0;
    unsigned char *mem_83499 = NULL;
    int64_t mem_83545_cached_sizze_85481 = 0;
    unsigned char *mem_83545 = NULL;
    int64_t mem_83551_cached_sizze_85482 = 0;
    unsigned char *mem_83551 = NULL;
    int64_t mem_83556_cached_sizze_85483 = 0;
    unsigned char *mem_83556 = NULL;
    int64_t mem_83567_cached_sizze_85484 = 0;
    unsigned char *mem_83567 = NULL;
    int64_t mem_83572_cached_sizze_85485 = 0;
    unsigned char *mem_83572 = NULL;
    int64_t mem_83583_cached_sizze_85486 = 0;
    unsigned char *mem_83583 = NULL;
    int64_t mem_83588_cached_sizze_85487 = 0;
    unsigned char *mem_83588 = NULL;
    int64_t mem_83595_cached_sizze_85488 = 0;
    unsigned char *mem_83595 = NULL;
    int64_t mem_83602_cached_sizze_85489 = 0;
    unsigned char *mem_83602 = NULL;
    int64_t mem_83613_cached_sizze_85490 = 0;
    unsigned char *mem_83613 = NULL;
    int64_t mem_83618_cached_sizze_85491 = 0;
    unsigned char *mem_83618 = NULL;
    int64_t mem_83634_cached_sizze_85492 = 0;
    unsigned char *mem_83634 = NULL;
    int64_t mem_83639_cached_sizze_85493 = 0;
    unsigned char *mem_83639 = NULL;
    int64_t mem_83650_cached_sizze_85494 = 0;
    unsigned char *mem_83650 = NULL;
    int64_t mem_83655_cached_sizze_85495 = 0;
    unsigned char *mem_83655 = NULL;
    int64_t mem_83666_cached_sizze_85496 = 0;
    unsigned char *mem_83666 = NULL;
    int64_t mem_83671_cached_sizze_85497 = 0;
    unsigned char *mem_83671 = NULL;
    int64_t mem_83682_cached_sizze_85498 = 0;
    unsigned char *mem_83682 = NULL;
    int64_t mem_83687_cached_sizze_85499 = 0;
    unsigned char *mem_83687 = NULL;
    int64_t mem_83694_cached_sizze_85500 = 0;
    unsigned char *mem_83694 = NULL;
    int64_t mem_83705_cached_sizze_85501 = 0;
    unsigned char *mem_83705 = NULL;
    int64_t mem_83710_cached_sizze_85502 = 0;
    unsigned char *mem_83710 = NULL;
    int64_t mem_83721_cached_sizze_85503 = 0;
    unsigned char *mem_83721 = NULL;
    int64_t mem_83726_cached_sizze_85504 = 0;
    unsigned char *mem_83726 = NULL;
    int64_t mem_83737_cached_sizze_85505 = 0;
    unsigned char *mem_83737 = NULL;
    int64_t mem_83742_cached_sizze_85506 = 0;
    unsigned char *mem_83742 = NULL;
    int64_t mem_83753_cached_sizze_85507 = 0;
    unsigned char *mem_83753 = NULL;
    int64_t mem_83758_cached_sizze_85508 = 0;
    unsigned char *mem_83758 = NULL;
    int64_t mem_83774_cached_sizze_85509 = 0;
    unsigned char *mem_83774 = NULL;
    struct memblock mem_83769;
    
    mem_83769.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83338_cached_sizze_85456 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83338, &mem_83338_cached_sizze_85456, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83343_cached_sizze_85457 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83343, &mem_83343_cached_sizze_85457, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82474 = 0; i_82474 < (int64_t) 16; i_82474++) {
        // futhark/microgpt.fut:348:41-50
        
        int64_t tmp_73021 = ((int64_t *) tokens_mem_83336.mem)[i_82474];
        
        // futhark/microgpt.fut:348:37-51
        
        bool x_73022 = sle64((int64_t) 0, tmp_73021);
        
        // futhark/microgpt.fut:348:37-51
        
        bool y_73023 = slt64(tmp_73021, (int64_t) 27);
        
        // futhark/microgpt.fut:348:37-51
        
        bool bounds_check_73024 = x_73022 && y_73023;
        
        // futhark/microgpt.fut:348:37-51
        
        bool index_certs_73025;
        
        if (!bounds_check_73024) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73021, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:348:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:348:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82470 = 0; i_82470 < (int64_t) 16; i_82470++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73032 = ((double *) wte_mem_83332.mem)[tmp_73021 * (int64_t) 16 + i_82470];
            
            ((double *) mem_83343)[i_82470] = lifted_lambda_res_73032;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83338, i_82474 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83343, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83354_cached_sizze_85458 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83354, &mem_83354_cached_sizze_85458, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83359_cached_sizze_85459 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83359, &mem_83359_cached_sizze_85459, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82482 = 0; i_82482 < (int64_t) 16; i_82482++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82478 = 0; i_82478 < (int64_t) 16; i_82478++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73064 = ((double *) wpe_mem_83330.mem)[i_82482 * (int64_t) 16 + i_82478];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73065 = ((double *) mem_83338)[i_82482 * (int64_t) 16 + i_82478];
            
            // futhark/microgpt.fut:149:38-70
            
            double zp_res_73066 = zp_lhs_73064 + zp_rhs_73065;
            
            ((double *) mem_83359)[i_82478] = zp_res_73066;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83354, i_82482 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83359, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83370_cached_sizze_85460 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83370, &mem_83370_cached_sizze_85460, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83375_cached_sizze_85461 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83375, &mem_83375_cached_sizze_85461, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83382_cached_sizze_85462 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83382, &mem_83382_cached_sizze_85462, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82494 = 0; i_82494 < (int64_t) 16; i_82494++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82486 = 0; i_82486 < (int64_t) 16; i_82486++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73081 = ((double *) mem_83354)[i_82494 * (int64_t) 16 + i_82486];
            
            // futhark/microgpt.fut:150:64-93
            
            double zt_res_73082 = zt_lhs_73081 * zt_lhs_73081;
            
            ((double *) mem_83375)[i_82486] = zt_res_73082;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73084;
        double r_73086 = 0.0;
        
        for (int64_t i_73085 = 0; i_73085 < (int64_t) 16; i_73085++) {
            // futhark/microgpt.fut:151:35-43
            
            double lifted_lambda_res_73087 = ((double *) mem_83375)[i_73085];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73088 = r_73086 + lifted_lambda_res_73087;
            double r_tmp_85133 = zp_res_73088;
            
            r_73086 = r_tmp_85133;
        }
        defunc_0_lifted_lambda_res_73084 = r_73086;
        // futhark/microgpt.fut:151:17-60
        
        double zs_res_73089 = defunc_0_lifted_lambda_res_73084 / 16.0;
        
        // futhark/microgpt.fut:152:24-55
        
        double zp_res_73090 = 1.0e-5 + zs_res_73089;
        
        // futhark/microgpt.fut:152:16-55
        
        double sqrt_res_73091 = futrts_sqrt64(zp_res_73090);
        
        // futhark/microgpt.fut:153:42-53
        
        double zs_res_73092 = 1.0 / sqrt_res_73091;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82490 = 0; i_82490 < (int64_t) 16; i_82490++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73099 = ((double *) mem_83354)[i_82494 * (int64_t) 16 + i_82490];
            
            // futhark/microgpt.fut:153:24-53
            
            double zt_res_73100 = zs_res_73092 * zt_lhs_73099;
            
            ((double *) mem_83382)[i_82490] = zt_res_73100;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83370, i_82494 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83382, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83393_cached_sizze_85463 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83393, &mem_83393_cached_sizze_85463, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83398_cached_sizze_85464 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83398, &mem_83398_cached_sizze_85464, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83405_cached_sizze_85465 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83405, &mem_83405_cached_sizze_85465, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82506 = 0; i_82506 < (int64_t) 16; i_82506++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82498 = 0; i_82498 < (int64_t) 16; i_82498++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73115 = ((double *) mem_83370)[i_82506 * (int64_t) 16 + i_82498];
            
            // futhark/microgpt.fut:154:64-93
            
            double zt_res_73116 = zt_lhs_73115 * zt_lhs_73115;
            
            ((double *) mem_83398)[i_82498] = zt_res_73116;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73118;
        double r_73120 = 0.0;
        
        for (int64_t i_73119 = 0; i_73119 < (int64_t) 16; i_73119++) {
            // futhark/microgpt.fut:155:35-43
            
            double lifted_lambda_res_73121 = ((double *) mem_83398)[i_73119];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73122 = r_73120 + lifted_lambda_res_73121;
            double r_tmp_85137 = zp_res_73122;
            
            r_73120 = r_tmp_85137;
        }
        defunc_0_lifted_lambda_res_73118 = r_73120;
        // futhark/microgpt.fut:155:17-60
        
        double zs_res_73123 = defunc_0_lifted_lambda_res_73118 / 16.0;
        
        // futhark/microgpt.fut:156:24-55
        
        double zp_res_73124 = 1.0e-5 + zs_res_73123;
        
        // futhark/microgpt.fut:156:16-55
        
        double sqrt_res_73125 = futrts_sqrt64(zp_res_73124);
        
        // futhark/microgpt.fut:157:42-53
        
        double zs_res_73126 = 1.0 / sqrt_res_73125;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82502 = 0; i_82502 < (int64_t) 16; i_82502++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73133 = ((double *) mem_83370)[i_82506 * (int64_t) 16 + i_82502];
            
            // futhark/microgpt.fut:157:24-53
            
            double zt_res_73134 = zs_res_73126 * zt_lhs_73133;
            
            ((double *) mem_83405)[i_82502] = zt_res_73134;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83393, i_82506 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83416_cached_sizze_85466 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83416, &mem_83416_cached_sizze_85466, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83417_cached_sizze_85467 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83417, &mem_83417_cached_sizze_85467, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83418_cached_sizze_85468 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83418, &mem_83418_cached_sizze_85468, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83431_cached_sizze_85469 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83431, &mem_83431_cached_sizze_85469, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83432_cached_sizze_85470 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83432, &mem_83432_cached_sizze_85470, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83433_cached_sizze_85471 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83433, &mem_83433_cached_sizze_85471, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82524 = 0; i_82524 < (int64_t) 16; i_82524++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82514 = 0; i_82514 < (int64_t) 16; i_82514++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78811;
            double r_78813 = 0.0;
            
            for (int64_t i_78812 = 0; i_78812 < (int64_t) 16; i_78812++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78814 = ((double *) wqry_mem_83331.mem)[i_82514 * (int64_t) 16 + i_78812];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78815 = ((double *) mem_83393)[i_82524 * (int64_t) 16 + i_78812];
                
                // futhark/microgpt.fut:158:72-103
                
                double zt_res_78816 = zt_lhs_78814 * zt_rhs_78815;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78817 = r_78813 + zt_res_78816;
                double r_tmp_85145 = zp_res_78817;
                
                r_78813 = r_tmp_85145;
            }
            defunc_0_lifted_lambda_res_78811 = r_78813;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78824;
            double r_78826 = 0.0;
            
            for (int64_t i_78825 = 0; i_78825 < (int64_t) 16; i_78825++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78827 = ((double *) wkey_mem_83328.mem)[i_82514 * (int64_t) 16 + i_78825];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78828 = ((double *) mem_83393)[i_82524 * (int64_t) 16 + i_78825];
                
                // futhark/microgpt.fut:159:72-103
                
                double zt_res_78829 = zt_lhs_78827 * zt_rhs_78828;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78830 = r_78826 + zt_res_78829;
                double r_tmp_85146 = zp_res_78830;
                
                r_78826 = r_tmp_85146;
            }
            defunc_0_lifted_lambda_res_78824 = r_78826;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78840;
            double r_78842 = 0.0;
            
            for (int64_t i_78841 = 0; i_78841 < (int64_t) 16; i_78841++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78843 = ((double *) wval_mem_83334.mem)[i_82514 * (int64_t) 16 + i_78841];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78844 = ((double *) mem_83393)[i_82524 * (int64_t) 16 + i_78841];
                
                // futhark/microgpt.fut:160:72-103
                
                double zt_res_78845 = zt_lhs_78843 * zt_rhs_78844;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78846 = r_78842 + zt_res_78845;
                double r_tmp_85147 = zp_res_78846;
                
                r_78842 = r_tmp_85147;
            }
            defunc_0_lifted_lambda_res_78840 = r_78842;
            ((double *) mem_83431)[i_82514] = defunc_0_lifted_lambda_res_78840;
            ((double *) mem_83432)[i_82514] = defunc_0_lifted_lambda_res_78824;
            ((double *) mem_83433)[i_82514] = defunc_0_lifted_lambda_res_78811;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83416, i_82524 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83431, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83417, i_82524 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83432, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83418, i_82524 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83433, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83464_cached_sizze_85472 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83464, &mem_83464_cached_sizze_85472, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83465_cached_sizze_85473 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83465, &mem_83465_cached_sizze_85473, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83466_cached_sizze_85474 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83466, &mem_83466_cached_sizze_85474, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83482_cached_sizze_85475 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83482, &mem_83482_cached_sizze_85475, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83483_cached_sizze_85476 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83483, &mem_83483_cached_sizze_85476, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83484_cached_sizze_85477 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83484, &mem_83484_cached_sizze_85477, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83497_cached_sizze_85478 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83497, &mem_83497_cached_sizze_85478, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83498_cached_sizze_85479 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83498, &mem_83498_cached_sizze_85479, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83499_cached_sizze_85480 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83499, &mem_83499_cached_sizze_85480, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82554 = 0; i_82554 < (int64_t) 4; i_82554++) {
        // futhark/microgpt.fut:161:83-86
        
        int64_t zp_lhs_78686 = mul64((int64_t) 4, i_82554);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82544 = 0; i_82544 < (int64_t) 16; i_82544++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82534 = 0; i_82534 < (int64_t) 4; i_82534++) {
                // futhark/microgpt.fut:161:88-93
                
                int64_t tmp_79004 = add64(zp_lhs_78686, i_82534);
                
                // futhark/microgpt.fut:161:69-95
                
                bool x_79005 = sle64((int64_t) 0, tmp_79004);
                
                // futhark/microgpt.fut:161:69-95
                
                bool y_79006 = slt64(tmp_79004, (int64_t) 16);
                
                // futhark/microgpt.fut:161:69-95
                
                bool bounds_check_79007 = x_79005 && y_79006;
                
                // futhark/microgpt.fut:161:69-95
                
                bool index_certs_79008;
                
                if (!bounds_check_79007) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79004, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:161:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:161:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:161:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:161:15-100\n   #10 futhark/microgpt.fut:349:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79009 = ((double *) mem_83418)[i_82544 * (int64_t) 16 + tmp_79004];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79017 = ((double *) mem_83417)[i_82544 * (int64_t) 16 + tmp_79004];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79028 = ((double *) mem_83416)[i_82544 * (int64_t) 16 + tmp_79004];
                
                ((double *) mem_83497)[i_82534] = lifted_lambda_res_79028;
                ((double *) mem_83498)[i_82534] = lifted_lambda_res_79017;
                ((double *) mem_83499)[i_82534] = lifted_lambda_res_79009;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83482, i_82544 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83497, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83483, i_82544 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83498, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83484, i_82544 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83499, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83464, i_82554 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83482, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83465, i_82554 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83483, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83466, i_82554 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83484, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83545_cached_sizze_85481 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83545, &mem_83545_cached_sizze_85481, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83551_cached_sizze_85482 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83551, &mem_83551_cached_sizze_85482, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83556_cached_sizze_85483 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83556, &mem_83556_cached_sizze_85483, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83567_cached_sizze_85484 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83567, &mem_83567_cached_sizze_85484, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83572_cached_sizze_85485 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83572, &mem_83572_cached_sizze_85485, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83583_cached_sizze_85486 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83583, &mem_83583_cached_sizze_85486, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83588_cached_sizze_85487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83588, &mem_83588_cached_sizze_85487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83595_cached_sizze_85488 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83595, &mem_83595_cached_sizze_85488, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83602_cached_sizze_85489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83602, &mem_83602_cached_sizze_85489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83613_cached_sizze_85490 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83613, &mem_83613_cached_sizze_85490, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83618_cached_sizze_85491 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83618, &mem_83618_cached_sizze_85491, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82602 = 0; i_82602 < (int64_t) 4; i_82602++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82564 = 0; i_82564 < (int64_t) 16; i_82564++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82560 = 0; i_82560 < (int64_t) 16; i_82560++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73279;
                double r_73281 = 0.0;
                
                for (int64_t i_73280 = 0; i_73280 < (int64_t) 4; i_73280++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73282 = ((double *) mem_83466)[i_82602 * (int64_t) 64 + i_82564 * (int64_t) 4 + i_73280];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73283 = ((double *) mem_83465)[i_82602 * (int64_t) 64 + i_82560 * (int64_t) 4 + i_73280];
                    
                    // futhark/microgpt.fut:164:100-139
                    
                    double zt_res_73284 = zt_lhs_73282 * zt_rhs_73283;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73285 = r_73281 + zt_res_73284;
                    double r_tmp_85160 = zp_res_73285;
                    
                    r_73281 = r_tmp_85160;
                }
                defunc_0_lifted_lambda_res_73279 = r_73281;
                ((double *) mem_83556)[i_82560] = defunc_0_lifted_lambda_res_73279;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83551, i_82564 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83556, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82572 = 0; i_82572 < (int64_t) 16; i_82572++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82568 = 0; i_82568 < (int64_t) 16; i_82568++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_73300 = ((double *) mem_83551)[i_82572 * (int64_t) 16 + i_82568];
                
                // futhark/microgpt.fut:165:43-70
                
                double zs_res_73301 = zs_lhs_73300 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_73302 = ((double *) mask_mem_83337.mem)[i_82572 * (int64_t) 16 + i_82568];
                
                // futhark/microgpt.fut:165:57-90
                
                double zp_res_73303 = zs_res_73301 + zp_rhs_73302;
                
                ((double *) mem_83572)[i_82568] = zp_res_73303;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83567, i_82572 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83572, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82590 = 0; i_82590 < (int64_t) 16; i_82590++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_79103;
            double redout_82574 = -INFINITY;
            
            for (int64_t i_82575 = 0; i_82575 < (int64_t) 16; i_82575++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79055 = ((double *) mem_83567)[i_82590 * (int64_t) 16 + i_82575];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_73324 = fmax64(lifted_lambda_res_79055, redout_82574);
                double redout_tmp_85164 = max_res_73324;
                
                redout_82574 = redout_tmp_85164;
            }
            defunc_0_reduce_res_79103 = redout_82574;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_73325 = -defunc_0_reduce_res_79103;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82578 = 0; i_82578 < (int64_t) 16; i_82578++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_73332 = ((double *) mem_83567)[i_82590 * (int64_t) 16 + i_82578];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_73333 = neg_res_73325 + lifted_lambda_res_73332;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_73334 = futrts_exp64(zp_res_73333);
                
                ((double *) mem_83588)[i_82578] = exp_res_73334;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73336;
            double r_73338 = 0.0;
            
            for (int64_t i_73337 = 0; i_73337 < (int64_t) 16; i_73337++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_73339 = ((double *) mem_83588)[i_73337];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73340 = r_73338 + lifted_lambda_res_73339;
                double r_tmp_85166 = zp_res_73340;
                
                r_73338 = r_tmp_85166;
            }
            defunc_0_lifted_lambda_res_73336 = r_73338;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82582 = 0; i_82582 < (int64_t) 16; i_82582++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_73347 = ((double *) mem_83588)[i_82582];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_73348 = zs_lhs_73347 / defunc_0_lifted_lambda_res_73336;
                
                ((double *) mem_83595)[i_82582] = zs_res_73348;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82586 = 0; i_82586 < (int64_t) 16; i_82586++) {
                // futhark/microgpt.fut:167:23-31
                
                double lifted_lambda_res_73356 = ((double *) mem_83595)[i_82586];
                
                ((double *) mem_83602)[i_82586] = lifted_lambda_res_73356;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83583, i_82590 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83602, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82598 = 0; i_82598 < (int64_t) 16; i_82598++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82594 = 0; i_82594 < (int64_t) 4; i_82594++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73371;
                double r_73373 = 0.0;
                
                for (int64_t i_73372 = 0; i_73372 < (int64_t) 16; i_73372++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73374 = ((double *) mem_83583)[i_82598 * (int64_t) 16 + i_73372];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73375 = ((double *) mem_83464)[i_82602 * (int64_t) 64 + i_73372 * (int64_t) 4 + i_82594];
                    
                    // futhark/microgpt.fut:168:61-96
                    
                    double zt_res_73376 = zt_lhs_73374 * zt_rhs_73375;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73377 = r_73373 + zt_res_73376;
                    double r_tmp_85171 = zp_res_73377;
                    
                    r_73373 = r_tmp_85171;
                }
                defunc_0_lifted_lambda_res_73371 = r_73373;
                ((double *) mem_83618)[i_82594] = defunc_0_lifted_lambda_res_73371;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83613, i_82598 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83618, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83545, i_82602 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83613, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83634_cached_sizze_85492 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83634, &mem_83634_cached_sizze_85492, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83639_cached_sizze_85493 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83639, &mem_83639_cached_sizze_85493, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82610 = 0; i_82610 < (int64_t) 16; i_82610++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82606 = 0; i_82606 < (int64_t) 16; i_82606++) {
            // futhark/microgpt.fut:169:61-64
            
            int64_t tmp_73389 = sdiv64(i_82606, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool x_73390 = sle64((int64_t) 0, tmp_73389);
            
            // futhark/microgpt.fut:169:53-66
            
            bool y_73391 = slt64(tmp_73389, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool bounds_check_73392 = x_73390 && y_73391;
            
            // futhark/microgpt.fut:169:53-66
            
            bool index_certs_73393;
            
            if (!bounds_check_73392) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73389, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:349:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:169:77-80
            
            int64_t tmp_73394 = smod64(i_82606, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool x_73395 = sle64((int64_t) 0, tmp_73394);
            
            // futhark/microgpt.fut:169:53-82
            
            bool y_73396 = slt64(tmp_73394, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool bounds_check_73397 = x_73395 && y_73396;
            
            // futhark/microgpt.fut:169:53-82
            
            bool index_certs_73398;
            
            if (!bounds_check_73397) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73394, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:349:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73399 = ((double *) mem_83545)[tmp_73389 * (int64_t) 64 + i_82610 * (int64_t) 4 + tmp_73394];
            
            ((double *) mem_83639)[i_82606] = lifted_lambda_res_73399;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83634, i_82610 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83639, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83650_cached_sizze_85494 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83650, &mem_83650_cached_sizze_85494, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83655_cached_sizze_85495 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83655, &mem_83655_cached_sizze_85495, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82618 = 0; i_82618 < (int64_t) 16; i_82618++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82614 = 0; i_82614 < (int64_t) 16; i_82614++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73414;
            double r_73416 = 0.0;
            
            for (int64_t i_73415 = 0; i_73415 < (int64_t) 16; i_73415++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73417 = ((double *) wout_mem_83329.mem)[i_82614 * (int64_t) 16 + i_73415];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73418 = ((double *) mem_83634)[i_82618 * (int64_t) 16 + i_73415];
                
                // futhark/microgpt.fut:170:73-105
                
                double zt_res_73419 = zt_lhs_73417 * zt_rhs_73418;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73420 = r_73416 + zt_res_73419;
                double r_tmp_85176 = zp_res_73420;
                
                r_73416 = r_tmp_85176;
            }
            defunc_0_lifted_lambda_res_73414 = r_73416;
            ((double *) mem_83655)[i_82614] = defunc_0_lifted_lambda_res_73414;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83650, i_82618 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83655, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83666_cached_sizze_85496 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83666, &mem_83666_cached_sizze_85496, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83671_cached_sizze_85497 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83671, &mem_83671_cached_sizze_85497, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82626 = 0; i_82626 < (int64_t) 16; i_82626++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82622 = 0; i_82622 < (int64_t) 16; i_82622++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73435 = ((double *) mem_83650)[i_82626 * (int64_t) 16 + i_82622];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73436 = ((double *) mem_83370)[i_82626 * (int64_t) 16 + i_82622];
            
            // futhark/microgpt.fut:171:42-72
            
            double zp_res_73437 = zp_lhs_73435 + zp_rhs_73436;
            
            ((double *) mem_83671)[i_82622] = zp_res_73437;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83666, i_82626 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83671, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83682_cached_sizze_85498 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83682, &mem_83682_cached_sizze_85498, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83687_cached_sizze_85499 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83687, &mem_83687_cached_sizze_85499, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83694_cached_sizze_85500 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83694, &mem_83694_cached_sizze_85500, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82638 = 0; i_82638 < (int64_t) 16; i_82638++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82630 = 0; i_82630 < (int64_t) 16; i_82630++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73452 = ((double *) mem_83666)[i_82638 * (int64_t) 16 + i_82630];
            
            // futhark/microgpt.fut:172:65-96
            
            double zt_res_73453 = zt_lhs_73452 * zt_lhs_73452;
            
            ((double *) mem_83687)[i_82630] = zt_res_73453;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73455;
        double r_73457 = 0.0;
        
        for (int64_t i_73456 = 0; i_73456 < (int64_t) 16; i_73456++) {
            // futhark/microgpt.fut:173:35-43
            
            double lifted_lambda_res_73458 = ((double *) mem_83687)[i_73456];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73459 = r_73457 + lifted_lambda_res_73458;
            double r_tmp_85181 = zp_res_73459;
            
            r_73457 = r_tmp_85181;
        }
        defunc_0_lifted_lambda_res_73455 = r_73457;
        // futhark/microgpt.fut:173:17-60
        
        double zs_res_73460 = defunc_0_lifted_lambda_res_73455 / 16.0;
        
        // futhark/microgpt.fut:174:24-55
        
        double zp_res_73461 = 1.0e-5 + zs_res_73460;
        
        // futhark/microgpt.fut:174:16-55
        
        double sqrt_res_73462 = futrts_sqrt64(zp_res_73461);
        
        // futhark/microgpt.fut:175:43-54
        
        double zs_res_73463 = 1.0 / sqrt_res_73462;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82634 = 0; i_82634 < (int64_t) 16; i_82634++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73470 = ((double *) mem_83666)[i_82638 * (int64_t) 16 + i_82634];
            
            // futhark/microgpt.fut:175:24-54
            
            double zt_res_73471 = zs_res_73463 * zt_lhs_73470;
            
            ((double *) mem_83694)[i_82634] = zt_res_73471;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83682, i_82638 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83705_cached_sizze_85501 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83705, &mem_83705_cached_sizze_85501, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83710_cached_sizze_85502 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83710, &mem_83710_cached_sizze_85502, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82646 = 0; i_82646 < (int64_t) 16; i_82646++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82642 = 0; i_82642 < (int64_t) 64; i_82642++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73487;
            double r_73489 = 0.0;
            
            for (int64_t i_73488 = 0; i_73488 < (int64_t) 16; i_73488++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73490 = ((double *) wup_mem_83333.mem)[i_82642 * (int64_t) 16 + i_73488];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73491 = ((double *) mem_83682)[i_82646 * (int64_t) 16 + i_73488];
                
                // futhark/microgpt.fut:176:73-104
                
                double zt_res_73492 = zt_lhs_73490 * zt_rhs_73491;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73493 = r_73489 + zt_res_73492;
                double r_tmp_85185 = zp_res_73493;
                
                r_73489 = r_tmp_85185;
            }
            defunc_0_lifted_lambda_res_73487 = r_73489;
            ((double *) mem_83710)[i_82642] = defunc_0_lifted_lambda_res_73487;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83705, i_82646 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83710, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83721_cached_sizze_85503 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83721, &mem_83721_cached_sizze_85503, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83726_cached_sizze_85504 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83726, &mem_83726_cached_sizze_85504, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82654 = 0; i_82654 < (int64_t) 16; i_82654++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82650 = 0; i_82650 < (int64_t) 64; i_82650++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_73508 = ((double *) mem_83705)[i_82654 * (int64_t) 64 + i_82650];
            
            // futhark/microgpt.fut:177:42-66
            
            double max_res_73509 = fmax64(0.0, max_arg0_73508);
            
            ((double *) mem_83726)[i_82650] = max_res_73509;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83721, i_82654 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83726, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83737_cached_sizze_85505 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83737, &mem_83737_cached_sizze_85505, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83742_cached_sizze_85506 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83742, &mem_83742_cached_sizze_85506, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82662 = 0; i_82662 < (int64_t) 16; i_82662++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82658 = 0; i_82658 < (int64_t) 16; i_82658++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73524;
            double r_73526 = 0.0;
            
            for (int64_t i_73525 = 0; i_73525 < (int64_t) 64; i_73525++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73527 = ((double *) wdown_mem_83327.mem)[i_82658 * (int64_t) 64 + i_73525];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73528 = ((double *) mem_83721)[i_82662 * (int64_t) 64 + i_73525];
                
                // futhark/microgpt.fut:178:73-106
                
                double zt_res_73529 = zt_lhs_73527 * zt_rhs_73528;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73530 = r_73526 + zt_res_73529;
                double r_tmp_85190 = zp_res_73530;
                
                r_73526 = r_tmp_85190;
            }
            defunc_0_lifted_lambda_res_73524 = r_73526;
            ((double *) mem_83742)[i_82658] = defunc_0_lifted_lambda_res_73524;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83737, i_82662 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83742, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83753_cached_sizze_85507 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83753, &mem_83753_cached_sizze_85507, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83758_cached_sizze_85508 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83758, &mem_83758_cached_sizze_85508, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82670 = 0; i_82670 < (int64_t) 16; i_82670++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82666 = 0; i_82666 < (int64_t) 16; i_82666++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73545 = ((double *) mem_83737)[i_82670 * (int64_t) 16 + i_82666];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73546 = ((double *) mem_83666)[i_82670 * (int64_t) 16 + i_82666];
            
            // futhark/microgpt.fut:179:42-73
            
            double zp_res_73547 = zp_lhs_73545 + zp_rhs_73546;
            
            ((double *) mem_83758)[i_82666] = zp_res_73547;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83753, i_82670 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83758, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83769, (int64_t) 3456, "mem_83769")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83774_cached_sizze_85509 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83774, &mem_83774_cached_sizze_85509, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82678 = 0; i_82678 < (int64_t) 16; i_82678++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82674 = 0; i_82674 < (int64_t) 27; i_82674++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73563;
            double r_73565 = 0.0;
            
            for (int64_t i_73564 = 0; i_73564 < (int64_t) 16; i_73564++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73566 = ((double *) wvoc_mem_83335.mem)[i_82674 * (int64_t) 16 + i_73564];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73567 = ((double *) mem_83753)[i_82678 * (int64_t) 16 + i_73564];
                
                // futhark/microgpt.fut:180:62-94
                
                double zt_res_73568 = zt_lhs_73566 * zt_rhs_73567;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73569 = r_73565 + zt_res_73568;
                double r_tmp_85195 = zp_res_73569;
                
                r_73565 = r_tmp_85195;
            }
            defunc_0_lifted_lambda_res_73563 = r_73565;
            ((double *) mem_83774)[i_82674] = defunc_0_lifted_lambda_res_73563;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83769.mem, i_82678 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83774, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_85126, &mem_83769, "mem_83769") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85455, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83338);
        free(mem_83343);
        free(mem_83354);
        free(mem_83359);
        free(mem_83370);
        free(mem_83375);
        free(mem_83382);
        free(mem_83393);
        free(mem_83398);
        free(mem_83405);
        free(mem_83416);
        free(mem_83417);
        free(mem_83418);
        free(mem_83431);
        free(mem_83432);
        free(mem_83433);
        free(mem_83464);
        free(mem_83465);
        free(mem_83466);
        free(mem_83482);
        free(mem_83483);
        free(mem_83484);
        free(mem_83497);
        free(mem_83498);
        free(mem_83499);
        free(mem_83545);
        free(mem_83551);
        free(mem_83556);
        free(mem_83567);
        free(mem_83572);
        free(mem_83583);
        free(mem_83588);
        free(mem_83595);
        free(mem_83602);
        free(mem_83613);
        free(mem_83618);
        free(mem_83634);
        free(mem_83639);
        free(mem_83650);
        free(mem_83655);
        free(mem_83666);
        free(mem_83671);
        free(mem_83682);
        free(mem_83687);
        free(mem_83694);
        free(mem_83705);
        free(mem_83710);
        free(mem_83721);
        free(mem_83726);
        free(mem_83737);
        free(mem_83742);
        free(mem_83753);
        free(mem_83758);
        free(mem_83774);
        if (memblock_unref(ctx, &mem_83769, "mem_83769") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85510, struct memblock *mem_out_p_85511, struct memblock *mem_out_p_85512, struct memblock *mem_out_p_85513, struct memblock *mem_out_p_85514, struct memblock *mem_out_p_85515, struct memblock *mem_out_p_85516, struct memblock *mem_out_p_85517, struct memblock *mem_out_p_85518, struct memblock wte_mem_83327, struct memblock wpe_mem_83328, struct memblock wqry_mem_83329, struct memblock wkey_mem_83330, struct memblock wval_mem_83331, struct memblock wout_mem_83332, struct memblock wup_mem_83333, struct memblock wdown_mem_83334, struct memblock wvoc_mem_83335)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    if (memblock_set(ctx, &mem_out_85126, &wdown_mem_83334, "wdown_mem_83334") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &wkey_mem_83330, "wkey_mem_83330") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &wout_mem_83332, "wout_mem_83332") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85129, &wpe_mem_83328, "wpe_mem_83328") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85130, &wqry_mem_83329, "wqry_mem_83329") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85131, &wte_mem_83327, "wte_mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85132, &wup_mem_83333, "wup_mem_83333") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85133, &wval_mem_83331, "wval_mem_83331") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85134, &wvoc_mem_83335, "wvoc_mem_83335") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85510, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85511, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85512, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85513, &mem_out_85129, "mem_out_85129") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85514, &mem_out_85130, "mem_out_85130") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85515, &mem_out_85131, "mem_out_85131") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85516, &mem_out_85132, "mem_out_85132") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85517, &mem_out_85133, "mem_out_85133") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85518, &mem_out_85134, "mem_out_85134") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85134, "mem_out_85134") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85133, "mem_out_85133") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85132, "mem_out_85132") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85131, "mem_out_85131") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85130, "mem_out_85130") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85129, "mem_out_85129") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85519, struct memblock *mem_out_p_85520, struct memblock *mem_out_p_85521, struct memblock *mem_out_p_85522, struct memblock *mem_out_p_85523, struct memblock *mem_out_p_85524, struct memblock *mem_out_p_85525, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock *mem_out_p_85540, struct memblock *mem_out_p_85541, struct memblock *mem_out_p_85542, struct memblock *mem_out_p_85543, struct memblock *mem_out_p_85544, struct memblock *mem_out_p_85545, struct memblock wdown_mem_83327, struct memblock wkey_mem_83328, struct memblock wout_mem_83329, struct memblock wpe_mem_83330, struct memblock wqry_mem_83331, struct memblock wte_mem_83332, struct memblock wup_mem_83333, struct memblock wval_mem_83334, struct memblock wvoc_mem_83335, struct memblock wdown_mem_83336, struct memblock wkey_mem_83337, struct memblock wout_mem_83338, struct memblock wpe_mem_83339, struct memblock wqry_mem_83340, struct memblock wte_mem_83341, struct memblock wup_mem_83342, struct memblock wval_mem_83343, struct memblock wvoc_mem_83344, struct memblock wdown_mem_83345, struct memblock wkey_mem_83346, struct memblock wout_mem_83347, struct memblock wpe_mem_83348, struct memblock wqry_mem_83349, struct memblock wte_mem_83350, struct memblock wup_mem_83351, struct memblock wval_mem_83352, struct memblock wvoc_mem_83353, struct memblock masks_mem_83354, struct memblock dls_mem_83355, struct memblock seqs_mem_83356, int64_t num_steps_62012)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83465_cached_sizze_85546 = 0;
    unsigned char *mem_83465 = NULL;
    int64_t mem_83466_cached_sizze_85547 = 0;
    unsigned char *mem_83466 = NULL;
    int64_t mem_83475_cached_sizze_85548 = 0;
    unsigned char *mem_83475 = NULL;
    int64_t mem_83482_cached_sizze_85549 = 0;
    unsigned char *mem_83482 = NULL;
    int64_t mem_83497_cached_sizze_85550 = 0;
    unsigned char *mem_83497 = NULL;
    int64_t mem_83498_cached_sizze_85551 = 0;
    unsigned char *mem_83498 = NULL;
    int64_t mem_83507_cached_sizze_85552 = 0;
    unsigned char *mem_83507 = NULL;
    int64_t mem_83514_cached_sizze_85553 = 0;
    unsigned char *mem_83514 = NULL;
    int64_t mem_83529_cached_sizze_85554 = 0;
    unsigned char *mem_83529 = NULL;
    int64_t mem_83530_cached_sizze_85555 = 0;
    unsigned char *mem_83530 = NULL;
    int64_t mem_83539_cached_sizze_85556 = 0;
    unsigned char *mem_83539 = NULL;
    int64_t mem_83540_cached_sizze_85557 = 0;
    unsigned char *mem_83540 = NULL;
    int64_t mem_83561_cached_sizze_85558 = 0;
    unsigned char *mem_83561 = NULL;
    int64_t mem_83562_cached_sizze_85559 = 0;
    unsigned char *mem_83562 = NULL;
    int64_t mem_83563_cached_sizze_85560 = 0;
    unsigned char *mem_83563 = NULL;
    int64_t mem_83575_cached_sizze_85561 = 0;
    unsigned char *mem_83575 = NULL;
    int64_t mem_83576_cached_sizze_85562 = 0;
    unsigned char *mem_83576 = NULL;
    int64_t mem_83600_cached_sizze_85563 = 0;
    unsigned char *mem_83600 = NULL;
    int64_t mem_83601_cached_sizze_85564 = 0;
    unsigned char *mem_83601 = NULL;
    int64_t mem_83602_cached_sizze_85565 = 0;
    unsigned char *mem_83602 = NULL;
    int64_t mem_83603_cached_sizze_85566 = 0;
    unsigned char *mem_83603 = NULL;
    int64_t mem_83604_cached_sizze_85567 = 0;
    unsigned char *mem_83604 = NULL;
    int64_t mem_83623_cached_sizze_85568 = 0;
    unsigned char *mem_83623 = NULL;
    int64_t mem_83624_cached_sizze_85569 = 0;
    unsigned char *mem_83624 = NULL;
    int64_t mem_83625_cached_sizze_85570 = 0;
    unsigned char *mem_83625 = NULL;
    int64_t mem_83662_cached_sizze_85571 = 0;
    unsigned char *mem_83662 = NULL;
    int64_t mem_83663_cached_sizze_85572 = 0;
    unsigned char *mem_83663 = NULL;
    int64_t mem_83664_cached_sizze_85573 = 0;
    unsigned char *mem_83664 = NULL;
    int64_t mem_83680_cached_sizze_85574 = 0;
    unsigned char *mem_83680 = NULL;
    int64_t mem_83681_cached_sizze_85575 = 0;
    unsigned char *mem_83681 = NULL;
    int64_t mem_83682_cached_sizze_85576 = 0;
    unsigned char *mem_83682 = NULL;
    int64_t mem_83695_cached_sizze_85577 = 0;
    unsigned char *mem_83695 = NULL;
    int64_t mem_83696_cached_sizze_85578 = 0;
    unsigned char *mem_83696 = NULL;
    int64_t mem_83697_cached_sizze_85579 = 0;
    unsigned char *mem_83697 = NULL;
    int64_t mem_83743_cached_sizze_85580 = 0;
    unsigned char *mem_83743 = NULL;
    int64_t mem_83744_cached_sizze_85581 = 0;
    unsigned char *mem_83744 = NULL;
    int64_t mem_83755_cached_sizze_85582 = 0;
    unsigned char *mem_83755 = NULL;
    int64_t mem_83756_cached_sizze_85583 = 0;
    unsigned char *mem_83756 = NULL;
    int64_t mem_83765_cached_sizze_85584 = 0;
    unsigned char *mem_83765 = NULL;
    int64_t mem_83766_cached_sizze_85585 = 0;
    unsigned char *mem_83766 = NULL;
    int64_t mem_83787_cached_sizze_85586 = 0;
    unsigned char *mem_83787 = NULL;
    int64_t mem_83792_cached_sizze_85587 = 0;
    unsigned char *mem_83792 = NULL;
    int64_t mem_83803_cached_sizze_85588 = 0;
    unsigned char *mem_83803 = NULL;
    int64_t mem_83808_cached_sizze_85589 = 0;
    unsigned char *mem_83808 = NULL;
    int64_t mem_83815_cached_sizze_85590 = 0;
    unsigned char *mem_83815 = NULL;
    int64_t mem_83822_cached_sizze_85591 = 0;
    unsigned char *mem_83822 = NULL;
    int64_t mem_83833_cached_sizze_85592 = 0;
    unsigned char *mem_83833 = NULL;
    int64_t mem_83838_cached_sizze_85593 = 0;
    unsigned char *mem_83838 = NULL;
    int64_t mem_83859_cached_sizze_85594 = 0;
    unsigned char *mem_83859 = NULL;
    int64_t mem_83860_cached_sizze_85595 = 0;
    unsigned char *mem_83860 = NULL;
    int64_t mem_83868_cached_sizze_85596 = 0;
    unsigned char *mem_83868 = NULL;
    int64_t mem_83882_cached_sizze_85597 = 0;
    unsigned char *mem_83882 = NULL;
    int64_t mem_83887_cached_sizze_85598 = 0;
    unsigned char *mem_83887 = NULL;
    int64_t mem_83898_cached_sizze_85599 = 0;
    unsigned char *mem_83898 = NULL;
    int64_t mem_83903_cached_sizze_85600 = 0;
    unsigned char *mem_83903 = NULL;
    int64_t mem_83914_cached_sizze_85601 = 0;
    unsigned char *mem_83914 = NULL;
    int64_t mem_83915_cached_sizze_85602 = 0;
    unsigned char *mem_83915 = NULL;
    int64_t mem_83924_cached_sizze_85603 = 0;
    unsigned char *mem_83924 = NULL;
    int64_t mem_83925_cached_sizze_85604 = 0;
    unsigned char *mem_83925 = NULL;
    int64_t mem_83946_cached_sizze_85605 = 0;
    unsigned char *mem_83946 = NULL;
    int64_t mem_83947_cached_sizze_85606 = 0;
    unsigned char *mem_83947 = NULL;
    int64_t mem_83955_cached_sizze_85607 = 0;
    unsigned char *mem_83955 = NULL;
    int64_t mem_83969_cached_sizze_85608 = 0;
    unsigned char *mem_83969 = NULL;
    int64_t mem_83970_cached_sizze_85609 = 0;
    unsigned char *mem_83970 = NULL;
    int64_t mem_83978_cached_sizze_85610 = 0;
    unsigned char *mem_83978 = NULL;
    int64_t mem_83992_cached_sizze_85611 = 0;
    unsigned char *mem_83992 = NULL;
    int64_t mem_83997_cached_sizze_85612 = 0;
    unsigned char *mem_83997 = NULL;
    int64_t mem_84008_cached_sizze_85613 = 0;
    unsigned char *mem_84008 = NULL;
    int64_t mem_84013_cached_sizze_85614 = 0;
    unsigned char *mem_84013 = NULL;
    int64_t mem_84024_cached_sizze_85615 = 0;
    unsigned char *mem_84024 = NULL;
    int64_t mem_84029_cached_sizze_85616 = 0;
    unsigned char *mem_84029 = NULL;
    int64_t mem_84040_cached_sizze_85617 = 0;
    unsigned char *mem_84040 = NULL;
    int64_t mem_84041_cached_sizze_85618 = 0;
    unsigned char *mem_84041 = NULL;
    int64_t mem_84050_cached_sizze_85619 = 0;
    unsigned char *mem_84050 = NULL;
    int64_t mem_84051_cached_sizze_85620 = 0;
    unsigned char *mem_84051 = NULL;
    int64_t mem_84064_cached_sizze_85621 = 0;
    unsigned char *mem_84064 = NULL;
    int64_t mem_84065_cached_sizze_85622 = 0;
    unsigned char *mem_84065 = NULL;
    int64_t mem_84078_cached_sizze_85623 = 0;
    unsigned char *mem_84078 = NULL;
    int64_t mem_84079_cached_sizze_85624 = 0;
    unsigned char *mem_84079 = NULL;
    int64_t mem_84100_cached_sizze_85625 = 0;
    unsigned char *mem_84100 = NULL;
    int64_t mem_84107_cached_sizze_85626 = 0;
    unsigned char *mem_84107 = NULL;
    int64_t mem_84112_cached_sizze_85627 = 0;
    unsigned char *mem_84112 = NULL;
    int64_t mem_84123_cached_sizze_85628 = 0;
    unsigned char *mem_84123 = NULL;
    int64_t mem_84128_cached_sizze_85629 = 0;
    unsigned char *mem_84128 = NULL;
    int64_t mem_84139_cached_sizze_85630 = 0;
    unsigned char *mem_84139 = NULL;
    int64_t mem_84140_cached_sizze_85631 = 0;
    unsigned char *mem_84140 = NULL;
    int64_t mem_84149_cached_sizze_85632 = 0;
    unsigned char *mem_84149 = NULL;
    int64_t mem_84150_cached_sizze_85633 = 0;
    unsigned char *mem_84150 = NULL;
    int64_t mem_84171_cached_sizze_85634 = 0;
    unsigned char *mem_84171 = NULL;
    int64_t mem_84176_cached_sizze_85635 = 0;
    unsigned char *mem_84176 = NULL;
    int64_t mem_84187_cached_sizze_85636 = 0;
    unsigned char *mem_84187 = NULL;
    int64_t mem_84192_cached_sizze_85637 = 0;
    unsigned char *mem_84192 = NULL;
    int64_t mem_84203_cached_sizze_85638 = 0;
    unsigned char *mem_84203 = NULL;
    int64_t mem_84210_cached_sizze_85639 = 0;
    unsigned char *mem_84210 = NULL;
    int64_t mem_84217_cached_sizze_85640 = 0;
    unsigned char *mem_84217 = NULL;
    int64_t mem_84227_cached_sizze_85641 = 0;
    unsigned char *mem_84227 = NULL;
    int64_t mem_84232_cached_sizze_85642 = 0;
    unsigned char *mem_84232 = NULL;
    int64_t mem_84243_cached_sizze_85643 = 0;
    unsigned char *mem_84243 = NULL;
    int64_t mem_84244_cached_sizze_85644 = 0;
    unsigned char *mem_84244 = NULL;
    int64_t mem_84253_cached_sizze_85645 = 0;
    unsigned char *mem_84253 = NULL;
    int64_t mem_84254_cached_sizze_85646 = 0;
    unsigned char *mem_84254 = NULL;
    int64_t mem_84275_cached_sizze_85647 = 0;
    unsigned char *mem_84275 = NULL;
    int64_t mem_84276_cached_sizze_85648 = 0;
    unsigned char *mem_84276 = NULL;
    int64_t mem_84287_cached_sizze_85649 = 0;
    unsigned char *mem_84287 = NULL;
    int64_t mem_84288_cached_sizze_85650 = 0;
    unsigned char *mem_84288 = NULL;
    int64_t mem_84297_cached_sizze_85651 = 0;
    unsigned char *mem_84297 = NULL;
    int64_t mem_84304_cached_sizze_85652 = 0;
    unsigned char *mem_84304 = NULL;
    int64_t mem_84329_cached_sizze_85653 = 0;
    unsigned char *mem_84329 = NULL;
    int64_t mem_84330_cached_sizze_85654 = 0;
    unsigned char *mem_84330 = NULL;
    int64_t mem_84341_cached_sizze_85655 = 0;
    unsigned char *mem_84341 = NULL;
    int64_t mem_84342_cached_sizze_85656 = 0;
    unsigned char *mem_84342 = NULL;
    int64_t mem_84351_cached_sizze_85657 = 0;
    unsigned char *mem_84351 = NULL;
    int64_t mem_84358_cached_sizze_85658 = 0;
    unsigned char *mem_84358 = NULL;
    int64_t mem_84365_cached_sizze_85659 = 0;
    unsigned char *mem_84365 = NULL;
    int64_t mem_84372_cached_sizze_85660 = 0;
    unsigned char *mem_84372 = NULL;
    int64_t mem_84397_cached_sizze_85661 = 0;
    unsigned char *mem_84397 = NULL;
    int64_t mem_84398_cached_sizze_85662 = 0;
    unsigned char *mem_84398 = NULL;
    int64_t mem_84409_cached_sizze_85663 = 0;
    unsigned char *mem_84409 = NULL;
    int64_t mem_84410_cached_sizze_85664 = 0;
    unsigned char *mem_84410 = NULL;
    int64_t mem_84419_cached_sizze_85665 = 0;
    unsigned char *mem_84419 = NULL;
    int64_t mem_84426_cached_sizze_85666 = 0;
    unsigned char *mem_84426 = NULL;
    int64_t mem_84451_cached_sizze_85667 = 0;
    unsigned char *mem_84451 = NULL;
    int64_t mem_84456_cached_sizze_85668 = 0;
    unsigned char *mem_84456 = NULL;
    int64_t mem_84467_cached_sizze_85669 = 0;
    unsigned char *mem_84467 = NULL;
    int64_t mem_84473_cached_sizze_85670 = 0;
    unsigned char *mem_84473 = NULL;
    int64_t mem_84478_cached_sizze_85671 = 0;
    unsigned char *mem_84478 = NULL;
    int64_t mem_84494_cached_sizze_85672 = 0;
    unsigned char *mem_84494 = NULL;
    int64_t mem_84500_cached_sizze_85673 = 0;
    unsigned char *mem_84500 = NULL;
    int64_t mem_84505_cached_sizze_85674 = 0;
    unsigned char *mem_84505 = NULL;
    int64_t mem_84521_cached_sizze_85675 = 0;
    unsigned char *mem_84521 = NULL;
    int64_t mem_84522_cached_sizze_85676 = 0;
    unsigned char *mem_84522 = NULL;
    int64_t mem_84533_cached_sizze_85677 = 0;
    unsigned char *mem_84533 = NULL;
    int64_t mem_84534_cached_sizze_85678 = 0;
    unsigned char *mem_84534 = NULL;
    int64_t mem_84543_cached_sizze_85679 = 0;
    unsigned char *mem_84543 = NULL;
    int64_t mem_84544_cached_sizze_85680 = 0;
    unsigned char *mem_84544 = NULL;
    int64_t mem_84575_cached_sizze_85681 = 0;
    unsigned char *mem_84575 = NULL;
    int64_t mem_84576_cached_sizze_85682 = 0;
    unsigned char *mem_84576 = NULL;
    int64_t mem_84577_cached_sizze_85683 = 0;
    unsigned char *mem_84577 = NULL;
    int64_t mem_84590_cached_sizze_85684 = 0;
    unsigned char *mem_84590 = NULL;
    int64_t mem_84591_cached_sizze_85685 = 0;
    unsigned char *mem_84591 = NULL;
    int64_t mem_84592_cached_sizze_85686 = 0;
    unsigned char *mem_84592 = NULL;
    int64_t mem_84623_cached_sizze_85687 = 0;
    unsigned char *mem_84623 = NULL;
    int64_t mem_84624_cached_sizze_85688 = 0;
    unsigned char *mem_84624 = NULL;
    int64_t mem_84625_cached_sizze_85689 = 0;
    unsigned char *mem_84625 = NULL;
    int64_t mem_84626_cached_sizze_85690 = 0;
    unsigned char *mem_84626 = NULL;
    int64_t mem_84643_cached_sizze_85691 = 0;
    unsigned char *mem_84643 = NULL;
    int64_t mem_84644_cached_sizze_85692 = 0;
    unsigned char *mem_84644 = NULL;
    int64_t mem_84645_cached_sizze_85693 = 0;
    unsigned char *mem_84645 = NULL;
    int64_t mem_84646_cached_sizze_85694 = 0;
    unsigned char *mem_84646 = NULL;
    int64_t mem_84687_cached_sizze_85695 = 0;
    unsigned char *mem_84687 = NULL;
    int64_t mem_84694_cached_sizze_85696 = 0;
    unsigned char *mem_84694 = NULL;
    int64_t mem_84701_cached_sizze_85697 = 0;
    unsigned char *mem_84701 = NULL;
    int64_t mem_84711_cached_sizze_85698 = 0;
    unsigned char *mem_84711 = NULL;
    int64_t mem_84716_cached_sizze_85699 = 0;
    unsigned char *mem_84716 = NULL;
    int64_t mem_84727_cached_sizze_85700 = 0;
    unsigned char *mem_84727 = NULL;
    int64_t mem_84734_cached_sizze_85701 = 0;
    unsigned char *mem_84734 = NULL;
    int64_t mem_84741_cached_sizze_85702 = 0;
    unsigned char *mem_84741 = NULL;
    int64_t mem_84751_cached_sizze_85703 = 0;
    unsigned char *mem_84751 = NULL;
    int64_t mem_84756_cached_sizze_85704 = 0;
    unsigned char *mem_84756 = NULL;
    int64_t mem_84767_cached_sizze_85705 = 0;
    unsigned char *mem_84767 = NULL;
    int64_t mem_84768_cached_sizze_85706 = 0;
    unsigned char *mem_84768 = NULL;
    int64_t mem_84777_cached_sizze_85707 = 0;
    unsigned char *mem_84777 = NULL;
    int64_t mem_84778_cached_sizze_85708 = 0;
    unsigned char *mem_84778 = NULL;
    int64_t mem_84799_cached_sizze_85709 = 0;
    unsigned char *mem_84799 = NULL;
    int64_t mem_84804_cached_sizze_85710 = 0;
    unsigned char *mem_84804 = NULL;
    int64_t mem_84815_cached_sizze_85711 = 0;
    unsigned char *mem_84815 = NULL;
    int64_t mem_84816_cached_sizze_85712 = 0;
    unsigned char *mem_84816 = NULL;
    int64_t mem_84825_cached_sizze_85713 = 0;
    unsigned char *mem_84825 = NULL;
    int64_t mem_84826_cached_sizze_85714 = 0;
    unsigned char *mem_84826 = NULL;
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
    
    struct memblock mem_param_tmp_85168;
    
    mem_param_tmp_85168.references = NULL;
    
    struct memblock mem_param_tmp_85167;
    
    mem_param_tmp_85167.references = NULL;
    
    struct memblock mem_param_tmp_85166;
    
    mem_param_tmp_85166.references = NULL;
    
    struct memblock mem_param_tmp_85165;
    
    mem_param_tmp_85165.references = NULL;
    
    struct memblock mem_param_tmp_85164;
    
    mem_param_tmp_85164.references = NULL;
    
    struct memblock mem_param_tmp_85163;
    
    mem_param_tmp_85163.references = NULL;
    
    struct memblock mem_param_tmp_85162;
    
    mem_param_tmp_85162.references = NULL;
    
    struct memblock mem_param_tmp_85161;
    
    mem_param_tmp_85161.references = NULL;
    
    struct memblock mem_param_tmp_85160;
    
    mem_param_tmp_85160.references = NULL;
    
    struct memblock mem_param_tmp_85159;
    
    mem_param_tmp_85159.references = NULL;
    
    struct memblock mem_param_tmp_85158;
    
    mem_param_tmp_85158.references = NULL;
    
    struct memblock mem_param_tmp_85157;
    
    mem_param_tmp_85157.references = NULL;
    
    struct memblock mem_param_tmp_85156;
    
    mem_param_tmp_85156.references = NULL;
    
    struct memblock mem_param_tmp_85155;
    
    mem_param_tmp_85155.references = NULL;
    
    struct memblock mem_param_tmp_85154;
    
    mem_param_tmp_85154.references = NULL;
    
    struct memblock mem_param_tmp_85153;
    
    mem_param_tmp_85153.references = NULL;
    
    struct memblock ext_mem_84943;
    
    ext_mem_84943.references = NULL;
    
    struct memblock ext_mem_84944;
    
    ext_mem_84944.references = NULL;
    
    struct memblock ext_mem_84945;
    
    ext_mem_84945.references = NULL;
    
    struct memblock mem_84941;
    
    mem_84941.references = NULL;
    
    struct memblock mem_84939;
    
    mem_84939.references = NULL;
    
    struct memblock mem_84937;
    
    mem_84937.references = NULL;
    
    struct memblock mem_84935;
    
    mem_84935.references = NULL;
    
    struct memblock ext_mem_84932;
    
    ext_mem_84932.references = NULL;
    
    struct memblock ext_mem_84933;
    
    ext_mem_84933.references = NULL;
    
    struct memblock ext_mem_84934;
    
    ext_mem_84934.references = NULL;
    
    struct memblock mem_84930;
    
    mem_84930.references = NULL;
    
    struct memblock mem_84928;
    
    mem_84928.references = NULL;
    
    struct memblock mem_84926;
    
    mem_84926.references = NULL;
    
    struct memblock mem_84924;
    
    mem_84924.references = NULL;
    
    struct memblock ext_mem_84921;
    
    ext_mem_84921.references = NULL;
    
    struct memblock ext_mem_84922;
    
    ext_mem_84922.references = NULL;
    
    struct memblock ext_mem_84923;
    
    ext_mem_84923.references = NULL;
    
    struct memblock mem_84919;
    
    mem_84919.references = NULL;
    
    struct memblock mem_84917;
    
    mem_84917.references = NULL;
    
    struct memblock mem_84915;
    
    mem_84915.references = NULL;
    
    struct memblock mem_84913;
    
    mem_84913.references = NULL;
    
    struct memblock ext_mem_84910;
    
    ext_mem_84910.references = NULL;
    
    struct memblock ext_mem_84911;
    
    ext_mem_84911.references = NULL;
    
    struct memblock ext_mem_84912;
    
    ext_mem_84912.references = NULL;
    
    struct memblock mem_84908;
    
    mem_84908.references = NULL;
    
    struct memblock mem_84906;
    
    mem_84906.references = NULL;
    
    struct memblock mem_84904;
    
    mem_84904.references = NULL;
    
    struct memblock mem_84902;
    
    mem_84902.references = NULL;
    
    struct memblock ext_mem_84899;
    
    ext_mem_84899.references = NULL;
    
    struct memblock ext_mem_84900;
    
    ext_mem_84900.references = NULL;
    
    struct memblock ext_mem_84901;
    
    ext_mem_84901.references = NULL;
    
    struct memblock mem_84897;
    
    mem_84897.references = NULL;
    
    struct memblock mem_84895;
    
    mem_84895.references = NULL;
    
    struct memblock mem_84893;
    
    mem_84893.references = NULL;
    
    struct memblock mem_84891;
    
    mem_84891.references = NULL;
    
    struct memblock ext_mem_84888;
    
    ext_mem_84888.references = NULL;
    
    struct memblock ext_mem_84889;
    
    ext_mem_84889.references = NULL;
    
    struct memblock ext_mem_84890;
    
    ext_mem_84890.references = NULL;
    
    struct memblock mem_84886;
    
    mem_84886.references = NULL;
    
    struct memblock mem_84884;
    
    mem_84884.references = NULL;
    
    struct memblock mem_84882;
    
    mem_84882.references = NULL;
    
    struct memblock mem_84880;
    
    mem_84880.references = NULL;
    
    struct memblock ext_mem_84877;
    
    ext_mem_84877.references = NULL;
    
    struct memblock ext_mem_84878;
    
    ext_mem_84878.references = NULL;
    
    struct memblock ext_mem_84879;
    
    ext_mem_84879.references = NULL;
    
    struct memblock mem_84875;
    
    mem_84875.references = NULL;
    
    struct memblock mem_84873;
    
    mem_84873.references = NULL;
    
    struct memblock mem_84871;
    
    mem_84871.references = NULL;
    
    struct memblock mem_84869;
    
    mem_84869.references = NULL;
    
    struct memblock ext_mem_84866;
    
    ext_mem_84866.references = NULL;
    
    struct memblock ext_mem_84867;
    
    ext_mem_84867.references = NULL;
    
    struct memblock ext_mem_84868;
    
    ext_mem_84868.references = NULL;
    
    struct memblock mem_84864;
    
    mem_84864.references = NULL;
    
    struct memblock mem_84862;
    
    mem_84862.references = NULL;
    
    struct memblock mem_84860;
    
    mem_84860.references = NULL;
    
    struct memblock mem_84858;
    
    mem_84858.references = NULL;
    
    struct memblock ext_mem_84855;
    
    ext_mem_84855.references = NULL;
    
    struct memblock ext_mem_84856;
    
    ext_mem_84856.references = NULL;
    
    struct memblock ext_mem_84857;
    
    ext_mem_84857.references = NULL;
    
    struct memblock mem_84853;
    
    mem_84853.references = NULL;
    
    struct memblock mem_84851;
    
    mem_84851.references = NULL;
    
    struct memblock mem_84849;
    
    mem_84849.references = NULL;
    
    struct memblock mem_84847;
    
    mem_84847.references = NULL;
    
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
    
    struct memblock mem_param_83372;
    
    mem_param_83372.references = NULL;
    
    struct memblock mem_param_83368;
    
    mem_param_83368.references = NULL;
    
    struct memblock mem_param_83364;
    
    mem_param_83364.references = NULL;
    
    struct memblock mem_param_83360;
    
    mem_param_83360.references = NULL;
    
    struct memblock ext_mem_85027;
    
    ext_mem_85027.references = NULL;
    
    struct memblock ext_mem_85028;
    
    ext_mem_85028.references = NULL;
    
    struct memblock ext_mem_85029;
    
    ext_mem_85029.references = NULL;
    
    struct memblock ext_mem_85030;
    
    ext_mem_85030.references = NULL;
    
    struct memblock ext_mem_85031;
    
    ext_mem_85031.references = NULL;
    
    struct memblock ext_mem_85032;
    
    ext_mem_85032.references = NULL;
    
    struct memblock ext_mem_85033;
    
    ext_mem_85033.references = NULL;
    
    struct memblock ext_mem_85034;
    
    ext_mem_85034.references = NULL;
    
    struct memblock ext_mem_85035;
    
    ext_mem_85035.references = NULL;
    
    struct memblock ext_mem_85036;
    
    ext_mem_85036.references = NULL;
    
    struct memblock ext_mem_85037;
    
    ext_mem_85037.references = NULL;
    
    struct memblock ext_mem_85038;
    
    ext_mem_85038.references = NULL;
    
    struct memblock ext_mem_85039;
    
    ext_mem_85039.references = NULL;
    
    struct memblock ext_mem_85040;
    
    ext_mem_85040.references = NULL;
    
    struct memblock ext_mem_85041;
    
    ext_mem_85041.references = NULL;
    
    struct memblock ext_mem_85042;
    
    ext_mem_85042.references = NULL;
    
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
    
    struct memblock mem_out_85141;
    
    mem_out_85141.references = NULL;
    
    struct memblock mem_out_85140;
    
    mem_out_85140.references = NULL;
    
    struct memblock mem_out_85139;
    
    mem_out_85139.references = NULL;
    
    struct memblock mem_out_85138;
    
    mem_out_85138.references = NULL;
    
    struct memblock mem_out_85137;
    
    mem_out_85137.references = NULL;
    
    struct memblock mem_out_85136;
    
    mem_out_85136.references = NULL;
    
    struct memblock mem_out_85135;
    
    mem_out_85135.references = NULL;
    
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_76653 = sitofp_i64_f64(num_steps_62012);
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83465_cached_sizze_85546 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83465, &mem_83465_cached_sizze_85546, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83466_cached_sizze_85547 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83466, &mem_83466_cached_sizze_85547, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83475_cached_sizze_85548 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83475, &mem_83475_cached_sizze_85548, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83482_cached_sizze_85549 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83482, &mem_83482_cached_sizze_85549, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83497_cached_sizze_85550 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83497, &mem_83497_cached_sizze_85550, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83498_cached_sizze_85551 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83498, &mem_83498_cached_sizze_85551, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83507_cached_sizze_85552 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83507, &mem_83507_cached_sizze_85552, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83514_cached_sizze_85553 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83514, &mem_83514_cached_sizze_85553, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83529_cached_sizze_85554 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83529, &mem_83529_cached_sizze_85554, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83530_cached_sizze_85555 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83530, &mem_83530_cached_sizze_85555, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83539_cached_sizze_85556 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83539, &mem_83539_cached_sizze_85556, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83540_cached_sizze_85557 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83540, &mem_83540_cached_sizze_85557, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83561_cached_sizze_85558 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83561, &mem_83561_cached_sizze_85558, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83562_cached_sizze_85559 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83562, &mem_83562_cached_sizze_85559, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83563_cached_sizze_85560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83563, &mem_83563_cached_sizze_85560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83575_cached_sizze_85561 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83575, &mem_83575_cached_sizze_85561, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83576_cached_sizze_85562 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83576, &mem_83576_cached_sizze_85562, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83600_cached_sizze_85563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83600, &mem_83600_cached_sizze_85563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83601_cached_sizze_85564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83601, &mem_83601_cached_sizze_85564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83602_cached_sizze_85565 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83602, &mem_83602_cached_sizze_85565, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83603_cached_sizze_85566 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83603, &mem_83603_cached_sizze_85566, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83604_cached_sizze_85567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83604, &mem_83604_cached_sizze_85567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83623_cached_sizze_85568 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83623, &mem_83623_cached_sizze_85568, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83624_cached_sizze_85569 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83624, &mem_83624_cached_sizze_85569, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83625_cached_sizze_85570 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83625, &mem_83625_cached_sizze_85570, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83662_cached_sizze_85571 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83662, &mem_83662_cached_sizze_85571, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83663_cached_sizze_85572 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83663, &mem_83663_cached_sizze_85572, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83664_cached_sizze_85573 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83664, &mem_83664_cached_sizze_85573, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83680_cached_sizze_85574 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83680, &mem_83680_cached_sizze_85574, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83681_cached_sizze_85575 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83681, &mem_83681_cached_sizze_85575, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83682_cached_sizze_85576 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83682, &mem_83682_cached_sizze_85576, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83695_cached_sizze_85577 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83695, &mem_83695_cached_sizze_85577, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83696_cached_sizze_85578 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83696, &mem_83696_cached_sizze_85578, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83697_cached_sizze_85579 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83697, &mem_83697_cached_sizze_85579, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83743_cached_sizze_85580 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83743, &mem_83743_cached_sizze_85580, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83744_cached_sizze_85581 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83744, &mem_83744_cached_sizze_85581, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83755_cached_sizze_85582 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83755, &mem_83755_cached_sizze_85582, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83756_cached_sizze_85583 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83756, &mem_83756_cached_sizze_85583, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83765_cached_sizze_85584 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83765, &mem_83765_cached_sizze_85584, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83766_cached_sizze_85585 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83766, &mem_83766_cached_sizze_85585, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83787_cached_sizze_85586 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83787, &mem_83787_cached_sizze_85586, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83792_cached_sizze_85587 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83792, &mem_83792_cached_sizze_85587, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83803_cached_sizze_85588 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83803, &mem_83803_cached_sizze_85588, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83808_cached_sizze_85589 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83808, &mem_83808_cached_sizze_85589, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83815_cached_sizze_85590 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83815, &mem_83815_cached_sizze_85590, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83822_cached_sizze_85591 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83822, &mem_83822_cached_sizze_85591, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83833_cached_sizze_85592 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83833, &mem_83833_cached_sizze_85592, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83838_cached_sizze_85593 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83838, &mem_83838_cached_sizze_85593, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83859_cached_sizze_85594 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83859, &mem_83859_cached_sizze_85594, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83860_cached_sizze_85595 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83860, &mem_83860_cached_sizze_85595, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83868_cached_sizze_85596 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83868, &mem_83868_cached_sizze_85596, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83882_cached_sizze_85597 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83882, &mem_83882_cached_sizze_85597, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83887_cached_sizze_85598 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83887, &mem_83887_cached_sizze_85598, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83898_cached_sizze_85599 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83898, &mem_83898_cached_sizze_85599, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83903_cached_sizze_85600 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83903, &mem_83903_cached_sizze_85600, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83914_cached_sizze_85601 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83914, &mem_83914_cached_sizze_85601, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83915_cached_sizze_85602 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83915, &mem_83915_cached_sizze_85602, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83924_cached_sizze_85603 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83924, &mem_83924_cached_sizze_85603, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83925_cached_sizze_85604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83925, &mem_83925_cached_sizze_85604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83946_cached_sizze_85605 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83946, &mem_83946_cached_sizze_85605, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83947_cached_sizze_85606 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83947, &mem_83947_cached_sizze_85606, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83955_cached_sizze_85607 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83955, &mem_83955_cached_sizze_85607, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83969_cached_sizze_85608 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83969, &mem_83969_cached_sizze_85608, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83970_cached_sizze_85609 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83970, &mem_83970_cached_sizze_85609, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83978_cached_sizze_85610 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83978, &mem_83978_cached_sizze_85610, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83992_cached_sizze_85611 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83992, &mem_83992_cached_sizze_85611, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83997_cached_sizze_85612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83997, &mem_83997_cached_sizze_85612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84008_cached_sizze_85613 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84008, &mem_84008_cached_sizze_85613, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84013_cached_sizze_85614 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84013, &mem_84013_cached_sizze_85614, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84024_cached_sizze_85615 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84024, &mem_84024_cached_sizze_85615, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84029_cached_sizze_85616 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84029, &mem_84029_cached_sizze_85616, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84040_cached_sizze_85617 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84040, &mem_84040_cached_sizze_85617, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84041_cached_sizze_85618 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84041, &mem_84041_cached_sizze_85618, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84050_cached_sizze_85619 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84050, &mem_84050_cached_sizze_85619, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84051_cached_sizze_85620 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84051, &mem_84051_cached_sizze_85620, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84064_cached_sizze_85621 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84064, &mem_84064_cached_sizze_85621, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84065_cached_sizze_85622 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84065, &mem_84065_cached_sizze_85622, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84078_cached_sizze_85623 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84078, &mem_84078_cached_sizze_85623, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84079_cached_sizze_85624 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84079, &mem_84079_cached_sizze_85624, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84100_cached_sizze_85625 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84100, &mem_84100_cached_sizze_85625, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84107_cached_sizze_85626 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84107, &mem_84107_cached_sizze_85626, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84112_cached_sizze_85627 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84112, &mem_84112_cached_sizze_85627, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84123_cached_sizze_85628 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84123, &mem_84123_cached_sizze_85628, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84128_cached_sizze_85629 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84128, &mem_84128_cached_sizze_85629, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84139_cached_sizze_85630 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84139, &mem_84139_cached_sizze_85630, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84140_cached_sizze_85631 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84140, &mem_84140_cached_sizze_85631, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84149_cached_sizze_85632 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84149, &mem_84149_cached_sizze_85632, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84150_cached_sizze_85633 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84150, &mem_84150_cached_sizze_85633, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84171_cached_sizze_85634 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84171, &mem_84171_cached_sizze_85634, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84176_cached_sizze_85635 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84176, &mem_84176_cached_sizze_85635, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84187_cached_sizze_85636 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84187, &mem_84187_cached_sizze_85636, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84192_cached_sizze_85637 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84192, &mem_84192_cached_sizze_85637, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84203_cached_sizze_85638 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84203, &mem_84203_cached_sizze_85638, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84210_cached_sizze_85639 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84210, &mem_84210_cached_sizze_85639, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84217_cached_sizze_85640 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84217, &mem_84217_cached_sizze_85640, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84227_cached_sizze_85641 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84227, &mem_84227_cached_sizze_85641, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84232_cached_sizze_85642 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84232, &mem_84232_cached_sizze_85642, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84243_cached_sizze_85643 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84243, &mem_84243_cached_sizze_85643, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84244_cached_sizze_85644 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84244, &mem_84244_cached_sizze_85644, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84253_cached_sizze_85645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84253, &mem_84253_cached_sizze_85645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84254_cached_sizze_85646 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84254, &mem_84254_cached_sizze_85646, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84275_cached_sizze_85647 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84275, &mem_84275_cached_sizze_85647, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84276_cached_sizze_85648 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84276, &mem_84276_cached_sizze_85648, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84287_cached_sizze_85649 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84287, &mem_84287_cached_sizze_85649, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84288_cached_sizze_85650 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84288, &mem_84288_cached_sizze_85650, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84297_cached_sizze_85651 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84297, &mem_84297_cached_sizze_85651, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84304_cached_sizze_85652 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84304, &mem_84304_cached_sizze_85652, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84329_cached_sizze_85653 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84329, &mem_84329_cached_sizze_85653, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84330_cached_sizze_85654 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84330, &mem_84330_cached_sizze_85654, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84341_cached_sizze_85655 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84341, &mem_84341_cached_sizze_85655, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84342_cached_sizze_85656 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84342, &mem_84342_cached_sizze_85656, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84351_cached_sizze_85657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84351, &mem_84351_cached_sizze_85657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84358_cached_sizze_85658 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84358, &mem_84358_cached_sizze_85658, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84365_cached_sizze_85659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84365, &mem_84365_cached_sizze_85659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84372_cached_sizze_85660 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84372, &mem_84372_cached_sizze_85660, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84397_cached_sizze_85661 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84397, &mem_84397_cached_sizze_85661, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84398_cached_sizze_85662 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84398, &mem_84398_cached_sizze_85662, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84409_cached_sizze_85663 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84409, &mem_84409_cached_sizze_85663, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84410_cached_sizze_85664 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84410, &mem_84410_cached_sizze_85664, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84419_cached_sizze_85665 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84419, &mem_84419_cached_sizze_85665, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84426_cached_sizze_85666 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84426, &mem_84426_cached_sizze_85666, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84451_cached_sizze_85667 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84451, &mem_84451_cached_sizze_85667, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84456_cached_sizze_85668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84456, &mem_84456_cached_sizze_85668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84467_cached_sizze_85669 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84467, &mem_84467_cached_sizze_85669, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84473_cached_sizze_85670 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84473, &mem_84473_cached_sizze_85670, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84478_cached_sizze_85671 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84478, &mem_84478_cached_sizze_85671, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84494_cached_sizze_85672 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84494, &mem_84494_cached_sizze_85672, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84500_cached_sizze_85673 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84500, &mem_84500_cached_sizze_85673, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84505_cached_sizze_85674 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84505, &mem_84505_cached_sizze_85674, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84521_cached_sizze_85675 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84521, &mem_84521_cached_sizze_85675, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84522_cached_sizze_85676 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84522, &mem_84522_cached_sizze_85676, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84533_cached_sizze_85677 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84533, &mem_84533_cached_sizze_85677, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84534_cached_sizze_85678 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84534, &mem_84534_cached_sizze_85678, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84543_cached_sizze_85679 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84543, &mem_84543_cached_sizze_85679, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84544_cached_sizze_85680 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84544, &mem_84544_cached_sizze_85680, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84575_cached_sizze_85681 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84575, &mem_84575_cached_sizze_85681, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84576_cached_sizze_85682 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84576, &mem_84576_cached_sizze_85682, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84577_cached_sizze_85683 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84577, &mem_84577_cached_sizze_85683, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84590_cached_sizze_85684 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84590, &mem_84590_cached_sizze_85684, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84591_cached_sizze_85685 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84591, &mem_84591_cached_sizze_85685, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84592_cached_sizze_85686 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84592, &mem_84592_cached_sizze_85686, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84623_cached_sizze_85687 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84623, &mem_84623_cached_sizze_85687, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84624_cached_sizze_85688 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84624, &mem_84624_cached_sizze_85688, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84625_cached_sizze_85689 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84625, &mem_84625_cached_sizze_85689, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84626_cached_sizze_85690 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84626, &mem_84626_cached_sizze_85690, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84643_cached_sizze_85691 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84643, &mem_84643_cached_sizze_85691, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84644_cached_sizze_85692 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84644, &mem_84644_cached_sizze_85692, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84645_cached_sizze_85693 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84645, &mem_84645_cached_sizze_85693, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84646_cached_sizze_85694 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84646, &mem_84646_cached_sizze_85694, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84687_cached_sizze_85695 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84687, &mem_84687_cached_sizze_85695, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84694_cached_sizze_85696 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84694, &mem_84694_cached_sizze_85696, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84701_cached_sizze_85697 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84701, &mem_84701_cached_sizze_85697, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84711_cached_sizze_85698 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84711, &mem_84711_cached_sizze_85698, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84716_cached_sizze_85699 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84716, &mem_84716_cached_sizze_85699, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84727_cached_sizze_85700 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84727, &mem_84727_cached_sizze_85700, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84734_cached_sizze_85701 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84734, &mem_84734_cached_sizze_85701, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84741_cached_sizze_85702 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84741, &mem_84741_cached_sizze_85702, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84751_cached_sizze_85703 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84751, &mem_84751_cached_sizze_85703, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84756_cached_sizze_85704 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84756, &mem_84756_cached_sizze_85704, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84767_cached_sizze_85705 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84767, &mem_84767_cached_sizze_85705, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84768_cached_sizze_85706 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84768, &mem_84768_cached_sizze_85706, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84777_cached_sizze_85707 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84777, &mem_84777_cached_sizze_85707, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84778_cached_sizze_85708 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84778, &mem_84778_cached_sizze_85708, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84799_cached_sizze_85709 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84799, &mem_84799_cached_sizze_85709, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84804_cached_sizze_85710 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84804, &mem_84804_cached_sizze_85710, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84815_cached_sizze_85711 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84815, &mem_84815_cached_sizze_85711, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84816_cached_sizze_85712 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84816, &mem_84816_cached_sizze_85712, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84825_cached_sizze_85713 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84825, &mem_84825_cached_sizze_85713, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84826_cached_sizze_85714 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84826, &mem_84826_cached_sizze_85714, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:436:5-441:61
    if (memblock_set(ctx, &mem_param_83360, &wdown_mem_83327, "wdown_mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83364, &wkey_mem_83328, "wkey_mem_83328") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83368, &wout_mem_83329, "wout_mem_83329") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83372, &wpe_mem_83330, "wpe_mem_83330") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83376, &wqry_mem_83331, "wqry_mem_83331") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83380, &wte_mem_83332, "wte_mem_83332") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83384, &wup_mem_83333, "wup_mem_83333") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83388, &wval_mem_83334, "wval_mem_83334") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83392, &wvoc_mem_83335, "wvoc_mem_83335") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83396, &wdown_mem_83336, "wdown_mem_83336") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83400, &wkey_mem_83337, "wkey_mem_83337") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83404, &wout_mem_83338, "wout_mem_83338") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83408, &wpe_mem_83339, "wpe_mem_83339") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83412, &wqry_mem_83340, "wqry_mem_83340") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83416, &wte_mem_83341, "wte_mem_83341") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83420, &wup_mem_83342, "wup_mem_83342") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83424, &wval_mem_83343, "wval_mem_83343") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83428, &wvoc_mem_83344, "wvoc_mem_83344") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83432, &wdown_mem_83345, "wdown_mem_83345") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83436, &wkey_mem_83346, "wkey_mem_83346") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83440, &wout_mem_83347, "wout_mem_83347") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83444, &wpe_mem_83348, "wpe_mem_83348") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83448, &wqry_mem_83349, "wqry_mem_83349") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83452, &wte_mem_83350, "wte_mem_83350") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83456, &wup_mem_83351, "wup_mem_83351") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83460, &wval_mem_83352, "wval_mem_83352") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83464, &wvoc_mem_83353, "wvoc_mem_83353") != 0)
        return 1;
    for (int64_t step_76681 = 0; step_76681 < num_steps_62012; step_76681++) {
        // futhark/microgpt.fut:438:16-25
        
        int64_t dl_76709 = ((int64_t *) dls_mem_83355.mem)[step_76681];
        
        // futhark/microgpt.fut:352:37-40
        
        int64_t zl_rhs_76710 = sub64(dl_76709, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82480 = 0; i_82480 < (int64_t) 16; i_82480++) {
            // futhark/microgpt.fut:352:25-81
            
            bool cond_78499 = slt64(i_82480, zl_rhs_76710);
            
            // futhark/microgpt.fut:352:56-59
            
            int64_t zeze_lhs_78500 = add64((int64_t) 1, i_82480);
            
            // futhark/microgpt.fut:352:47-60
            
            bool x_78501 = sle64((int64_t) 0, zeze_lhs_78500);
            
            // futhark/microgpt.fut:352:47-60
            
            bool y_78502 = slt64(zeze_lhs_78500, (int64_t) 16);
            
            // futhark/microgpt.fut:352:47-60
            
            bool bounds_check_78503 = x_78501 && y_78502;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_78504 = !cond_78499;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_78505 = bounds_check_78503 || loop_not_taken_78504;
            
            // futhark/microgpt.fut:352:47-60
            
            bool index_certs_78506;
            
            if (!protect_assert_disj_78505) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_78500, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:352:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:352:3-83\n   #6  futhark/microgpt.fut:410:18-38\n   #7  futhark/microgpt.fut:419:44-425:31\n   #8  futhark/microgpt.fut:441:11-60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_78521 = ((int64_t *) seqs_mem_83356.mem)[step_76681 * (int64_t) 16 + i_82480];
            
            // futhark/microgpt.fut:412:37-51
            
            bool x_78522 = sle64((int64_t) 0, tmp_78521);
            
            // futhark/microgpt.fut:412:37-51
            
            bool y_78523 = slt64(tmp_78521, (int64_t) 27);
            
            // futhark/microgpt.fut:412:37-51
            
            bool bounds_check_78524 = x_78522 && y_78523;
            
            // futhark/microgpt.fut:412:37-51
            
            bool index_certs_78525;
            
            if (!bounds_check_78524) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_78521, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:412:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:412:16-55\n   #6  futhark/microgpt.fut:419:44-425:31\n   #7  futhark/microgpt.fut:441:11-60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:352:47-60
            
            int64_t zeze_lhs_78507;
            
            if (cond_78499) {
                int64_t x_82289 = ((int64_t *) seqs_mem_83356.mem)[step_76681 * (int64_t) 16 + zeze_lhs_78500];
                
                zeze_lhs_78507 = x_82289;
            } else {
                zeze_lhs_78507 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82470 = 0; i_82470 < (int64_t) 27; i_82470++) {
                // futhark/microgpt.fut:352:61-65
                
                bool cond_t_res_78511 = zeze_lhs_78507 == i_82470;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_78512 = cond_78499 && cond_t_res_78511;
                
                // futhark/microgpt.fut:352:25-81
                
                double lifted_lambda_res_78513;
                
                if (x_78512) {
                    lifted_lambda_res_78513 = 1.0;
                } else {
                    lifted_lambda_res_78513 = 0.0;
                }
                ((double *) mem_83475)[i_82470] = lifted_lambda_res_78513;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82474 = 0; i_82474 < (int64_t) 16; i_82474++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_78532 = ((double *) mem_param_83380.mem)[tmp_78521 * (int64_t) 16 + i_82474];
                
                ((double *) mem_83482)[i_82474] = lifted_lambda_res_78532;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83465, i_82480 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83482, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83466, i_82480 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83475, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82495 = 0; i_82495 < (int64_t) 16; i_82495++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82485 = 0; i_82485 < (int64_t) 16; i_82485++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78557 = ((double *) mem_param_83372.mem)[i_82495 * (int64_t) 16 + i_82485];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_78558 = ((double *) mem_83465)[i_82495 * (int64_t) 16 + i_82485];
                
                // futhark/microgpt.fut:211:35-63
                
                double zp_res_78559 = zp_lhs_78557 + zp_rhs_78558;
                
                ((double *) mem_83507)[i_82485] = zp_res_78559;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82489 = 0; i_82489 < (int64_t) 27; i_82489++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78573 = ((double *) mem_83466)[i_82495 * (int64_t) 27 + i_82489];
                
                // futhark/microgpt.fut:243:51-87
                
                double zt_res_78574 = -6.25e-2 * zt_rhs_78573;
                
                ((double *) mem_83514)[i_82489] = zt_res_78574;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83497, i_82495 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83514, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83498, i_82495 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83507, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82509 = 0; i_82509 < (int64_t) 16; i_82509++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78593;
            double r_78595 = 0.0;
            
            for (int64_t i_78594 = 0; i_78594 < (int64_t) 16; i_78594++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78596 = ((double *) mem_83498)[i_82509 * (int64_t) 16 + i_78594];
                
                // futhark/microgpt.fut:212:58-83
                
                double zt_res_78597 = zt_lhs_78596 * zt_lhs_78596;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78598 = r_78595 + zt_res_78597;
                double r_tmp_85217 = zp_res_78598;
                
                r_78595 = r_tmp_85217;
            }
            defunc_0_lifted_lambda_res_78593 = r_78595;
            // futhark/microgpt.fut:212:40-101
            
            double zs_res_78599 = defunc_0_lifted_lambda_res_78593 / 16.0;
            
            // futhark/microgpt.fut:213:23-53
            
            double zp_res_78600 = 1.0e-5 + zs_res_78599;
            
            // futhark/microgpt.fut:213:15-53
            
            double sqrt_res_78601 = futrts_sqrt64(zp_res_78600);
            
            // futhark/microgpt.fut:214:39-49
            
            double zs_res_78602 = 1.0 / sqrt_res_78601;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82502 = 0; i_82502 < (int64_t) 16; i_82502++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80598 = ((double *) mem_83498)[i_82509 * (int64_t) 16 + i_82502];
                
                // futhark/microgpt.fut:214:23-49
                
                double zt_res_80599 = zs_res_78602 * zt_lhs_80598;
                
                // futhark/microgpt.fut:286:53-86
                
                double zt_res_80607 = zt_lhs_80598 * zt_lhs_80598;
                
                ((double *) mem_83539)[i_82502] = zt_res_80607;
                ((double *) mem_83540)[i_82502] = zt_res_80599;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83529, i_82509 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83539, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83530, i_82509 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83540, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82525 = 0; i_82525 < (int64_t) 16; i_82525++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78701;
            double r_78703 = 0.0;
            
            for (int64_t i_78702 = 0; i_78702 < (int64_t) 16; i_78702++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78704 = ((double *) mem_83530)[i_82525 * (int64_t) 16 + i_78702];
                
                // futhark/microgpt.fut:215:61-90
                
                double zt_res_78705 = zt_lhs_78704 * zt_lhs_78704;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78706 = r_78703 + zt_res_78705;
                double r_tmp_85223 = zp_res_78706;
                
                r_78703 = r_tmp_85223;
            }
            defunc_0_lifted_lambda_res_78701 = r_78703;
            // futhark/microgpt.fut:215:42-108
            
            double zs_res_78707 = defunc_0_lifted_lambda_res_78701 / 16.0;
            
            // futhark/microgpt.fut:216:24-55
            
            double zp_res_78708 = 1.0e-5 + zs_res_78707;
            
            // futhark/microgpt.fut:216:16-55
            
            double sqrt_res_78709 = futrts_sqrt64(zp_res_78708);
            
            // futhark/microgpt.fut:217:42-53
            
            double zs_res_78710 = 1.0 / sqrt_res_78709;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82516 = 0; i_82516 < (int64_t) 16; i_82516++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80627 = ((double *) mem_83530)[i_82525 * (int64_t) 16 + i_82516];
                
                // futhark/microgpt.fut:217:24-53
                
                double zt_res_80628 = zs_res_78710 * zt_lhs_80627;
                
                // futhark/microgpt.fut:279:53-86
                
                double zt_res_80636 = zt_lhs_80627 * zt_lhs_80627;
                
                ((double *) mem_83575)[i_82516] = zt_res_80636;
                ((double *) mem_83576)[i_82516] = zt_res_80628;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78744;
            double r_78746 = 0.0;
            
            for (int64_t i_78745 = 0; i_78745 < (int64_t) 16; i_78745++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_78747 = ((double *) mem_83529)[i_82525 * (int64_t) 16 + i_78745];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78748 = r_78746 + lifted_lambda_res_78747;
                double r_tmp_85226 = zp_res_78748;
                
                r_78746 = r_tmp_85226;
            }
            defunc_0_lifted_lambda_res_78744 = r_78746;
            // futhark/microgpt.fut:287:34-86
            
            double zs_res_78749 = defunc_0_lifted_lambda_res_78744 / 16.0;
            
            ((double *) mem_83561)[i_82525] = zs_res_78749;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83562, i_82525 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83575, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83563, i_82525 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83576, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82549 = 0; i_82549 < (int64_t) 16; i_82549++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82535 = 0; i_82535 < (int64_t) 16; i_82535++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80699;
                double r_80701 = 0.0;
                
                for (int64_t i_80700 = 0; i_80700 < (int64_t) 16; i_80700++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80702 = ((double *) mem_param_83376.mem)[i_82535 * (int64_t) 16 + i_80700];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80703 = ((double *) mem_83563)[i_82549 * (int64_t) 16 + i_80700];
                    
                    // futhark/microgpt.fut:218:69-100
                    
                    double zt_res_80704 = zt_lhs_80702 * zt_rhs_80703;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80705 = r_80701 + zt_res_80704;
                    double r_tmp_85235 = zp_res_80705;
                    
                    r_80701 = r_tmp_85235;
                }
                defunc_0_lifted_lambda_res_80699 = r_80701;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80712;
                double r_80714 = 0.0;
                
                for (int64_t i_80713 = 0; i_80713 < (int64_t) 16; i_80713++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80715 = ((double *) mem_param_83364.mem)[i_82535 * (int64_t) 16 + i_80713];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80716 = ((double *) mem_83563)[i_82549 * (int64_t) 16 + i_80713];
                    
                    // futhark/microgpt.fut:219:69-100
                    
                    double zt_res_80717 = zt_lhs_80715 * zt_rhs_80716;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80718 = r_80714 + zt_res_80717;
                    double r_tmp_85236 = zp_res_80718;
                    
                    r_80714 = r_tmp_85236;
                }
                defunc_0_lifted_lambda_res_80712 = r_80714;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80728;
                double r_80730 = 0.0;
                
                for (int64_t i_80729 = 0; i_80729 < (int64_t) 16; i_80729++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80731 = ((double *) mem_param_83388.mem)[i_82535 * (int64_t) 16 + i_80729];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80732 = ((double *) mem_83563)[i_82549 * (int64_t) 16 + i_80729];
                    
                    // futhark/microgpt.fut:220:69-100
                    
                    double zt_res_80733 = zt_lhs_80731 * zt_rhs_80732;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80734 = r_80730 + zt_res_80733;
                    double r_tmp_85237 = zp_res_80734;
                    
                    r_80730 = r_tmp_85237;
                }
                defunc_0_lifted_lambda_res_80728 = r_80730;
                ((double *) mem_83623)[i_82535] = defunc_0_lifted_lambda_res_80728;
                ((double *) mem_83624)[i_82535] = defunc_0_lifted_lambda_res_80712;
                ((double *) mem_83625)[i_82535] = defunc_0_lifted_lambda_res_80699;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79091;
            double r_79093 = 0.0;
            
            for (int64_t i_79092 = 0; i_79092 < (int64_t) 16; i_79092++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79094 = ((double *) mem_83562)[i_82549 * (int64_t) 16 + i_79092];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79095 = r_79093 + lifted_lambda_res_79094;
                double r_tmp_85238 = zp_res_79095;
                
                r_79093 = r_tmp_85238;
            }
            defunc_0_lifted_lambda_res_79091 = r_79093;
            // futhark/microgpt.fut:280:34-86
            
            double zs_res_79096 = defunc_0_lifted_lambda_res_79091 / 16.0;
            
            // futhark/microgpt.fut:288:41-51
            
            double zp_lhs_79110 = ((double *) mem_83561)[i_82549];
            
            // futhark/microgpt.fut:288:41-79
            
            double zp_res_79111 = 1.0e-5 + zp_lhs_79110;
            
            // futhark/microgpt.fut:288:33-79
            
            double sqrt_res_79112 = futrts_sqrt64(zp_res_79111);
            
            ((double *) mem_83600)[i_82549] = sqrt_res_79112;
            ((double *) mem_83601)[i_82549] = zs_res_79096;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83602, i_82549 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83623, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83603, i_82549 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83624, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83604, i_82549 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83625, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82581 = 0; i_82581 < (int64_t) 4; i_82581++) {
            // futhark/microgpt.fut:221:81-84
            
            int64_t zp_lhs_79184 = mul64((int64_t) 4, i_82581);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82571 = 0; i_82571 < (int64_t) 16; i_82571++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82561 = 0; i_82561 < (int64_t) 4; i_82561++) {
                    // futhark/microgpt.fut:221:86-91
                    
                    int64_t tmp_80892 = add64(zp_lhs_79184, i_82561);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool x_80893 = sle64((int64_t) 0, tmp_80892);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool y_80894 = slt64(tmp_80892, (int64_t) 16);
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool bounds_check_80895 = x_80893 && y_80894;
                    
                    // futhark/microgpt.fut:221:66-93
                    
                    bool index_certs_80896;
                    
                    if (!bounds_check_80895) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_80892, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:221:66-93\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:221:49-94\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:221:30-96\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:221:12-98\n   #10 futhark/microgpt.fut:415:5-76\n   #11 futhark/microgpt.fut:419:44-425:31\n   #12 futhark/microgpt.fut:441:11-60\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80897 = ((double *) mem_83604)[i_82571 * (int64_t) 16 + tmp_80892];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80905 = ((double *) mem_83603)[i_82571 * (int64_t) 16 + tmp_80892];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80916 = ((double *) mem_83602)[i_82571 * (int64_t) 16 + tmp_80892];
                    
                    ((double *) mem_83695)[i_82561] = lifted_lambda_res_80916;
                    ((double *) mem_83696)[i_82561] = lifted_lambda_res_80905;
                    ((double *) mem_83697)[i_82561] = lifted_lambda_res_80897;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83680, i_82571 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83681, i_82571 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83696, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83682, i_82571 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83697, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83662, i_82581 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83680, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83663, i_82581 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83681, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83664, i_82581 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83682, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82637 = 0; i_82637 < (int64_t) 4; i_82637++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82596 = 0; i_82596 < (int64_t) 16; i_82596++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82589 = 0; i_82589 < (int64_t) 16; i_82589++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_80995;
                    double r_80997 = 0.0;
                    
                    for (int64_t i_80996 = 0; i_80996 < (int64_t) 4; i_80996++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_80998 = ((double *) mem_83664)[i_82637 * (int64_t) 64 + i_82596 * (int64_t) 4 + i_80996];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_80999 = ((double *) mem_83663)[i_82637 * (int64_t) 64 + i_82589 * (int64_t) 4 + i_80996];
                        
                        // futhark/microgpt.fut:224:97-138
                        
                        double zt_res_81000 = zt_lhs_80998 * zt_rhs_80999;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81001 = r_80997 + zt_res_81000;
                        double r_tmp_85254 = zp_res_81001;
                        
                        r_80997 = r_tmp_85254;
                    }
                    defunc_0_lifted_lambda_res_80995 = r_80997;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81008;
                    double r_81010 = 0.0;
                    
                    for (int64_t i_81009 = 0; i_81009 < (int64_t) 4; i_81009++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81011 = ((double *) mem_83664)[i_82637 * (int64_t) 64 + i_82596 * (int64_t) 4 + i_81009];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81012 = ((double *) mem_83663)[i_82637 * (int64_t) 64 + i_82589 * (int64_t) 4 + i_81009];
                        
                        // futhark/microgpt.fut:263:91-138
                        
                        double zt_res_81013 = zt_lhs_81011 * zt_rhs_81012;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81014 = r_81010 + zt_res_81013;
                        double r_tmp_85255 = zp_res_81014;
                        
                        r_81010 = r_tmp_85255;
                    }
                    defunc_0_lifted_lambda_res_81008 = r_81010;
                    ((double *) mem_83765)[i_82589] = defunc_0_lifted_lambda_res_81008;
                    ((double *) mem_83766)[i_82589] = defunc_0_lifted_lambda_res_80995;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83755, i_82596 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83765, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83756, i_82596 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83766, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82605 = 0; i_82605 < (int64_t) 16; i_82605++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82601 = 0; i_82601 < (int64_t) 16; i_82601++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_79293 = ((double *) mem_83756)[i_82605 * (int64_t) 16 + i_82601];
                    
                    // futhark/microgpt.fut:225:43-70
                    
                    double zs_res_79294 = zs_lhs_79293 / 2.0;
                    double zp_rhs_79295 = ((double *) masks_mem_83354.mem)[step_76681 * (int64_t) 256 + i_82605 * (int64_t) 16 + i_82601];
                    
                    // futhark/microgpt.fut:225:57-90
                    
                    double zp_res_79296 = zs_res_79294 + zp_rhs_79295;
                    
                    ((double *) mem_83792)[i_82601] = zp_res_79296;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83787, i_82605 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83792, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82623 = 0; i_82623 < (int64_t) 16; i_82623++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_82310;
                double redout_82607 = -INFINITY;
                
                for (int64_t i_82608 = 0; i_82608 < (int64_t) 16; i_82608++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81032 = ((double *) mem_83787)[i_82623 * (int64_t) 16 + i_82608];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_79317 = fmax64(lifted_lambda_res_81032, redout_82607);
                    double redout_tmp_85259 = max_res_79317;
                    
                    redout_82607 = redout_tmp_85259;
                }
                defunc_0_reduce_res_82310 = redout_82607;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_79318 = -defunc_0_reduce_res_82310;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82611 = 0; i_82611 < (int64_t) 16; i_82611++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_79325 = ((double *) mem_83787)[i_82623 * (int64_t) 16 + i_82611];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_79326 = neg_res_79318 + lifted_lambda_res_79325;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_79327 = futrts_exp64(zp_res_79326);
                    
                    ((double *) mem_83808)[i_82611] = exp_res_79327;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79329;
                double r_79331 = 0.0;
                
                for (int64_t i_79330 = 0; i_79330 < (int64_t) 16; i_79330++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_79332 = ((double *) mem_83808)[i_79330];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79333 = r_79331 + lifted_lambda_res_79332;
                    double r_tmp_85261 = zp_res_79333;
                    
                    r_79331 = r_tmp_85261;
                }
                defunc_0_lifted_lambda_res_79329 = r_79331;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82615 = 0; i_82615 < (int64_t) 16; i_82615++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_79340 = ((double *) mem_83808)[i_82615];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_79341 = zs_lhs_79340 / defunc_0_lifted_lambda_res_79329;
                    
                    ((double *) mem_83815)[i_82615] = zs_res_79341;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82619 = 0; i_82619 < (int64_t) 16; i_82619++) {
                    // futhark/microgpt.fut:227:23-31
                    
                    double lifted_lambda_res_79349 = ((double *) mem_83815)[i_82619];
                    
                    ((double *) mem_83822)[i_82619] = lifted_lambda_res_79349;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83803, i_82623 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83822, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82631 = 0; i_82631 < (int64_t) 16; i_82631++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82627 = 0; i_82627 < (int64_t) 4; i_82627++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_79364;
                    double r_79366 = 0.0;
                    
                    for (int64_t i_79365 = 0; i_79365 < (int64_t) 16; i_79365++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_79367 = ((double *) mem_83803)[i_82631 * (int64_t) 16 + i_79365];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_79368 = ((double *) mem_83662)[i_82637 * (int64_t) 64 + i_79365 * (int64_t) 4 + i_82627];
                        
                        // futhark/microgpt.fut:228:61-97
                        
                        double zt_res_79369 = zt_lhs_79367 * zt_rhs_79368;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_79370 = r_79366 + zt_res_79369;
                        double r_tmp_85266 = zp_res_79370;
                        
                        r_79366 = r_tmp_85266;
                    }
                    defunc_0_lifted_lambda_res_79364 = r_79366;
                    ((double *) mem_83838)[i_82627] = defunc_0_lifted_lambda_res_79364;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83833, i_82631 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83743, i_82637 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_83755, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83744, i_82637 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83833, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82648 = 0; i_82648 < (int64_t) 16; i_82648++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82642 = 0; i_82642 < (int64_t) 16; i_82642++) {
                // futhark/microgpt.fut:229:58-61
                
                int64_t tmp_79419 = sdiv64(i_82642, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-63
                
                bool x_79420 = sle64((int64_t) 0, tmp_79419);
                
                // futhark/microgpt.fut:229:49-63
                
                bool y_79421 = slt64(tmp_79419, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-63
                
                bool bounds_check_79422 = x_79420 && y_79421;
                
                // futhark/microgpt.fut:229:49-63
                
                bool index_certs_79423;
                
                if (!bounds_check_79422) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79419, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:229:49-63\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:229:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:229:12-82\n   #7  futhark/microgpt.fut:415:5-76\n   #8  futhark/microgpt.fut:419:44-425:31\n   #9  futhark/microgpt.fut:441:11-60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:229:74-77
                
                int64_t tmp_79424 = smod64(i_82642, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-79
                
                bool x_79425 = sle64((int64_t) 0, tmp_79424);
                
                // futhark/microgpt.fut:229:49-79
                
                bool y_79426 = slt64(tmp_79424, (int64_t) 4);
                
                // futhark/microgpt.fut:229:49-79
                
                bool bounds_check_79427 = x_79425 && y_79426;
                
                // futhark/microgpt.fut:229:49-79
                
                bool index_certs_79428;
                
                if (!bounds_check_79427) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79424, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:229:49-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:229:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:229:12-82\n   #7  futhark/microgpt.fut:415:5-76\n   #8  futhark/microgpt.fut:419:44-425:31\n   #9  futhark/microgpt.fut:441:11-60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79429 = ((double *) mem_83744)[tmp_79419 * (int64_t) 64 + i_82648 * (int64_t) 4 + tmp_79424];
                
                ((double *) mem_83868)[i_82642] = lifted_lambda_res_79429;
            }
            // futhark/microgpt.fut:281:41-51
            
            double zp_lhs_79437 = ((double *) mem_83601)[i_82648];
            
            // futhark/microgpt.fut:281:41-79
            
            double zp_res_79438 = 1.0e-5 + zp_lhs_79437;
            
            // futhark/microgpt.fut:281:33-79
            
            double sqrt_res_79439 = futrts_sqrt64(zp_res_79438);
            
            ((double *) mem_83859)[i_82648] = sqrt_res_79439;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83860, i_82648 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83868, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82657 = 0; i_82657 < (int64_t) 16; i_82657++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82653 = 0; i_82653 < (int64_t) 16; i_82653++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77095;
                double r_77097 = 0.0;
                
                for (int64_t i_77096 = 0; i_77096 < (int64_t) 16; i_77096++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77098 = ((double *) mem_param_83368.mem)[i_82653 * (int64_t) 16 + i_77096];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77099 = ((double *) mem_83860)[i_82657 * (int64_t) 16 + i_77096];
                    
                    // futhark/microgpt.fut:230:69-101
                    
                    double zt_res_77100 = zt_lhs_77098 * zt_rhs_77099;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77101 = r_77097 + zt_res_77100;
                    double r_tmp_85272 = zp_res_77101;
                    
                    r_77097 = r_tmp_85272;
                }
                defunc_0_lifted_lambda_res_77095 = r_77097;
                ((double *) mem_83887)[i_82653] = defunc_0_lifted_lambda_res_77095;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83882, i_82657 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83887, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82665 = 0; i_82665 < (int64_t) 16; i_82665++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82661 = 0; i_82661 < (int64_t) 16; i_82661++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77116 = ((double *) mem_83882)[i_82665 * (int64_t) 16 + i_82661];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77117 = ((double *) mem_83530)[i_82665 * (int64_t) 16 + i_82661];
                
                // futhark/microgpt.fut:231:38-68
                
                double zp_res_77118 = zp_lhs_77116 + zp_rhs_77117;
                
                ((double *) mem_83903)[i_82661] = zp_res_77118;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83898, i_82665 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83903, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82678 = 0; i_82678 < (int64_t) 16; i_82678++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79457;
            double r_79459 = 0.0;
            
            for (int64_t i_79458 = 0; i_79458 < (int64_t) 16; i_79458++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_79460 = ((double *) mem_83898)[i_82678 * (int64_t) 16 + i_79458];
                
                // futhark/microgpt.fut:232:62-93
                
                double zt_res_79461 = zt_lhs_79460 * zt_lhs_79460;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79462 = r_79459 + zt_res_79461;
                double r_tmp_85277 = zp_res_79462;
                
                r_79459 = r_tmp_85277;
            }
            defunc_0_lifted_lambda_res_79457 = r_79459;
            // futhark/microgpt.fut:232:43-111
            
            double zs_res_79463 = defunc_0_lifted_lambda_res_79457 / 16.0;
            
            // futhark/microgpt.fut:233:24-55
            
            double zp_res_79464 = 1.0e-5 + zs_res_79463;
            
            // futhark/microgpt.fut:233:16-55
            
            double sqrt_res_79465 = futrts_sqrt64(zp_res_79464);
            
            // futhark/microgpt.fut:234:43-54
            
            double zs_res_79466 = 1.0 / sqrt_res_79465;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82671 = 0; i_82671 < (int64_t) 16; i_82671++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81073 = ((double *) mem_83898)[i_82678 * (int64_t) 16 + i_82671];
                
                // futhark/microgpt.fut:234:24-54
                
                double zt_res_81074 = zs_res_79466 * zt_lhs_81073;
                
                // futhark/microgpt.fut:254:53-88
                
                double zt_res_81082 = zt_lhs_81073 * zt_lhs_81073;
                
                ((double *) mem_83924)[i_82671] = zt_res_81082;
                ((double *) mem_83925)[i_82671] = zt_res_81074;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83914, i_82678 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83924, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83915, i_82678 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83925, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82689 = 0; i_82689 < (int64_t) 16; i_82689++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82683 = 0; i_82683 < (int64_t) 64; i_82683++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79514;
                double r_79516 = 0.0;
                
                for (int64_t i_79515 = 0; i_79515 < (int64_t) 16; i_79515++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_79517 = ((double *) mem_param_83384.mem)[i_82683 * (int64_t) 16 + i_79515];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_79518 = ((double *) mem_83915)[i_82689 * (int64_t) 16 + i_79515];
                    
                    // futhark/microgpt.fut:235:69-100
                    
                    double zt_res_79519 = zt_lhs_79517 * zt_rhs_79518;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79520 = r_79516 + zt_res_79519;
                    double r_tmp_85283 = zp_res_79520;
                    
                    r_79516 = r_tmp_85283;
                }
                defunc_0_lifted_lambda_res_79514 = r_79516;
                ((double *) mem_83955)[i_82683] = defunc_0_lifted_lambda_res_79514;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79528;
            double r_79530 = 0.0;
            
            for (int64_t i_79529 = 0; i_79529 < (int64_t) 16; i_79529++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79531 = ((double *) mem_83914)[i_82689 * (int64_t) 16 + i_79529];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79532 = r_79530 + lifted_lambda_res_79531;
                double r_tmp_85284 = zp_res_79532;
                
                r_79530 = r_tmp_85284;
            }
            defunc_0_lifted_lambda_res_79528 = r_79530;
            // futhark/microgpt.fut:255:34-86
            
            double zs_res_79533 = defunc_0_lifted_lambda_res_79528 / 16.0;
            
            ((double *) mem_83946)[i_82689] = zs_res_79533;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83947, i_82689 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83955, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82700 = 0; i_82700 < (int64_t) 16; i_82700++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82694 = 0; i_82694 < (int64_t) 64; i_82694++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_79557 = ((double *) mem_83947)[i_82700 * (int64_t) 64 + i_82694];
                
                // futhark/microgpt.fut:236:38-62
                
                double max_res_79558 = fmax64(0.0, max_arg0_79557);
                
                ((double *) mem_83978)[i_82694] = max_res_79558;
            }
            // futhark/microgpt.fut:256:41-51
            
            double zp_lhs_79566 = ((double *) mem_83946)[i_82700];
            
            // futhark/microgpt.fut:256:41-79
            
            double zp_res_79567 = 1.0e-5 + zp_lhs_79566;
            
            // futhark/microgpt.fut:256:33-79
            
            double sqrt_res_79568 = futrts_sqrt64(zp_res_79567);
            
            ((double *) mem_83969)[i_82700] = sqrt_res_79568;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83970, i_82700 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83978, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82709 = 0; i_82709 < (int64_t) 16; i_82709++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82705 = 0; i_82705 < (int64_t) 16; i_82705++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77196;
                double r_77198 = 0.0;
                
                for (int64_t i_77197 = 0; i_77197 < (int64_t) 64; i_77197++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77199 = ((double *) mem_param_83360.mem)[i_82705 * (int64_t) 64 + i_77197];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77200 = ((double *) mem_83970)[i_82709 * (int64_t) 64 + i_77197];
                    
                    // futhark/microgpt.fut:237:69-102
                    
                    double zt_res_77201 = zt_lhs_77199 * zt_rhs_77200;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77202 = r_77198 + zt_res_77201;
                    double r_tmp_85290 = zp_res_77202;
                    
                    r_77198 = r_tmp_85290;
                }
                defunc_0_lifted_lambda_res_77196 = r_77198;
                ((double *) mem_83997)[i_82705] = defunc_0_lifted_lambda_res_77196;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83992, i_82709 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83997, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82717 = 0; i_82717 < (int64_t) 16; i_82717++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82713 = 0; i_82713 < (int64_t) 16; i_82713++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77217 = ((double *) mem_83992)[i_82717 * (int64_t) 16 + i_82713];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77218 = ((double *) mem_83898)[i_82717 * (int64_t) 16 + i_82713];
                
                // futhark/microgpt.fut:238:38-69
                
                double zp_res_77219 = zp_lhs_77217 + zp_rhs_77218;
                
                ((double *) mem_84013)[i_82713] = zp_res_77219;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84008, i_82717 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84013, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82725 = 0; i_82725 < (int64_t) 16; i_82725++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82721 = 0; i_82721 < (int64_t) 27; i_82721++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77234;
                double r_77236 = 0.0;
                
                for (int64_t i_77235 = 0; i_77235 < (int64_t) 16; i_77235++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77237 = ((double *) mem_param_83392.mem)[i_82721 * (int64_t) 16 + i_77235];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77238 = ((double *) mem_84008)[i_82725 * (int64_t) 16 + i_77235];
                    
                    // futhark/microgpt.fut:239:69-101
                    
                    double zt_res_77239 = zt_lhs_77237 * zt_rhs_77238;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77240 = r_77236 + zt_res_77239;
                    double r_tmp_85295 = zp_res_77240;
                    
                    r_77236 = r_tmp_85295;
                }
                defunc_0_lifted_lambda_res_77234 = r_77236;
                ((double *) mem_84029)[i_82721] = defunc_0_lifted_lambda_res_77234;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84024, i_82725 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84029, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82755 = 0; i_82755 < (int64_t) 16; i_82755++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_82330;
            double defunc_0_reduce_res_82331;
            double redout_82727;
            double redout_82728;
            
            redout_82727 = -INFINITY;
            redout_82728 = -INFINITY;
            for (int64_t i_82729 = 0; i_82729 < (int64_t) 27; i_82729++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81150 = ((double *) mem_84024)[i_82755 * (int64_t) 27 + i_82729];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79598 = fmax64(lifted_lambda_res_81150, redout_82727);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79650 = fmax64(lifted_lambda_res_81150, redout_82728);
                double redout_tmp_85298 = max_res_79598;
                double redout_tmp_85299 = max_res_79650;
                
                redout_82727 = redout_tmp_85298;
                redout_82728 = redout_tmp_85299;
            }
            defunc_0_reduce_res_82330 = redout_82727;
            defunc_0_reduce_res_82331 = redout_82728;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79599 = -defunc_0_reduce_res_82330;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79651 = -defunc_0_reduce_res_82331;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82734 = 0; i_82734 < (int64_t) 27; i_82734++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_81189 = ((double *) mem_84024)[i_82755 * (int64_t) 27 + i_82734];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81190 = neg_res_79599 + lifted_lambda_res_81189;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81191 = futrts_exp64(zp_res_81190);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81199 = neg_res_79651 + lifted_lambda_res_81189;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81200 = futrts_exp64(zp_res_81199);
                
                ((double *) mem_84050)[i_82734] = exp_res_81200;
                ((double *) mem_84051)[i_82734] = exp_res_81191;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79610;
            double r_79612 = 0.0;
            
            for (int64_t i_79611 = 0; i_79611 < (int64_t) 27; i_79611++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79613 = ((double *) mem_84051)[i_79611];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79614 = r_79612 + lifted_lambda_res_79613;
                double r_tmp_85302 = zp_res_79614;
                
                r_79612 = r_tmp_85302;
            }
            defunc_0_lifted_lambda_res_79610 = r_79612;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79662;
            double r_79664 = 0.0;
            
            for (int64_t i_79663 = 0; i_79663 < (int64_t) 27; i_79663++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79665 = ((double *) mem_84050)[i_79663];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79666 = r_79664 + lifted_lambda_res_79665;
                double r_tmp_85303 = zp_res_79666;
                
                r_79664 = r_tmp_85303;
            }
            defunc_0_lifted_lambda_res_79662 = r_79664;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82741 = 0; i_82741 < (int64_t) 27; i_82741++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81218 = ((double *) mem_84051)[i_82741];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81219 = zs_lhs_81218 / defunc_0_lifted_lambda_res_79610;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81226 = ((double *) mem_84050)[i_82741];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81227 = zs_lhs_81226 / defunc_0_lifted_lambda_res_79662;
                
                ((double *) mem_84064)[i_82741] = zs_res_81227;
                ((double *) mem_84065)[i_82741] = zs_res_81219;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82748 = 0; i_82748 < (int64_t) 27; i_82748++) {
                // futhark/microgpt.fut:245:24-34
                
                double lifted_lambda_res_81245 = ((double *) mem_84065)[i_82748];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81252 = ((double *) mem_83497)[i_82755 * (int64_t) 27 + i_82748];
                
                // futhark/microgpt.fut:247:4-14
                
                double zs_rhs_81253 = ((double *) mem_84064)[i_82748];
                
                // futhark/microgpt.fut:246:74-247:14
                
                double zs_res_81254 = 1.0 / zs_rhs_81253;
                
                // futhark/microgpt.fut:246:53-247:14
                
                double zt_res_81255 = zt_lhs_81252 * zs_res_81254;
                
                ((double *) mem_84078)[i_82748] = zt_res_81255;
                ((double *) mem_84079)[i_82748] = lifted_lambda_res_81245;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84040, i_82755 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84078, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84041, i_82755 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84079, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82760 = 0; i_82760 < (int64_t) 16; i_82760++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77374;
            double r_77376 = 0.0;
            
            for (int64_t i_77375 = 0; i_77375 < (int64_t) 27; i_77375++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77377 = ((double *) mem_84040)[i_82760 * (int64_t) 27 + i_77375];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77378 = ((double *) mem_84041)[i_82760 * (int64_t) 27 + i_77375];
                
                // futhark/microgpt.fut:248:53-90
                
                double zt_res_77379 = zt_lhs_77377 * zt_rhs_77378;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77380 = r_77376 + zt_res_77379;
                double r_tmp_85309 = zp_res_77380;
                
                r_77376 = r_tmp_85309;
            }
            defunc_0_lifted_lambda_res_77374 = r_77376;
            ((double *) mem_84100)[i_82760] = defunc_0_lifted_lambda_res_77374;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82768 = 0; i_82768 < (int64_t) 16; i_82768++) {
            // futhark/microgpt.fut:249:103-113
            
            double neg_arg0_77388 = ((double *) mem_84100)[i_82768];
            
            // futhark/microgpt.fut:249:97-113
            
            double neg_res_77389 = -neg_arg0_77388;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82764 = 0; i_82764 < (int64_t) 27; i_82764++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77396 = ((double *) mem_84041)[i_82768 * (int64_t) 27 + i_82764];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77397 = ((double *) mem_84040)[i_82768 * (int64_t) 27 + i_82764];
                
                // futhark/microgpt.fut:249:75-113
                
                double zp_res_77398 = neg_res_77389 + zp_lhs_77397;
                
                // futhark/microgpt.fut:249:53-113
                
                double zt_res_77399 = zt_lhs_77396 * zp_res_77398;
                
                ((double *) mem_84112)[i_82764] = zt_res_77399;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84107, i_82768 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84112, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82776 = 0; i_82776 < (int64_t) 16; i_82776++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82772 = 0; i_82772 < (int64_t) 16; i_82772++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77414;
                double r_77416 = 0.0;
                
                for (int64_t i_77415 = 0; i_77415 < (int64_t) 27; i_77415++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77417 = ((double *) mem_param_83392.mem)[i_77415 * (int64_t) 16 + i_82772];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77418 = ((double *) mem_84107)[i_82776 * (int64_t) 27 + i_77415];
                    
                    // futhark/microgpt.fut:250:73-110
                    
                    double zt_res_77419 = zt_lhs_77417 * zt_rhs_77418;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77420 = r_77416 + zt_res_77419;
                    double r_tmp_85314 = zp_res_77420;
                    
                    r_77416 = r_tmp_85314;
                }
                defunc_0_lifted_lambda_res_77414 = r_77416;
                ((double *) mem_84128)[i_82772] = defunc_0_lifted_lambda_res_77414;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84123, i_82776 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84128, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82789 = 0; i_82789 < (int64_t) 16; i_82789++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82782 = 0; i_82782 < (int64_t) 64; i_82782++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81283;
                double r_81285 = 0.0;
                
                for (int64_t i_81284 = 0; i_81284 < (int64_t) 16; i_81284++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81286 = ((double *) mem_param_83360.mem)[i_81284 * (int64_t) 64 + i_82782];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81287 = ((double *) mem_84123)[i_82789 * (int64_t) 16 + i_81284];
                    
                    // futhark/microgpt.fut:251:73-111
                    
                    double zt_res_81288 = zt_lhs_81286 * zt_rhs_81287;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81289 = r_81285 + zt_res_81288;
                    double r_tmp_85319 = zp_res_81289;
                    
                    r_81285 = r_tmp_85319;
                }
                defunc_0_lifted_lambda_res_81283 = r_81285;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81296;
                double r_81298 = 0.0;
                
                for (int64_t i_81297 = 0; i_81297 < (int64_t) 16; i_81297++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81299 = ((double *) mem_84123)[i_81297 * (int64_t) 16 + i_82789];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81300 = ((double *) mem_83970)[i_81297 * (int64_t) 64 + i_82782];
                    
                    // futhark/microgpt.fut:301:75-111
                    
                    double zt_res_81301 = zt_lhs_81299 * zt_rhs_81300;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81302 = r_81298 + zt_res_81301;
                    double r_tmp_85320 = zp_res_81302;
                    
                    r_81298 = r_tmp_85320;
                }
                defunc_0_lifted_lambda_res_81296 = r_81298;
                ((double *) mem_84149)[i_82782] = defunc_0_lifted_lambda_res_81296;
                ((double *) mem_84150)[i_82782] = defunc_0_lifted_lambda_res_81283;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84139, i_82789 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84149, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84140, i_82789 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84150, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82798 = 0; i_82798 < (int64_t) 16; i_82798++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82794 = 0; i_82794 < (int64_t) 64; i_82794++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_77456 = ((double *) mem_83947)[i_82798 * (int64_t) 64 + i_82794];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_77457 = fmax64(0.0, indicatorp_arg0_77456);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_77458 = fsignum64(max_res_77457);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77459 = ((double *) mem_84140)[i_82798 * (int64_t) 64 + i_82794];
                
                // futhark/microgpt.fut:252:42-90
                
                double zt_res_77460 = sgn_res_77458 * zt_rhs_77459;
                
                ((double *) mem_84176)[i_82794] = zt_res_77460;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84171, i_82798 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84176, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82806 = 0; i_82806 < (int64_t) 16; i_82806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82802 = 0; i_82802 < (int64_t) 16; i_82802++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77475;
                double r_77477 = 0.0;
                
                for (int64_t i_77476 = 0; i_77476 < (int64_t) 64; i_77476++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77478 = ((double *) mem_param_83384.mem)[i_77476 * (int64_t) 16 + i_82802];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77479 = ((double *) mem_84171)[i_82806 * (int64_t) 64 + i_77476];
                    
                    // futhark/microgpt.fut:253:73-109
                    
                    double zt_res_77480 = zt_lhs_77478 * zt_rhs_77479;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77481 = r_77477 + zt_res_77480;
                    double r_tmp_85325 = zp_res_77481;
                    
                    r_77477 = r_tmp_85325;
                }
                defunc_0_lifted_lambda_res_77475 = r_77477;
                ((double *) mem_84192)[i_82802] = defunc_0_lifted_lambda_res_77475;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84187, i_82806 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82810 = 0; i_82810 < (int64_t) 16; i_82810++) {
            // futhark/microgpt.fut:257:49-59
            
            double zs_rhs_77529 = ((double *) mem_83969)[i_82810];
            
            // futhark/microgpt.fut:257:41-59
            
            double zs_res_77530 = 1.0 / zs_rhs_77529;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77531;
            double r_77533 = 0.0;
            
            for (int64_t i_77532 = 0; i_77532 < (int64_t) 16; i_77532++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77534 = ((double *) mem_83898)[i_82810 * (int64_t) 16 + i_77532];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77535 = ((double *) mem_84187)[i_82810 * (int64_t) 16 + i_77532];
                
                // futhark/microgpt.fut:257:87-123
                
                double zt_res_77536 = zt_lhs_77534 * zt_rhs_77535;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77537 = r_77533 + zt_res_77536;
                double r_tmp_85327 = zp_res_77537;
                
                r_77533 = r_tmp_85327;
            }
            defunc_0_lifted_lambda_res_77531 = r_77533;
            // futhark/microgpt.fut:257:67-150
            
            double zt_res_77538 = zs_res_77530 * defunc_0_lifted_lambda_res_77531;
            
            // futhark/microgpt.fut:257:45-150
            
            double zt_res_77539 = zs_res_77530 * zt_res_77538;
            
            // futhark/microgpt.fut:257:33-150
            
            double neg_res_77540 = -zt_res_77539;
            
            ((double *) mem_84203)[i_82810] = neg_res_77540;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82814 = 0; i_82814 < (int64_t) 16; i_82814++) {
            // futhark/microgpt.fut:258:33-43
            
            double zt_lhs_77548 = ((double *) mem_84203)[i_82814];
            
            // futhark/microgpt.fut:258:85-95
            
            double zp_lhs_77549 = ((double *) mem_83946)[i_82814];
            
            // futhark/microgpt.fut:258:85-123
            
            double zp_res_77550 = 1.0e-5 + zp_lhs_77549;
            
            // futhark/microgpt.fut:258:77-123
            
            double sqrt_res_77551 = futrts_sqrt64(zp_res_77550);
            
            // futhark/microgpt.fut:258:63-125
            
            double zt_res_77552 = 2.0 * sqrt_res_77551;
            
            // futhark/microgpt.fut:258:49-125
            
            double zs_res_77553 = 1.0 / zt_res_77552;
            
            // futhark/microgpt.fut:258:33-125
            
            double zt_res_77554 = zt_lhs_77548 * zs_res_77553;
            
            ((double *) mem_84210)[i_82814] = zt_res_77554;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82818 = 0; i_82818 < (int64_t) 16; i_82818++) {
            // futhark/microgpt.fut:259:53-63
            
            double zs_lhs_77562 = ((double *) mem_84210)[i_82818];
            
            // futhark/microgpt.fut:259:53-78
            
            double zs_res_77563 = zs_lhs_77562 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85330 = 0; nest_i_85330 < (int64_t) 16; nest_i_85330++) {
                ((double *) mem_84217)[i_82818 * (int64_t) 16 + nest_i_85330] = zs_res_77563;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82826 = 0; i_82826 < (int64_t) 16; i_82826++) {
            // futhark/microgpt.fut:260:107-117
            
            double zs_rhs_77572 = ((double *) mem_83969)[i_82826];
            
            // futhark/microgpt.fut:260:99-117
            
            double zs_res_77573 = 1.0 / zs_rhs_77572;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82822 = 0; i_82822 < (int64_t) 16; i_82822++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77580 = ((double *) mem_84123)[i_82826 * (int64_t) 16 + i_82822];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77581 = ((double *) mem_84187)[i_82826 * (int64_t) 16 + i_82822];
                
                // futhark/microgpt.fut:260:77-117
                
                double zt_res_77582 = zs_res_77573 * zt_lhs_77581;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77583 = ((double *) mem_83898)[i_82826 * (int64_t) 16 + i_82822];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77584 = ((double *) mem_84217)[i_82826 * (int64_t) 16 + i_82822];
                
                // futhark/microgpt.fut:260:125-161
                
                double zt_res_77585 = zt_lhs_77583 * zt_rhs_77584;
                
                // futhark/microgpt.fut:260:94-161
                
                double zp_res_77586 = zt_res_77582 + zt_res_77585;
                
                // futhark/microgpt.fut:260:120-205
                
                double zp_res_77587 = zt_res_77585 + zp_res_77586;
                
                // futhark/microgpt.fut:260:53-205
                
                double zp_res_77588 = zp_lhs_77580 + zp_res_77587;
                
                ((double *) mem_84232)[i_82822] = zp_res_77588;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84227, i_82826 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84232, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82839 = 0; i_82839 < (int64_t) 16; i_82839++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82832 = 0; i_82832 < (int64_t) 16; i_82832++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81325;
                double r_81327 = 0.0;
                
                for (int64_t i_81326 = 0; i_81326 < (int64_t) 16; i_81326++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81328 = ((double *) mem_param_83368.mem)[i_81326 * (int64_t) 16 + i_82832];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81329 = ((double *) mem_84227)[i_82839 * (int64_t) 16 + i_81326];
                    
                    // futhark/microgpt.fut:261:73-110
                    
                    double zt_res_81330 = zt_lhs_81328 * zt_rhs_81329;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81331 = r_81327 + zt_res_81330;
                    double r_tmp_85337 = zp_res_81331;
                    
                    r_81327 = r_tmp_85337;
                }
                defunc_0_lifted_lambda_res_81325 = r_81327;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81338;
                double r_81340 = 0.0;
                
                for (int64_t i_81339 = 0; i_81339 < (int64_t) 16; i_81339++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81341 = ((double *) mem_84227)[i_81339 * (int64_t) 16 + i_82839];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81342 = ((double *) mem_83860)[i_81339 * (int64_t) 16 + i_82832];
                    
                    // futhark/microgpt.fut:299:74-110
                    
                    double zt_res_81343 = zt_lhs_81341 * zt_rhs_81342;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81344 = r_81340 + zt_res_81343;
                    double r_tmp_85338 = zp_res_81344;
                    
                    r_81340 = r_tmp_85338;
                }
                defunc_0_lifted_lambda_res_81338 = r_81340;
                ((double *) mem_84253)[i_82832] = defunc_0_lifted_lambda_res_81338;
                ((double *) mem_84254)[i_82832] = defunc_0_lifted_lambda_res_81325;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84243, i_82839 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84253, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84244, i_82839 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84254, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82861 = 0; i_82861 < (int64_t) 4; i_82861++) {
            // futhark/microgpt.fut:262:88-91
            
            int64_t zp_lhs_79802 = mul64((int64_t) 4, i_82861);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82854 = 0; i_82854 < (int64_t) 16; i_82854++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82844 = 0; i_82844 < (int64_t) 4; i_82844++) {
                    // futhark/microgpt.fut:262:93-99
                    
                    int64_t tmp_81366 = add64(zp_lhs_79802, i_82844);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool x_81367 = sle64((int64_t) 0, tmp_81366);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool y_81368 = slt64(tmp_81366, (int64_t) 16);
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool bounds_check_81369 = x_81367 && y_81368;
                    
                    // futhark/microgpt.fut:262:70-101
                    
                    bool index_certs_81370;
                    
                    if (!bounds_check_81369) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81366, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:262:70-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:262:52-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:262:32-104\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:262:13-106\n   #10 futhark/microgpt.fut:415:5-76\n   #11 futhark/microgpt.fut:419:44-425:31\n   #12 futhark/microgpt.fut:441:11-60\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81371 = ((double *) mem_84244)[i_82854 * (int64_t) 16 + tmp_81366];
                    
                    ((double *) mem_84297)[i_82844] = lifted_lambda_res_81371;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82848 = 0; i_82848 < (int64_t) 16; i_82848++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_81385 = ((double *) mem_83743)[i_82861 * (int64_t) 256 + i_82854 * (int64_t) 16 + i_82848];
                    
                    // futhark/microgpt.fut:264:61-97
                    
                    double zs_res_81386 = zs_lhs_81385 / 2.0;
                    double zp_rhs_81387 = ((double *) masks_mem_83354.mem)[step_76681 * (int64_t) 256 + i_82854 * (int64_t) 16 + i_82848];
                    
                    // futhark/microgpt.fut:264:84-119
                    
                    double zp_res_81388 = zs_res_81386 + zp_rhs_81387;
                    
                    ((double *) mem_84304)[i_82848] = zp_res_81388;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84287, i_82854 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84304, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84288, i_82854 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84297, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84275, i_82861 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84287, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84276, i_82861 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84288, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82892 = 0; i_82892 < (int64_t) 4; i_82892++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82885 = 0; i_82885 < (int64_t) 16; i_82885++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_82351;
                double redout_82865 = -INFINITY;
                
                for (int64_t i_82867 = 0; i_82867 < (int64_t) 16; i_82867++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81514 = ((double *) mem_84275)[i_82892 * (int64_t) 256 + i_82885 * (int64_t) 16 + i_82867];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81525;
                    double r_81527 = 0.0;
                    
                    for (int64_t i_81526 = 0; i_81526 < (int64_t) 4; i_81526++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81528 = ((double *) mem_84276)[i_82892 * (int64_t) 64 + i_82885 * (int64_t) 4 + i_81526];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81529 = ((double *) mem_83662)[i_82892 * (int64_t) 64 + i_82867 * (int64_t) 4 + i_81526];
                        
                        // futhark/microgpt.fut:267:91-139
                        
                        double zt_res_81530 = zt_lhs_81528 * zt_rhs_81529;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81531 = r_81527 + zt_res_81530;
                        double r_tmp_85351 = zp_res_81531;
                        
                        r_81527 = r_tmp_85351;
                    }
                    defunc_0_lifted_lambda_res_81525 = r_81527;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_81425 = fmax64(lifted_lambda_res_81514, redout_82865);
                    
                    ((double *) mem_84351)[i_82867] = defunc_0_lifted_lambda_res_81525;
                    
                    double redout_tmp_85349 = max_res_81425;
                    
                    redout_82865 = redout_tmp_85349;
                }
                defunc_0_reduce_res_82351 = redout_82865;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_81426 = -defunc_0_reduce_res_82351;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82871 = 0; i_82871 < (int64_t) 16; i_82871++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_81433 = ((double *) mem_84275)[i_82892 * (int64_t) 256 + i_82885 * (int64_t) 16 + i_82871];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_81434 = neg_res_81426 + lifted_lambda_res_81433;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_81435 = futrts_exp64(zp_res_81434);
                    
                    ((double *) mem_84358)[i_82871] = exp_res_81435;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81437;
                double r_81439 = 0.0;
                
                for (int64_t i_81438 = 0; i_81438 < (int64_t) 16; i_81438++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_81440 = ((double *) mem_84358)[i_81438];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81441 = r_81439 + lifted_lambda_res_81440;
                    double r_tmp_85353 = zp_res_81441;
                    
                    r_81439 = r_tmp_85353;
                }
                defunc_0_lifted_lambda_res_81437 = r_81439;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82875 = 0; i_82875 < (int64_t) 16; i_82875++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_81448 = ((double *) mem_84358)[i_82875];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_81449 = zs_lhs_81448 / defunc_0_lifted_lambda_res_81437;
                    
                    ((double *) mem_84365)[i_82875] = zs_res_81449;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82879 = 0; i_82879 < (int64_t) 16; i_82879++) {
                    // futhark/microgpt.fut:266:24-34
                    
                    double lifted_lambda_res_81457 = ((double *) mem_84365)[i_82879];
                    
                    ((double *) mem_84372)[i_82879] = lifted_lambda_res_81457;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84341, i_82885 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84351, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84342, i_82885 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84372, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84329, i_82892 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84341, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84330, i_82892 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84342, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82914 = 0; i_82914 < (int64_t) 4; i_82914++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82907 = 0; i_82907 < (int64_t) 16; i_82907++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82897 = 0; i_82897 < (int64_t) 16; i_82897++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81567 = ((double *) mem_84329)[i_82914 * (int64_t) 256 + i_82907 * (int64_t) 16 + i_82897];
                    
                    ((double *) mem_84419)[i_82897] = lifted_lambda_res_81567;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82901 = 0; i_82901 < (int64_t) 4; i_82901++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81581;
                    double r_81583 = 0.0;
                    
                    for (int64_t i_81582 = 0; i_81582 < (int64_t) 16; i_81582++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81584 = ((double *) mem_84330)[i_82914 * (int64_t) 256 + i_81582 * (int64_t) 16 + i_82907];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81585 = ((double *) mem_84276)[i_82914 * (int64_t) 64 + i_81582 * (int64_t) 4 + i_82901];
                        
                        // futhark/microgpt.fut:272:91-140
                        
                        double zt_res_81586 = zt_lhs_81584 * zt_rhs_81585;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81587 = r_81583 + zt_res_81586;
                        double r_tmp_85362 = zp_res_81587;
                        
                        r_81583 = r_tmp_85362;
                    }
                    defunc_0_lifted_lambda_res_81581 = r_81583;
                    ((double *) mem_84426)[i_82901] = defunc_0_lifted_lambda_res_81581;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84409, i_82907 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84426, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84410, i_82907 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84419, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84397, i_82914 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84409, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84398, i_82914 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84410, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82923 = 0; i_82923 < (int64_t) 4; i_82923++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82919 = 0; i_82919 < (int64_t) 16; i_82919++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77807;
                double r_77809 = 0.0;
                
                for (int64_t i_77808 = 0; i_77808 < (int64_t) 16; i_77808++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77810 = ((double *) mem_84398)[i_82923 * (int64_t) 256 + i_82919 * (int64_t) 16 + i_77808];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77811 = ((double *) mem_84330)[i_82923 * (int64_t) 256 + i_82919 * (int64_t) 16 + i_77808];
                    
                    // futhark/microgpt.fut:269:72-121
                    
                    double zt_res_77812 = zt_lhs_77810 * zt_rhs_77811;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77813 = r_77809 + zt_res_77812;
                    double r_tmp_85365 = zp_res_77813;
                    
                    r_77809 = r_tmp_85365;
                }
                defunc_0_lifted_lambda_res_77807 = r_77809;
                ((double *) mem_84456)[i_82919] = defunc_0_lifted_lambda_res_77807;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84451, i_82923 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84456, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82935 = 0; i_82935 < (int64_t) 4; i_82935++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82931 = 0; i_82931 < (int64_t) 16; i_82931++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_77828 = ((double *) mem_84451)[i_82935 * (int64_t) 16 + i_82931];
                
                // futhark/microgpt.fut:270:128-150
                
                double neg_res_77829 = -neg_arg0_77828;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82927 = 0; i_82927 < (int64_t) 16; i_82927++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_77836 = ((double *) mem_84330)[i_82935 * (int64_t) 256 + i_82931 * (int64_t) 16 + i_82927];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_77837 = ((double *) mem_84398)[i_82935 * (int64_t) 256 + i_82931 * (int64_t) 16 + i_82927];
                    
                    // futhark/microgpt.fut:270:100-150
                    
                    double zp_res_77838 = neg_res_77829 + zp_lhs_77837;
                    
                    // futhark/microgpt.fut:270:72-150
                    
                    double zt_res_77839 = zt_lhs_77836 * zp_res_77838;
                    
                    ((double *) mem_84478)[i_82927] = zt_res_77839;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84473, i_82931 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84478, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84467, i_82935 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84473, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82947 = 0; i_82947 < (int64_t) 4; i_82947++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82943 = 0; i_82943 < (int64_t) 16; i_82943++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82939 = 0; i_82939 < (int64_t) 16; i_82939++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_77861 = ((double *) mem_84467)[i_82947 * (int64_t) 256 + i_82943 * (int64_t) 16 + i_82939];
                    
                    // futhark/microgpt.fut:271:60-96
                    
                    double zs_res_77862 = zs_lhs_77861 / 2.0;
                    
                    ((double *) mem_84505)[i_82939] = zs_res_77862;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84500, i_82943 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84505, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84494, i_82947 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84500, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82967 = 0; i_82967 < (int64_t) 4; i_82967++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82960 = 0; i_82960 < (int64_t) 16; i_82960++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82953 = 0; i_82953 < (int64_t) 4; i_82953++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81668;
                    double r_81670 = 0.0;
                    
                    for (int64_t i_81669 = 0; i_81669 < (int64_t) 16; i_81669++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81671 = ((double *) mem_83664)[i_82967 * (int64_t) 64 + i_81669 * (int64_t) 4 + i_82953];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81672 = ((double *) mem_84494)[i_82967 * (int64_t) 256 + i_81669 * (int64_t) 16 + i_82960];
                        
                        // futhark/microgpt.fut:273:91-139
                        
                        double zt_res_81673 = zt_lhs_81671 * zt_rhs_81672;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81674 = r_81670 + zt_res_81673;
                        double r_tmp_85378 = zp_res_81674;
                        
                        r_81670 = r_tmp_85378;
                    }
                    defunc_0_lifted_lambda_res_81668 = r_81670;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81681;
                    double r_81683 = 0.0;
                    
                    for (int64_t i_81682 = 0; i_81682 < (int64_t) 16; i_81682++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81684 = ((double *) mem_84494)[i_82967 * (int64_t) 256 + i_82960 * (int64_t) 16 + i_81682];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81685 = ((double *) mem_83663)[i_82967 * (int64_t) 64 + i_81682 * (int64_t) 4 + i_82953];
                        
                        // futhark/microgpt.fut:274:91-139
                        
                        double zt_res_81686 = zt_lhs_81684 * zt_rhs_81685;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81687 = r_81683 + zt_res_81686;
                        double r_tmp_85379 = zp_res_81687;
                        
                        r_81683 = r_tmp_85379;
                    }
                    defunc_0_lifted_lambda_res_81681 = r_81683;
                    ((double *) mem_84543)[i_82953] = defunc_0_lifted_lambda_res_81681;
                    ((double *) mem_84544)[i_82953] = defunc_0_lifted_lambda_res_81668;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84533, i_82960 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84543, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84534, i_82960 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84544, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84521, i_82967 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84533, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84522, i_82967 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84534, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82986 = 0; i_82986 < (int64_t) 16; i_82986++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82976 = 0; i_82976 < (int64_t) 16; i_82976++) {
                // futhark/microgpt.fut:275:63-66
                
                int64_t tmp_81750 = sdiv64(i_82976, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-68
                
                bool x_81751 = sle64((int64_t) 0, tmp_81750);
                
                // futhark/microgpt.fut:275:52-68
                
                bool y_81752 = slt64(tmp_81750, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-68
                
                bool bounds_check_81753 = x_81751 && y_81752;
                
                // futhark/microgpt.fut:275:52-68
                
                bool index_certs_81754;
                
                if (!bounds_check_81753) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81750, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:275:52-68\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:275:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:275:13-89\n   #7  futhark/microgpt.fut:415:5-76\n   #8  futhark/microgpt.fut:419:44-425:31\n   #9  futhark/microgpt.fut:441:11-60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:275:81-84
                
                int64_t tmp_81755 = smod64(i_82976, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-86
                
                bool x_81756 = sle64((int64_t) 0, tmp_81755);
                
                // futhark/microgpt.fut:275:52-86
                
                bool y_81757 = slt64(tmp_81755, (int64_t) 4);
                
                // futhark/microgpt.fut:275:52-86
                
                bool bounds_check_81758 = x_81756 && y_81757;
                
                // futhark/microgpt.fut:275:52-86
                
                bool index_certs_81759;
                
                if (!bounds_check_81758) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81755, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:275:52-86\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:275:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:275:13-89\n   #7  futhark/microgpt.fut:415:5-76\n   #8  futhark/microgpt.fut:419:44-425:31\n   #9  futhark/microgpt.fut:441:11-60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81760 = ((double *) mem_84397)[tmp_81750 * (int64_t) 64 + i_82986 * (int64_t) 4 + tmp_81755];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81773 = ((double *) mem_84522)[tmp_81750 * (int64_t) 64 + i_82986 * (int64_t) 4 + tmp_81755];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81789 = ((double *) mem_84521)[tmp_81750 * (int64_t) 64 + i_82986 * (int64_t) 4 + tmp_81755];
                
                ((double *) mem_84590)[i_82976] = lifted_lambda_res_81789;
                ((double *) mem_84591)[i_82976] = lifted_lambda_res_81773;
                ((double *) mem_84592)[i_82976] = lifted_lambda_res_81760;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84575, i_82986 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84590, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84576, i_82986 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84591, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84577, i_82986 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84592, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83011 = 0; i_83011 < (int64_t) 16; i_83011++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82998 = 0; i_82998 < (int64_t) 16; i_82998++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81952;
                double r_81954 = 0.0;
                
                for (int64_t i_81953 = 0; i_81953 < (int64_t) 16; i_81953++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81955 = ((double *) mem_param_83388.mem)[i_81953 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81956 = ((double *) mem_84577)[i_83011 * (int64_t) 16 + i_81953];
                    
                    // futhark/microgpt.fut:278:75-112
                    
                    double zt_res_81957 = zt_lhs_81955 * zt_rhs_81956;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81958 = r_81954 + zt_res_81957;
                    double r_tmp_85394 = zp_res_81958;
                    
                    r_81954 = r_tmp_85394;
                }
                defunc_0_lifted_lambda_res_81952 = r_81954;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81959;
                double r_81961 = 0.0;
                
                for (int64_t i_81960 = 0; i_81960 < (int64_t) 16; i_81960++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81962 = ((double *) mem_param_83364.mem)[i_81960 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81963 = ((double *) mem_84576)[i_83011 * (int64_t) 16 + i_81960];
                    
                    // futhark/microgpt.fut:278:141-178
                    
                    double zt_res_81964 = zt_lhs_81962 * zt_rhs_81963;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81965 = r_81961 + zt_res_81964;
                    double r_tmp_85395 = zp_res_81965;
                    
                    r_81961 = r_tmp_85395;
                }
                defunc_0_lifted_lambda_res_81959 = r_81961;
                // futhark/microgpt.fut:278:55-180
                
                double zp_res_81966 = defunc_0_lifted_lambda_res_81952 + defunc_0_lifted_lambda_res_81959;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81967;
                double r_81969 = 0.0;
                
                for (int64_t i_81968 = 0; i_81968 < (int64_t) 16; i_81968++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81970 = ((double *) mem_param_83376.mem)[i_81968 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81971 = ((double *) mem_84575)[i_83011 * (int64_t) 16 + i_81968];
                    
                    // futhark/microgpt.fut:278:208-245
                    
                    double zt_res_81972 = zt_lhs_81970 * zt_rhs_81971;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81973 = r_81969 + zt_res_81972;
                    double r_tmp_85396 = zp_res_81973;
                    
                    r_81969 = r_tmp_85396;
                }
                defunc_0_lifted_lambda_res_81967 = r_81969;
                // futhark/microgpt.fut:278:116-247
                
                double zp_res_81974 = zp_res_81966 + defunc_0_lifted_lambda_res_81967;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81981;
                double r_81983 = 0.0;
                
                for (int64_t i_81982 = 0; i_81982 < (int64_t) 16; i_81982++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81984 = ((double *) mem_84575)[i_81982 * (int64_t) 16 + i_83011];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81985 = ((double *) mem_83563)[i_81982 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:296:74-109
                    
                    double zt_res_81986 = zt_lhs_81984 * zt_rhs_81985;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81987 = r_81983 + zt_res_81986;
                    double r_tmp_85397 = zp_res_81987;
                    
                    r_81983 = r_tmp_85397;
                }
                defunc_0_lifted_lambda_res_81981 = r_81983;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81997;
                double r_81999 = 0.0;
                
                for (int64_t i_81998 = 0; i_81998 < (int64_t) 16; i_81998++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82000 = ((double *) mem_84576)[i_81998 * (int64_t) 16 + i_83011];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82001 = ((double *) mem_83563)[i_81998 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:297:74-109
                    
                    double zt_res_82002 = zt_lhs_82000 * zt_rhs_82001;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82003 = r_81999 + zt_res_82002;
                    double r_tmp_85398 = zp_res_82003;
                    
                    r_81999 = r_tmp_85398;
                }
                defunc_0_lifted_lambda_res_81997 = r_81999;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82015;
                double r_82017 = 0.0;
                
                for (int64_t i_82016 = 0; i_82016 < (int64_t) 16; i_82016++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82018 = ((double *) mem_84577)[i_82016 * (int64_t) 16 + i_83011];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82019 = ((double *) mem_83563)[i_82016 * (int64_t) 16 + i_82998];
                    
                    // futhark/microgpt.fut:298:74-109
                    
                    double zt_res_82020 = zt_lhs_82018 * zt_rhs_82019;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82021 = r_82017 + zt_res_82020;
                    double r_tmp_85399 = zp_res_82021;
                    
                    r_82017 = r_tmp_85399;
                }
                defunc_0_lifted_lambda_res_82015 = r_82017;
                ((double *) mem_84643)[i_82998] = defunc_0_lifted_lambda_res_82015;
                ((double *) mem_84644)[i_82998] = defunc_0_lifted_lambda_res_81997;
                ((double *) mem_84645)[i_82998] = defunc_0_lifted_lambda_res_81981;
                ((double *) mem_84646)[i_82998] = zp_res_81974;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84623, i_83011 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84624, i_83011 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84644, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84625, i_83011 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84645, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84626, i_83011 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84646, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83018 = 0; i_83018 < (int64_t) 16; i_83018++) {
            // futhark/microgpt.fut:282:49-59
            
            double zs_rhs_78095 = ((double *) mem_83859)[i_83018];
            
            // futhark/microgpt.fut:282:41-59
            
            double zs_res_78096 = 1.0 / zs_rhs_78095;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78097;
            double r_78099 = 0.0;
            
            for (int64_t i_78098 = 0; i_78098 < (int64_t) 16; i_78098++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78100 = ((double *) mem_83530)[i_83018 * (int64_t) 16 + i_78098];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78101 = ((double *) mem_84626)[i_83018 * (int64_t) 16 + i_78098];
                
                // futhark/microgpt.fut:282:87-122
                
                double zt_res_78102 = zt_lhs_78100 * zt_rhs_78101;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78103 = r_78099 + zt_res_78102;
                double r_tmp_85401 = zp_res_78103;
                
                r_78099 = r_tmp_85401;
            }
            defunc_0_lifted_lambda_res_78097 = r_78099;
            // futhark/microgpt.fut:282:67-149
            
            double zt_res_78104 = zs_res_78096 * defunc_0_lifted_lambda_res_78097;
            
            // futhark/microgpt.fut:282:45-149
            
            double zt_res_78105 = zs_res_78096 * zt_res_78104;
            
            // futhark/microgpt.fut:282:33-149
            
            double neg_res_78106 = -zt_res_78105;
            
            ((double *) mem_84687)[i_83018] = neg_res_78106;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83022 = 0; i_83022 < (int64_t) 16; i_83022++) {
            // futhark/microgpt.fut:283:33-43
            
            double zt_lhs_78114 = ((double *) mem_84687)[i_83022];
            
            // futhark/microgpt.fut:283:85-95
            
            double zp_lhs_78115 = ((double *) mem_83601)[i_83022];
            
            // futhark/microgpt.fut:283:85-123
            
            double zp_res_78116 = 1.0e-5 + zp_lhs_78115;
            
            // futhark/microgpt.fut:283:77-123
            
            double sqrt_res_78117 = futrts_sqrt64(zp_res_78116);
            
            // futhark/microgpt.fut:283:63-125
            
            double zt_res_78118 = 2.0 * sqrt_res_78117;
            
            // futhark/microgpt.fut:283:49-125
            
            double zs_res_78119 = 1.0 / zt_res_78118;
            
            // futhark/microgpt.fut:283:33-125
            
            double zt_res_78120 = zt_lhs_78114 * zs_res_78119;
            
            ((double *) mem_84694)[i_83022] = zt_res_78120;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83026 = 0; i_83026 < (int64_t) 16; i_83026++) {
            // futhark/microgpt.fut:284:53-63
            
            double zs_lhs_78128 = ((double *) mem_84694)[i_83026];
            
            // futhark/microgpt.fut:284:53-78
            
            double zs_res_78129 = zs_lhs_78128 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85404 = 0; nest_i_85404 < (int64_t) 16; nest_i_85404++) {
                ((double *) mem_84701)[i_83026 * (int64_t) 16 + nest_i_85404] = zs_res_78129;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83034 = 0; i_83034 < (int64_t) 16; i_83034++) {
            // futhark/microgpt.fut:285:107-117
            
            double zs_rhs_78138 = ((double *) mem_83859)[i_83034];
            
            // futhark/microgpt.fut:285:99-117
            
            double zs_res_78139 = 1.0 / zs_rhs_78138;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83030 = 0; i_83030 < (int64_t) 16; i_83030++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78146 = ((double *) mem_84227)[i_83034 * (int64_t) 16 + i_83030];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78147 = ((double *) mem_84626)[i_83034 * (int64_t) 16 + i_83030];
                
                // futhark/microgpt.fut:285:77-117
                
                double zt_res_78148 = zs_res_78139 * zt_lhs_78147;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78149 = ((double *) mem_83530)[i_83034 * (int64_t) 16 + i_83030];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78150 = ((double *) mem_84701)[i_83034 * (int64_t) 16 + i_83030];
                
                // futhark/microgpt.fut:285:125-160
                
                double zt_res_78151 = zt_lhs_78149 * zt_rhs_78150;
                
                // futhark/microgpt.fut:285:94-160
                
                double zp_res_78152 = zt_res_78148 + zt_res_78151;
                
                // futhark/microgpt.fut:285:120-203
                
                double zp_res_78153 = zt_res_78151 + zp_res_78152;
                
                // futhark/microgpt.fut:285:53-203
                
                double zp_res_78154 = zp_lhs_78146 + zp_res_78153;
                
                ((double *) mem_84716)[i_83030] = zp_res_78154;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84711, i_83034 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84716, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83038 = 0; i_83038 < (int64_t) 16; i_83038++) {
            // futhark/microgpt.fut:289:49-59
            
            double zs_rhs_78202 = ((double *) mem_83600)[i_83038];
            
            // futhark/microgpt.fut:289:41-59
            
            double zs_res_78203 = 1.0 / zs_rhs_78202;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78204;
            double r_78206 = 0.0;
            
            for (int64_t i_78205 = 0; i_78205 < (int64_t) 16; i_78205++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78207 = ((double *) mem_83498)[i_83038 * (int64_t) 16 + i_78205];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78208 = ((double *) mem_84711)[i_83038 * (int64_t) 16 + i_78205];
                
                // futhark/microgpt.fut:289:87-122
                
                double zt_res_78209 = zt_lhs_78207 * zt_rhs_78208;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78210 = r_78206 + zt_res_78209;
                double r_tmp_85408 = zp_res_78210;
                
                r_78206 = r_tmp_85408;
            }
            defunc_0_lifted_lambda_res_78204 = r_78206;
            // futhark/microgpt.fut:289:67-149
            
            double zt_res_78211 = zs_res_78203 * defunc_0_lifted_lambda_res_78204;
            
            // futhark/microgpt.fut:289:45-149
            
            double zt_res_78212 = zs_res_78203 * zt_res_78211;
            
            // futhark/microgpt.fut:289:33-149
            
            double neg_res_78213 = -zt_res_78212;
            
            ((double *) mem_84727)[i_83038] = neg_res_78213;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83042 = 0; i_83042 < (int64_t) 16; i_83042++) {
            // futhark/microgpt.fut:290:33-43
            
            double zt_lhs_78221 = ((double *) mem_84727)[i_83042];
            
            // futhark/microgpt.fut:290:85-95
            
            double zp_lhs_78222 = ((double *) mem_83561)[i_83042];
            
            // futhark/microgpt.fut:290:85-123
            
            double zp_res_78223 = 1.0e-5 + zp_lhs_78222;
            
            // futhark/microgpt.fut:290:77-123
            
            double sqrt_res_78224 = futrts_sqrt64(zp_res_78223);
            
            // futhark/microgpt.fut:290:63-125
            
            double zt_res_78225 = 2.0 * sqrt_res_78224;
            
            // futhark/microgpt.fut:290:49-125
            
            double zs_res_78226 = 1.0 / zt_res_78225;
            
            // futhark/microgpt.fut:290:33-125
            
            double zt_res_78227 = zt_lhs_78221 * zs_res_78226;
            
            ((double *) mem_84734)[i_83042] = zt_res_78227;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83046 = 0; i_83046 < (int64_t) 16; i_83046++) {
            // futhark/microgpt.fut:291:53-63
            
            double zs_lhs_78235 = ((double *) mem_84734)[i_83046];
            
            // futhark/microgpt.fut:291:53-78
            
            double zs_res_78236 = zs_lhs_78235 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85411 = 0; nest_i_85411 < (int64_t) 16; nest_i_85411++) {
                ((double *) mem_84741)[i_83046 * (int64_t) 16 + nest_i_85411] = zs_res_78236;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83054 = 0; i_83054 < (int64_t) 16; i_83054++) {
            // futhark/microgpt.fut:292:85-95
            
            double zs_rhs_78245 = ((double *) mem_83600)[i_83054];
            
            // futhark/microgpt.fut:292:77-95
            
            double zs_res_78246 = 1.0 / zs_rhs_78245;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83050 = 0; i_83050 < (int64_t) 16; i_83050++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78253 = ((double *) mem_84711)[i_83054 * (int64_t) 16 + i_83050];
                
                // futhark/microgpt.fut:292:55-95
                
                double zt_res_78254 = zs_res_78246 * zt_lhs_78253;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78255 = ((double *) mem_83498)[i_83054 * (int64_t) 16 + i_83050];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78256 = ((double *) mem_84741)[i_83054 * (int64_t) 16 + i_83050];
                
                // futhark/microgpt.fut:292:103-138
                
                double zt_res_78257 = zt_lhs_78255 * zt_rhs_78256;
                
                // futhark/microgpt.fut:292:72-138
                
                double zp_res_78258 = zt_res_78254 + zt_res_78257;
                
                // futhark/microgpt.fut:292:98-181
                
                double zp_res_78259 = zt_res_78257 + zp_res_78258;
                
                ((double *) mem_84756)[i_83050] = zp_res_78259;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84751, i_83054 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84756, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83067 = 0; i_83067 < (int64_t) 16; i_83067++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83060 = 0; i_83060 < (int64_t) 16; i_83060++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82047 = ((double *) mem_84751)[i_83067 * (int64_t) 16 + i_83060];
                
                ((double *) mem_84777)[i_83060] = lifted_lambda_res_82047;
                ((double *) mem_84778)[i_83060] = lifted_lambda_res_82047;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84767, i_83067 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84777, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84768, i_83067 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84778, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83076 = 0; i_83076 < (int64_t) 64; i_83076++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83072 = 0; i_83072 < (int64_t) 16; i_83072++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_78373;
                double r_78375 = 0.0;
                
                for (int64_t i_78374 = 0; i_78374 < (int64_t) 16; i_78374++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_78376 = ((double *) mem_84171)[i_78374 * (int64_t) 64 + i_83076];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_78377 = ((double *) mem_83915)[i_78374 * (int64_t) 16 + i_83072];
                    
                    // futhark/microgpt.fut:300:73-109
                    
                    double zt_res_78378 = zt_lhs_78376 * zt_rhs_78377;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_78379 = r_78375 + zt_res_78378;
                    double r_tmp_85420 = zp_res_78379;
                    
                    r_78375 = r_tmp_85420;
                }
                defunc_0_lifted_lambda_res_78373 = r_78375;
                ((double *) mem_84804)[i_83072] = defunc_0_lifted_lambda_res_78373;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84799, i_83076 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84804, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83089 = 0; i_83089 < (int64_t) 27; i_83089++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83082 = 0; i_83082 < (int64_t) 16; i_83082++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82075;
                double r_82077 = 0.0;
                
                for (int64_t i_82076 = 0; i_82076 < (int64_t) 16; i_82076++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82078 = ((double *) mem_84107)[i_82076 * (int64_t) 27 + i_83089];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82079 = ((double *) mem_84008)[i_82076 * (int64_t) 16 + i_83082];
                    
                    // futhark/microgpt.fut:302:74-110
                    
                    double zt_res_82080 = zt_lhs_82078 * zt_rhs_82079;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82081 = r_82077 + zt_res_82080;
                    double r_tmp_85425 = zp_res_82081;
                    
                    r_82077 = r_tmp_85425;
                }
                defunc_0_lifted_lambda_res_82075 = r_82077;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82084;
                double r_82086 = 0.0;
                
                for (int64_t i_82085 = 0; i_82085 < (int64_t) 16; i_82085++) {
                    int64_t zeze_lhs_82087 = ((int64_t *) seqs_mem_83356.mem)[step_76681 * (int64_t) 16 + i_82085];
                    
                    // futhark/microgpt.fut:416:58-109
                    
                    bool cond_82088 = zeze_lhs_82087 == i_83089;
                    
                    // futhark/microgpt.fut:416:58-109
                    
                    double lifted_lambda_res_82089;
                    
                    if (cond_82088) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_82387 = ((double *) mem_84767)[i_82085 * (int64_t) 16 + i_83082];
                        
                        lifted_lambda_res_82089 = lifted_lambda_res_t_res_82387;
                    } else {
                        lifted_lambda_res_82089 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82095 = r_82086 + lifted_lambda_res_82089;
                    double r_tmp_85426 = zp_res_82095;
                    
                    r_82086 = r_tmp_85426;
                }
                defunc_0_lifted_lambda_res_82084 = r_82086;
                ((double *) mem_84825)[i_83082] = defunc_0_lifted_lambda_res_82084;
                ((double *) mem_84826)[i_83082] = defunc_0_lifted_lambda_res_82075;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84815, i_83089 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84825, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84816, i_83089 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84826, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_78457 = sitofp_i64_f64(step_76681);
        
        // futhark/microgpt.fut:372:46-57
        
        double zm_rhs_78458 = i64_res_78457 / i64_res_76653;
        
        // futhark/microgpt.fut:372:24-57
        
        double zt_rhs_78459 = 1.0 - zm_rhs_78458;
        
        // futhark/microgpt.fut:372:19-57
        
        double lt_r_78460 = 1.0e-2 * zt_rhs_78459;
        
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84847, (int64_t) 3456, "mem_84847")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84847.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83380.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84849, (int64_t) 3456, "mem_84849")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84849.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83416.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84851, (int64_t) 3456, "mem_84851")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84851.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83452.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84853, (int64_t) 3456, "mem_84853")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84853.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84815, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (futrts_adam_opt_w_10345(ctx, &ext_mem_84857, &ext_mem_84856, &ext_mem_84855, mem_84847, mem_84849, mem_84851, mem_84853, (int64_t) 27, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84847, "mem_84847") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84849, "mem_84849") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84851, "mem_84851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84853, "mem_84853") != 0)
            return 1;
        // futhark/microgpt.fut:376:5-52
        if (memblock_alloc(ctx, &mem_84858, (int64_t) 2048, "mem_84858")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-52
        // futhark/microgpt.fut:376:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84858.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83372.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-52
        if (memblock_alloc(ctx, &mem_84860, (int64_t) 2048, "mem_84860")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-52
        // futhark/microgpt.fut:376:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84860.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83408.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-52
        if (memblock_alloc(ctx, &mem_84862, (int64_t) 2048, "mem_84862")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-52
        // futhark/microgpt.fut:376:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84862.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83444.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-52
        if (memblock_alloc(ctx, &mem_84864, (int64_t) 2048, "mem_84864")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-52
        // futhark/microgpt.fut:376:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84864.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84768, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-52
        if (futrts_adam_opt_w_10346(ctx, &ext_mem_84868, &ext_mem_84867, &ext_mem_84866, mem_84858, mem_84860, mem_84862, mem_84864, (int64_t) 16, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84858, "mem_84858") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84860, "mem_84860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84862, "mem_84862") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84864, "mem_84864") != 0)
            return 1;
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84869, (int64_t) 2048, "mem_84869")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84869.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83376.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84871, (int64_t) 2048, "mem_84871")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84871.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83412.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84873, (int64_t) 2048, "mem_84873")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84873.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83448.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84875, (int64_t) 2048, "mem_84875")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84875.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84625, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (futrts_adam_opt_w_10346(ctx, &ext_mem_84879, &ext_mem_84878, &ext_mem_84877, mem_84869, mem_84871, mem_84873, mem_84875, (int64_t) 16, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84869, "mem_84869") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84871, "mem_84871") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84873, "mem_84873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84875, "mem_84875") != 0)
            return 1;
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84880, (int64_t) 2048, "mem_84880")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84880.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83364.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84882, (int64_t) 2048, "mem_84882")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84882.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83400.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84884, (int64_t) 2048, "mem_84884")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84884.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83436.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84886, (int64_t) 2048, "mem_84886")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84886.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84624, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (futrts_adam_opt_w_10346(ctx, &ext_mem_84890, &ext_mem_84889, &ext_mem_84888, mem_84880, mem_84882, mem_84884, mem_84886, (int64_t) 16, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84880, "mem_84880") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84882, "mem_84882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84884, "mem_84884") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84886, "mem_84886") != 0)
            return 1;
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84891, (int64_t) 2048, "mem_84891")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84891.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83388.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84893, (int64_t) 2048, "mem_84893")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84893.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83424.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84895, (int64_t) 2048, "mem_84895")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84895.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83460.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84897, (int64_t) 2048, "mem_84897")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84897.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84623, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (futrts_adam_opt_w_10346(ctx, &ext_mem_84901, &ext_mem_84900, &ext_mem_84899, mem_84891, mem_84893, mem_84895, mem_84897, (int64_t) 16, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84891, "mem_84891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84893, "mem_84893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84895, "mem_84895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84897, "mem_84897") != 0)
            return 1;
        // futhark/microgpt.fut:384:5-56
        if (memblock_alloc(ctx, &mem_84902, (int64_t) 2048, "mem_84902")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-56
        // futhark/microgpt.fut:384:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84902.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83368.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:384:5-56
        if (memblock_alloc(ctx, &mem_84904, (int64_t) 2048, "mem_84904")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-56
        // futhark/microgpt.fut:384:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84904.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83404.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:384:5-56
        if (memblock_alloc(ctx, &mem_84906, (int64_t) 2048, "mem_84906")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-56
        // futhark/microgpt.fut:384:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84906.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83440.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:384:5-56
        if (memblock_alloc(ctx, &mem_84908, (int64_t) 2048, "mem_84908")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-56
        // futhark/microgpt.fut:384:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84908.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84243, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:384:5-56
        if (futrts_adam_opt_w_10346(ctx, &ext_mem_84912, &ext_mem_84911, &ext_mem_84910, mem_84902, mem_84904, mem_84906, mem_84908, (int64_t) 16, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84902, "mem_84902") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84904, "mem_84904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84906, "mem_84906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84908, "mem_84908") != 0)
            return 1;
        // futhark/microgpt.fut:386:5-52
        if (memblock_alloc(ctx, &mem_84913, (int64_t) 8192, "mem_84913")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-52
        // futhark/microgpt.fut:386:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84913.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83384.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:386:5-52
        if (memblock_alloc(ctx, &mem_84915, (int64_t) 8192, "mem_84915")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-52
        // futhark/microgpt.fut:386:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84915.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83420.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:386:5-52
        if (memblock_alloc(ctx, &mem_84917, (int64_t) 8192, "mem_84917")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-52
        // futhark/microgpt.fut:386:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84917.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83456.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:386:5-52
        if (memblock_alloc(ctx, &mem_84919, (int64_t) 8192, "mem_84919")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-52
        // futhark/microgpt.fut:386:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84919.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84799, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:386:5-52
        if (futrts_adam_opt_w_10345(ctx, &ext_mem_84923, &ext_mem_84922, &ext_mem_84921, mem_84913, mem_84915, mem_84917, mem_84919, (int64_t) 64, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84913, "mem_84913") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84915, "mem_84915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84917, "mem_84917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84919, "mem_84919") != 0)
            return 1;
        // futhark/microgpt.fut:388:5-60
        if (memblock_alloc(ctx, &mem_84924, (int64_t) 8192, "mem_84924")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-60
        // futhark/microgpt.fut:388:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84924.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83360.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:388:5-60
        if (memblock_alloc(ctx, &mem_84926, (int64_t) 8192, "mem_84926")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-60
        // futhark/microgpt.fut:388:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84926.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83396.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:388:5-60
        if (memblock_alloc(ctx, &mem_84928, (int64_t) 8192, "mem_84928")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-60
        // futhark/microgpt.fut:388:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84928.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83432.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:388:5-60
        if (memblock_alloc(ctx, &mem_84930, (int64_t) 8192, "mem_84930")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-60
        // futhark/microgpt.fut:388:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84930.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_84139, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:388:5-60
        if (futrts_adam_opt_w_10345(ctx, &ext_mem_84934, &ext_mem_84933, &ext_mem_84932, mem_84924, mem_84926, mem_84928, mem_84930, (int64_t) 16, (int64_t) 64, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84924, "mem_84924") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84926, "mem_84926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84928, "mem_84928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84930, "mem_84930") != 0)
            return 1;
        // futhark/microgpt.fut:390:5-56
        if (memblock_alloc(ctx, &mem_84935, (int64_t) 3456, "mem_84935")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:390:5-56
        // futhark/microgpt.fut:390:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84935.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83392.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:390:5-56
        if (memblock_alloc(ctx, &mem_84937, (int64_t) 3456, "mem_84937")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:390:5-56
        // futhark/microgpt.fut:390:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84937.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83428.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:390:5-56
        if (memblock_alloc(ctx, &mem_84939, (int64_t) 3456, "mem_84939")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:390:5-56
        // futhark/microgpt.fut:390:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84939.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83464.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:390:5-56
        if (memblock_alloc(ctx, &mem_84941, (int64_t) 3456, "mem_84941")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:390:5-56
        // futhark/microgpt.fut:390:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84941.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84816, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:390:5-56
        if (futrts_adam_opt_w_10345(ctx, &ext_mem_84945, &ext_mem_84944, &ext_mem_84943, mem_84935, mem_84937, mem_84939, mem_84941, (int64_t) 27, (int64_t) 16, step_76681, lt_r_78460) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84935, "mem_84935") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84937, "mem_84937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84939, "mem_84939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84941, "mem_84941") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85153, &ext_mem_84934, "ext_mem_84934") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85154, &ext_mem_84890, "ext_mem_84890") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85155, &ext_mem_84912, "ext_mem_84912") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85156, &ext_mem_84868, "ext_mem_84868") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85157, &ext_mem_84879, "ext_mem_84879") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85158, &ext_mem_84857, "ext_mem_84857") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85159, &ext_mem_84923, "ext_mem_84923") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85160, &ext_mem_84901, "ext_mem_84901") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85161, &ext_mem_84945, "ext_mem_84945") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85162, &ext_mem_84933, "ext_mem_84933") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85163, &ext_mem_84889, "ext_mem_84889") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85164, &ext_mem_84911, "ext_mem_84911") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85165, &ext_mem_84867, "ext_mem_84867") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85166, &ext_mem_84878, "ext_mem_84878") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85167, &ext_mem_84856, "ext_mem_84856") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85168, &ext_mem_84922, "ext_mem_84922") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85169, &ext_mem_84900, "ext_mem_84900") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85170, &ext_mem_84944, "ext_mem_84944") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85171, &ext_mem_84932, "ext_mem_84932") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85172, &ext_mem_84888, "ext_mem_84888") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85173, &ext_mem_84910, "ext_mem_84910") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85174, &ext_mem_84866, "ext_mem_84866") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85175, &ext_mem_84877, "ext_mem_84877") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85176, &ext_mem_84855, "ext_mem_84855") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85177, &ext_mem_84921, "ext_mem_84921") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85178, &ext_mem_84899, "ext_mem_84899") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85179, &ext_mem_84943, "ext_mem_84943") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83360, &mem_param_tmp_85153, "mem_param_tmp_85153") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83364, &mem_param_tmp_85154, "mem_param_tmp_85154") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83368, &mem_param_tmp_85155, "mem_param_tmp_85155") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83372, &mem_param_tmp_85156, "mem_param_tmp_85156") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83376, &mem_param_tmp_85157, "mem_param_tmp_85157") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83380, &mem_param_tmp_85158, "mem_param_tmp_85158") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83384, &mem_param_tmp_85159, "mem_param_tmp_85159") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83388, &mem_param_tmp_85160, "mem_param_tmp_85160") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83392, &mem_param_tmp_85161, "mem_param_tmp_85161") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83396, &mem_param_tmp_85162, "mem_param_tmp_85162") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83400, &mem_param_tmp_85163, "mem_param_tmp_85163") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83404, &mem_param_tmp_85164, "mem_param_tmp_85164") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83408, &mem_param_tmp_85165, "mem_param_tmp_85165") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83412, &mem_param_tmp_85166, "mem_param_tmp_85166") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83416, &mem_param_tmp_85167, "mem_param_tmp_85167") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83420, &mem_param_tmp_85168, "mem_param_tmp_85168") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83424, &mem_param_tmp_85169, "mem_param_tmp_85169") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83428, &mem_param_tmp_85170, "mem_param_tmp_85170") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83432, &mem_param_tmp_85171, "mem_param_tmp_85171") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83436, &mem_param_tmp_85172, "mem_param_tmp_85172") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83440, &mem_param_tmp_85173, "mem_param_tmp_85173") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83444, &mem_param_tmp_85174, "mem_param_tmp_85174") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83448, &mem_param_tmp_85175, "mem_param_tmp_85175") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83452, &mem_param_tmp_85176, "mem_param_tmp_85176") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83456, &mem_param_tmp_85177, "mem_param_tmp_85177") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83460, &mem_param_tmp_85178, "mem_param_tmp_85178") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83464, &mem_param_tmp_85179, "mem_param_tmp_85179") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_85053, &mem_param_83360, "mem_param_83360") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85052, &mem_param_83364, "mem_param_83364") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85051, &mem_param_83368, "mem_param_83368") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85050, &mem_param_83372, "mem_param_83372") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85049, &mem_param_83376, "mem_param_83376") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85048, &mem_param_83380, "mem_param_83380") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85047, &mem_param_83384, "mem_param_83384") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85046, &mem_param_83388, "mem_param_83388") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85045, &mem_param_83392, "mem_param_83392") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85044, &mem_param_83396, "mem_param_83396") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85043, &mem_param_83400, "mem_param_83400") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85042, &mem_param_83404, "mem_param_83404") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85041, &mem_param_83408, "mem_param_83408") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85040, &mem_param_83412, "mem_param_83412") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85039, &mem_param_83416, "mem_param_83416") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85038, &mem_param_83420, "mem_param_83420") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85037, &mem_param_83424, "mem_param_83424") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85036, &mem_param_83428, "mem_param_83428") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85035, &mem_param_83432, "mem_param_83432") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85034, &mem_param_83436, "mem_param_83436") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85033, &mem_param_83440, "mem_param_83440") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85032, &mem_param_83444, "mem_param_83444") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85031, &mem_param_83448, "mem_param_83448") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85030, &mem_param_83452, "mem_param_83452") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85029, &mem_param_83456, "mem_param_83456") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85028, &mem_param_83460, "mem_param_83460") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85027, &mem_param_83464, "mem_param_83464") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85126, &ext_mem_85048, "ext_mem_85048") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &ext_mem_85050, "ext_mem_85050") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &ext_mem_85049, "ext_mem_85049") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85129, &ext_mem_85052, "ext_mem_85052") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85130, &ext_mem_85046, "ext_mem_85046") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85131, &ext_mem_85051, "ext_mem_85051") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85132, &ext_mem_85047, "ext_mem_85047") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85133, &ext_mem_85053, "ext_mem_85053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85134, &ext_mem_85045, "ext_mem_85045") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85135, &ext_mem_85039, "ext_mem_85039") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85136, &ext_mem_85041, "ext_mem_85041") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85137, &ext_mem_85040, "ext_mem_85040") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85138, &ext_mem_85043, "ext_mem_85043") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85139, &ext_mem_85037, "ext_mem_85037") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85140, &ext_mem_85042, "ext_mem_85042") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85141, &ext_mem_85038, "ext_mem_85038") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85142, &ext_mem_85044, "ext_mem_85044") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &ext_mem_85036, "ext_mem_85036") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &ext_mem_85030, "ext_mem_85030") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85145, &ext_mem_85032, "ext_mem_85032") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85146, &ext_mem_85031, "ext_mem_85031") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85147, &ext_mem_85034, "ext_mem_85034") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85148, &ext_mem_85028, "ext_mem_85028") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85149, &ext_mem_85033, "ext_mem_85033") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85150, &ext_mem_85029, "ext_mem_85029") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85151, &ext_mem_85035, "ext_mem_85035") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85152, &ext_mem_85027, "ext_mem_85027") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85519, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85520, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85521, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85522, &mem_out_85129, "mem_out_85129") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85523, &mem_out_85130, "mem_out_85130") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85524, &mem_out_85131, "mem_out_85131") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85525, &mem_out_85132, "mem_out_85132") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85526, &mem_out_85133, "mem_out_85133") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85527, &mem_out_85134, "mem_out_85134") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85528, &mem_out_85135, "mem_out_85135") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85529, &mem_out_85136, "mem_out_85136") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85530, &mem_out_85137, "mem_out_85137") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85531, &mem_out_85138, "mem_out_85138") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85532, &mem_out_85139, "mem_out_85139") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85533, &mem_out_85140, "mem_out_85140") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85534, &mem_out_85141, "mem_out_85141") != 0)
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
    
  cleanup:
    {
        free(mem_83465);
        free(mem_83466);
        free(mem_83475);
        free(mem_83482);
        free(mem_83497);
        free(mem_83498);
        free(mem_83507);
        free(mem_83514);
        free(mem_83529);
        free(mem_83530);
        free(mem_83539);
        free(mem_83540);
        free(mem_83561);
        free(mem_83562);
        free(mem_83563);
        free(mem_83575);
        free(mem_83576);
        free(mem_83600);
        free(mem_83601);
        free(mem_83602);
        free(mem_83603);
        free(mem_83604);
        free(mem_83623);
        free(mem_83624);
        free(mem_83625);
        free(mem_83662);
        free(mem_83663);
        free(mem_83664);
        free(mem_83680);
        free(mem_83681);
        free(mem_83682);
        free(mem_83695);
        free(mem_83696);
        free(mem_83697);
        free(mem_83743);
        free(mem_83744);
        free(mem_83755);
        free(mem_83756);
        free(mem_83765);
        free(mem_83766);
        free(mem_83787);
        free(mem_83792);
        free(mem_83803);
        free(mem_83808);
        free(mem_83815);
        free(mem_83822);
        free(mem_83833);
        free(mem_83838);
        free(mem_83859);
        free(mem_83860);
        free(mem_83868);
        free(mem_83882);
        free(mem_83887);
        free(mem_83898);
        free(mem_83903);
        free(mem_83914);
        free(mem_83915);
        free(mem_83924);
        free(mem_83925);
        free(mem_83946);
        free(mem_83947);
        free(mem_83955);
        free(mem_83969);
        free(mem_83970);
        free(mem_83978);
        free(mem_83992);
        free(mem_83997);
        free(mem_84008);
        free(mem_84013);
        free(mem_84024);
        free(mem_84029);
        free(mem_84040);
        free(mem_84041);
        free(mem_84050);
        free(mem_84051);
        free(mem_84064);
        free(mem_84065);
        free(mem_84078);
        free(mem_84079);
        free(mem_84100);
        free(mem_84107);
        free(mem_84112);
        free(mem_84123);
        free(mem_84128);
        free(mem_84139);
        free(mem_84140);
        free(mem_84149);
        free(mem_84150);
        free(mem_84171);
        free(mem_84176);
        free(mem_84187);
        free(mem_84192);
        free(mem_84203);
        free(mem_84210);
        free(mem_84217);
        free(mem_84227);
        free(mem_84232);
        free(mem_84243);
        free(mem_84244);
        free(mem_84253);
        free(mem_84254);
        free(mem_84275);
        free(mem_84276);
        free(mem_84287);
        free(mem_84288);
        free(mem_84297);
        free(mem_84304);
        free(mem_84329);
        free(mem_84330);
        free(mem_84341);
        free(mem_84342);
        free(mem_84351);
        free(mem_84358);
        free(mem_84365);
        free(mem_84372);
        free(mem_84397);
        free(mem_84398);
        free(mem_84409);
        free(mem_84410);
        free(mem_84419);
        free(mem_84426);
        free(mem_84451);
        free(mem_84456);
        free(mem_84467);
        free(mem_84473);
        free(mem_84478);
        free(mem_84494);
        free(mem_84500);
        free(mem_84505);
        free(mem_84521);
        free(mem_84522);
        free(mem_84533);
        free(mem_84534);
        free(mem_84543);
        free(mem_84544);
        free(mem_84575);
        free(mem_84576);
        free(mem_84577);
        free(mem_84590);
        free(mem_84591);
        free(mem_84592);
        free(mem_84623);
        free(mem_84624);
        free(mem_84625);
        free(mem_84626);
        free(mem_84643);
        free(mem_84644);
        free(mem_84645);
        free(mem_84646);
        free(mem_84687);
        free(mem_84694);
        free(mem_84701);
        free(mem_84711);
        free(mem_84716);
        free(mem_84727);
        free(mem_84734);
        free(mem_84741);
        free(mem_84751);
        free(mem_84756);
        free(mem_84767);
        free(mem_84768);
        free(mem_84777);
        free(mem_84778);
        free(mem_84799);
        free(mem_84804);
        free(mem_84815);
        free(mem_84816);
        free(mem_84825);
        free(mem_84826);
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
        if (memblock_unref(ctx, &mem_param_tmp_85168, "mem_param_tmp_85168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85167, "mem_param_tmp_85167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85166, "mem_param_tmp_85166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85165, "mem_param_tmp_85165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85164, "mem_param_tmp_85164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85163, "mem_param_tmp_85163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85162, "mem_param_tmp_85162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85161, "mem_param_tmp_85161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85160, "mem_param_tmp_85160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85159, "mem_param_tmp_85159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85158, "mem_param_tmp_85158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85157, "mem_param_tmp_85157") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85156, "mem_param_tmp_85156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85155, "mem_param_tmp_85155") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85154, "mem_param_tmp_85154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85153, "mem_param_tmp_85153") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84943, "ext_mem_84943") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84944, "ext_mem_84944") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84945, "ext_mem_84945") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84941, "mem_84941") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84939, "mem_84939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84937, "mem_84937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84935, "mem_84935") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84932, "ext_mem_84932") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84933, "ext_mem_84933") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84934, "ext_mem_84934") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84930, "mem_84930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84928, "mem_84928") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84926, "mem_84926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84924, "mem_84924") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84921, "ext_mem_84921") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84922, "ext_mem_84922") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84923, "ext_mem_84923") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84919, "mem_84919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84917, "mem_84917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84915, "mem_84915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84913, "mem_84913") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84910, "ext_mem_84910") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84911, "ext_mem_84911") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84912, "ext_mem_84912") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84908, "mem_84908") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84906, "mem_84906") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84904, "mem_84904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84902, "mem_84902") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84899, "ext_mem_84899") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84900, "ext_mem_84900") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84901, "ext_mem_84901") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84897, "mem_84897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84895, "mem_84895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84893, "mem_84893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84891, "mem_84891") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84888, "ext_mem_84888") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84889, "ext_mem_84889") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84890, "ext_mem_84890") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84886, "mem_84886") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84884, "mem_84884") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84882, "mem_84882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84880, "mem_84880") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84877, "ext_mem_84877") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84878, "ext_mem_84878") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84879, "ext_mem_84879") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84875, "mem_84875") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84873, "mem_84873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84871, "mem_84871") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84869, "mem_84869") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84866, "ext_mem_84866") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84867, "ext_mem_84867") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84868, "ext_mem_84868") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84864, "mem_84864") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84862, "mem_84862") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84860, "mem_84860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84858, "mem_84858") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84855, "ext_mem_84855") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84856, "ext_mem_84856") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84857, "ext_mem_84857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84853, "mem_84853") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84851, "mem_84851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84849, "mem_84849") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84847, "mem_84847") != 0)
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
        if (memblock_unref(ctx, &mem_param_83372, "mem_param_83372") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83368, "mem_param_83368") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83364, "mem_param_83364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83360, "mem_param_83360") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85027, "ext_mem_85027") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85028, "ext_mem_85028") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85029, "ext_mem_85029") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85030, "ext_mem_85030") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85031, "ext_mem_85031") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85032, "ext_mem_85032") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85033, "ext_mem_85033") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85034, "ext_mem_85034") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85035, "ext_mem_85035") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85036, "ext_mem_85036") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85037, "ext_mem_85037") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85038, "ext_mem_85038") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85039, "ext_mem_85039") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85040, "ext_mem_85040") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85041, "ext_mem_85041") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85042, "ext_mem_85042") != 0)
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
        if (memblock_unref(ctx, &mem_out_85141, "mem_out_85141") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85140, "mem_out_85140") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85139, "mem_out_85139") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85138, "mem_out_85138") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85137, "mem_out_85137") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85136, "mem_out_85136") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85135, "mem_out_85135") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85134, "mem_out_85134") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85133, "mem_out_85133") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85132, "mem_out_85132") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85131, "mem_out_85131") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85130, "mem_out_85130") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85129, "mem_out_85129") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85715, struct memblock *mem_out_p_85716, struct memblock *mem_out_p_85717, struct memblock *mem_out_p_85718, struct memblock *mem_out_p_85719, struct memblock *mem_out_p_85720, struct memblock *mem_out_p_85721, struct memblock *mem_out_p_85722, struct memblock *mem_out_p_85723)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    struct memblock mem_83321 = ctx->constants->mem_83321;
    struct memblock mem_83322 = ctx->constants->mem_83322;
    struct memblock mem_83323 = ctx->constants->mem_83323;
    struct memblock mem_83324 = ctx->constants->mem_83324;
    struct memblock mem_83325 = ctx->constants->mem_83325;
    struct memblock mem_83326 = ctx->constants->mem_83326;
    
    if (memblock_set(ctx, &mem_out_85126, &mem_83325, "mem_83325") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &mem_83321, "mem_83321") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &mem_83323, "mem_83323") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85129, &mem_83319, "mem_83319") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85130, &mem_83320, "mem_83320") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85131, &mem_83318, "mem_83318") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85132, &mem_83324, "mem_83324") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85133, &mem_83322, "mem_83322") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85134, &mem_83326, "mem_83326") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85715, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85716, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85717, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85718, &mem_out_85129, "mem_out_85129") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85719, &mem_out_85130, "mem_out_85130") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85720, &mem_out_85131, "mem_out_85131") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85721, &mem_out_85132, "mem_out_85132") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85722, &mem_out_85133, "mem_out_85133") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85723, &mem_out_85134, "mem_out_85134") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85134, "mem_out_85134") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85133, "mem_out_85133") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85132, "mem_out_85132") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85131, "mem_out_85131") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85130, "mem_out_85130") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85129, "mem_out_85129") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mask_mem_83337;
    
    mask_mem_83337.references = NULL;
    
    struct memblock tokens_mem_83336;
    
    tokens_mem_83336.references = NULL;
    
    struct memblock wvoc_mem_83335;
    
    wvoc_mem_83335.references = NULL;
    
    struct memblock wval_mem_83334;
    
    wval_mem_83334.references = NULL;
    
    struct memblock wup_mem_83333;
    
    wup_mem_83333.references = NULL;
    
    struct memblock wte_mem_83332;
    
    wte_mem_83332.references = NULL;
    
    struct memblock wqry_mem_83331;
    
    wqry_mem_83331.references = NULL;
    
    struct memblock wpe_mem_83330;
    
    wpe_mem_83330.references = NULL;
    
    struct memblock wout_mem_83329;
    
    wout_mem_83329.references = NULL;
    
    struct memblock wkey_mem_83328;
    
    wkey_mem_83328.references = NULL;
    
    struct memblock wdown_mem_83327;
    
    wdown_mem_83327.references = NULL;
    wdown_mem_83327 = in0->v0->mem;
    wkey_mem_83328 = in0->v1->mem;
    wout_mem_83329 = in0->v2->mem;
    wpe_mem_83330 = in0->v3->mem;
    wqry_mem_83331 = in0->v4->mem;
    wte_mem_83332 = in0->v5->mem;
    wup_mem_83333 = in0->v6->mem;
    wval_mem_83334 = in0->v7->mem;
    wvoc_mem_83335 = in0->v8->mem;
    tokens_mem_83336 = in1->mem;
    mask_mem_83337 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_85126, wdown_mem_83327, wkey_mem_83328, wout_mem_83329, wpe_mem_83330, wqry_mem_83331, wte_mem_83332, wup_mem_83333, wval_mem_83334, wvoc_mem_83335, tokens_mem_83336, mask_mem_83337);
        if (ret == 0) {
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            struct memblock mem_83321 = ctx->constants->mem_83321;
            struct memblock mem_83322 = ctx->constants->mem_83322;
            struct memblock mem_83323 = ctx->constants->mem_83323;
            struct memblock mem_83324 = ctx->constants->mem_83324;
            struct memblock mem_83325 = ctx->constants->mem_83325;
            struct memblock mem_83326 = ctx->constants->mem_83326;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_85126;
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
    
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock wvoc_mem_83335;
    
    wvoc_mem_83335.references = NULL;
    
    struct memblock wdown_mem_83334;
    
    wdown_mem_83334.references = NULL;
    
    struct memblock wup_mem_83333;
    
    wup_mem_83333.references = NULL;
    
    struct memblock wout_mem_83332;
    
    wout_mem_83332.references = NULL;
    
    struct memblock wval_mem_83331;
    
    wval_mem_83331.references = NULL;
    
    struct memblock wkey_mem_83330;
    
    wkey_mem_83330.references = NULL;
    
    struct memblock wqry_mem_83329;
    
    wqry_mem_83329.references = NULL;
    
    struct memblock wpe_mem_83328;
    
    wpe_mem_83328.references = NULL;
    
    struct memblock wte_mem_83327;
    
    wte_mem_83327.references = NULL;
    wte_mem_83327 = in0->mem;
    wpe_mem_83328 = in1->mem;
    wqry_mem_83329 = in2->mem;
    wkey_mem_83330 = in3->mem;
    wval_mem_83331 = in4->mem;
    wout_mem_83332 = in5->mem;
    wup_mem_83333 = in6->mem;
    wdown_mem_83334 = in7->mem;
    wvoc_mem_83335 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_85126, &mem_out_85127, &mem_out_85128, &mem_out_85129, &mem_out_85130, &mem_out_85131, &mem_out_85132, &mem_out_85133, &mem_out_85134, wte_mem_83327, wpe_mem_83328, wqry_mem_83329, wkey_mem_83330, wval_mem_83331, wout_mem_83332, wup_mem_83333, wdown_mem_83334, wvoc_mem_83335);
        if (ret == 0) {
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            struct memblock mem_83321 = ctx->constants->mem_83321;
            struct memblock mem_83322 = ctx->constants->mem_83322;
            struct memblock mem_83323 = ctx->constants->mem_83323;
            struct memblock mem_83324 = ctx->constants->mem_83324;
            struct memblock mem_83325 = ctx->constants->mem_83325;
            struct memblock mem_83326 = ctx->constants->mem_83326;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85126;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85127;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85128;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85129;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85130;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85131;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85132;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85133;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85134;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 **out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_opaque_params *in3, const struct futhark_f64_3d *in4, const struct futhark_i64_1d *in5, const struct futhark_i64_2d *in6)
{
    int64_t num_steps_62012 = (int64_t) 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
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
    
    struct memblock mem_out_85141;
    
    mem_out_85141.references = NULL;
    
    struct memblock mem_out_85140;
    
    mem_out_85140.references = NULL;
    
    struct memblock mem_out_85139;
    
    mem_out_85139.references = NULL;
    
    struct memblock mem_out_85138;
    
    mem_out_85138.references = NULL;
    
    struct memblock mem_out_85137;
    
    mem_out_85137.references = NULL;
    
    struct memblock mem_out_85136;
    
    mem_out_85136.references = NULL;
    
    struct memblock mem_out_85135;
    
    mem_out_85135.references = NULL;
    
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock seqs_mem_83356;
    
    seqs_mem_83356.references = NULL;
    
    struct memblock dls_mem_83355;
    
    dls_mem_83355.references = NULL;
    
    struct memblock masks_mem_83354;
    
    masks_mem_83354.references = NULL;
    
    struct memblock wvoc_mem_83353;
    
    wvoc_mem_83353.references = NULL;
    
    struct memblock wval_mem_83352;
    
    wval_mem_83352.references = NULL;
    
    struct memblock wup_mem_83351;
    
    wup_mem_83351.references = NULL;
    
    struct memblock wte_mem_83350;
    
    wte_mem_83350.references = NULL;
    
    struct memblock wqry_mem_83349;
    
    wqry_mem_83349.references = NULL;
    
    struct memblock wpe_mem_83348;
    
    wpe_mem_83348.references = NULL;
    
    struct memblock wout_mem_83347;
    
    wout_mem_83347.references = NULL;
    
    struct memblock wkey_mem_83346;
    
    wkey_mem_83346.references = NULL;
    
    struct memblock wdown_mem_83345;
    
    wdown_mem_83345.references = NULL;
    
    struct memblock wvoc_mem_83344;
    
    wvoc_mem_83344.references = NULL;
    
    struct memblock wval_mem_83343;
    
    wval_mem_83343.references = NULL;
    
    struct memblock wup_mem_83342;
    
    wup_mem_83342.references = NULL;
    
    struct memblock wte_mem_83341;
    
    wte_mem_83341.references = NULL;
    
    struct memblock wqry_mem_83340;
    
    wqry_mem_83340.references = NULL;
    
    struct memblock wpe_mem_83339;
    
    wpe_mem_83339.references = NULL;
    
    struct memblock wout_mem_83338;
    
    wout_mem_83338.references = NULL;
    
    struct memblock wkey_mem_83337;
    
    wkey_mem_83337.references = NULL;
    
    struct memblock wdown_mem_83336;
    
    wdown_mem_83336.references = NULL;
    
    struct memblock wvoc_mem_83335;
    
    wvoc_mem_83335.references = NULL;
    
    struct memblock wval_mem_83334;
    
    wval_mem_83334.references = NULL;
    
    struct memblock wup_mem_83333;
    
    wup_mem_83333.references = NULL;
    
    struct memblock wte_mem_83332;
    
    wte_mem_83332.references = NULL;
    
    struct memblock wqry_mem_83331;
    
    wqry_mem_83331.references = NULL;
    
    struct memblock wpe_mem_83330;
    
    wpe_mem_83330.references = NULL;
    
    struct memblock wout_mem_83329;
    
    wout_mem_83329.references = NULL;
    
    struct memblock wkey_mem_83328;
    
    wkey_mem_83328.references = NULL;
    
    struct memblock wdown_mem_83327;
    
    wdown_mem_83327.references = NULL;
    num_steps_62012 = in0;
    wdown_mem_83327 = in1->v0->mem;
    wkey_mem_83328 = in1->v1->mem;
    wout_mem_83329 = in1->v2->mem;
    wpe_mem_83330 = in1->v3->mem;
    wqry_mem_83331 = in1->v4->mem;
    wte_mem_83332 = in1->v5->mem;
    wup_mem_83333 = in1->v6->mem;
    wval_mem_83334 = in1->v7->mem;
    wvoc_mem_83335 = in1->v8->mem;
    wdown_mem_83336 = in2->v0->mem;
    wkey_mem_83337 = in2->v1->mem;
    wout_mem_83338 = in2->v2->mem;
    wpe_mem_83339 = in2->v3->mem;
    wqry_mem_83340 = in2->v4->mem;
    wte_mem_83341 = in2->v5->mem;
    wup_mem_83342 = in2->v6->mem;
    wval_mem_83343 = in2->v7->mem;
    wvoc_mem_83344 = in2->v8->mem;
    wdown_mem_83345 = in3->v0->mem;
    wkey_mem_83346 = in3->v1->mem;
    wout_mem_83347 = in3->v2->mem;
    wpe_mem_83348 = in3->v3->mem;
    wqry_mem_83349 = in3->v4->mem;
    wte_mem_83350 = in3->v5->mem;
    wup_mem_83351 = in3->v6->mem;
    wval_mem_83352 = in3->v7->mem;
    wvoc_mem_83353 = in3->v8->mem;
    masks_mem_83354 = in4->mem;
    dls_mem_83355 = in5->mem;
    seqs_mem_83356 = in6->mem;
    if (!(((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in3->v0->shape[0] && ((int64_t) 64 == in3->v0->shape[1] && ((int64_t) 16 == in3->v1->shape[0] && ((int64_t) 16 == in3->v1->shape[1] && ((int64_t) 16 == in3->v2->shape[0] && ((int64_t) 16 == in3->v2->shape[1] && ((int64_t) 16 == in3->v3->shape[0] && ((int64_t) 16 == in3->v3->shape[1] && ((int64_t) 16 == in3->v4->shape[0] && ((int64_t) 16 == in3->v4->shape[1] && ((int64_t) 27 == in3->v5->shape[0] && ((int64_t) 16 == in3->v5->shape[1] && ((int64_t) 64 == in3->v6->shape[0] && ((int64_t) 16 == in3->v6->shape[1] && ((int64_t) 16 == in3->v7->shape[0] && ((int64_t) 16 == in3->v7->shape[1] && ((int64_t) 27 == in3->v8->shape[0] && (int64_t) 16 == in3->v8->shape[1]))))))))))))))))) && ((num_steps_62012 == in4->shape[0] && ((int64_t) 16 == in4->shape[1] && (int64_t) 16 == in4->shape[2])) && (num_steps_62012 == in5->shape[0] && (num_steps_62012 == in6->shape[0] && (int64_t) 16 == in6->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_85126, &mem_out_85127, &mem_out_85128, &mem_out_85129, &mem_out_85130, &mem_out_85131, &mem_out_85132, &mem_out_85133, &mem_out_85134, &mem_out_85135, &mem_out_85136, &mem_out_85137, &mem_out_85138, &mem_out_85139, &mem_out_85140, &mem_out_85141, &mem_out_85142, &mem_out_85143, &mem_out_85144, &mem_out_85145, &mem_out_85146, &mem_out_85147, &mem_out_85148, &mem_out_85149, &mem_out_85150, &mem_out_85151, &mem_out_85152, wdown_mem_83327, wkey_mem_83328, wout_mem_83329, wpe_mem_83330, wqry_mem_83331, wte_mem_83332, wup_mem_83333, wval_mem_83334, wvoc_mem_83335, wdown_mem_83336, wkey_mem_83337, wout_mem_83338, wpe_mem_83339, wqry_mem_83340, wte_mem_83341, wup_mem_83342, wval_mem_83343, wvoc_mem_83344, wdown_mem_83345, wkey_mem_83346, wout_mem_83347, wpe_mem_83348, wqry_mem_83349, wte_mem_83350, wup_mem_83351, wval_mem_83352, wvoc_mem_83353, masks_mem_83354, dls_mem_83355, seqs_mem_83356, num_steps_62012);
        if (ret == 0) {
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            struct memblock mem_83321 = ctx->constants->mem_83321;
            struct memblock mem_83322 = ctx->constants->mem_83322;
            struct memblock mem_83323 = ctx->constants->mem_83323;
            struct memblock mem_83324 = ctx->constants->mem_83324;
            struct memblock mem_83325 = ctx->constants->mem_83325;
            struct memblock mem_83326 = ctx->constants->mem_83326;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85126;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85127;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85128;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85129;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85130;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85131;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85132;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85133;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85134;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_85135;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_85136;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_85137;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_85138;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_85139;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_85140;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_85141;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_85142;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_85143;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_85144;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_85145;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_85146;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_85147;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_85148;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_85149;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_85150;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_85151;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_85152;
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
    
    struct memblock mem_out_85134;
    
    mem_out_85134.references = NULL;
    
    struct memblock mem_out_85133;
    
    mem_out_85133.references = NULL;
    
    struct memblock mem_out_85132;
    
    mem_out_85132.references = NULL;
    
    struct memblock mem_out_85131;
    
    mem_out_85131.references = NULL;
    
    struct memblock mem_out_85130;
    
    mem_out_85130.references = NULL;
    
    struct memblock mem_out_85129;
    
    mem_out_85129.references = NULL;
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_85126, &mem_out_85127, &mem_out_85128, &mem_out_85129, &mem_out_85130, &mem_out_85131, &mem_out_85132, &mem_out_85133, &mem_out_85134);
        if (ret == 0) {
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            struct memblock mem_83321 = ctx->constants->mem_83321;
            struct memblock mem_83322 = ctx->constants->mem_83322;
            struct memblock mem_83323 = ctx->constants->mem_83323;
            struct memblock mem_83324 = ctx->constants->mem_83324;
            struct memblock mem_83325 = ctx->constants->mem_83325;
            struct memblock mem_83326 = ctx->constants->mem_83326;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85126;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85127;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85128;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85129;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85130;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85131;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85132;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85133;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85134;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
