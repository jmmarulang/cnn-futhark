
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
    struct memblock mem_99044;
    struct memblock mem_99045;
    struct memblock mem_99046;
    struct memblock mem_99047;
    struct memblock mem_99048;
    struct memblock mem_99049;
    struct memblock mem_99050;
    struct memblock mem_99051;
    struct memblock mem_99052;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10560(struct futhark_context *ctx, struct memblock *mem_out_p_101371, struct memblock *mem_out_p_101372, struct memblock *mem_out_p_101373, struct memblock w_mem_99053, struct memblock mw_mem_99054, struct memblock vw_mem_99055, struct memblock dw_mem_99056, int64_t n_69613, int64_t m_69614, int64_t step_69619, double lt_r_69620);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10561(struct futhark_context *ctx, struct memblock *mem_out_p_101376, struct memblock *mem_out_p_101377, struct memblock *mem_out_p_101378, struct memblock w_mem_99053, struct memblock mw_mem_99054, struct memblock vw_mem_99055, struct memblock dw_mem_99056, int64_t n_70646, int64_t m_70647, int64_t step_70652, double lt_r_70653);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_101381, struct memblock wdown_mem_99053, struct memblock wkey_mem_99054, struct memblock wout_mem_99055, struct memblock wpe_mem_99056, struct memblock wqry_mem_99057, struct memblock wte_mem_99058, struct memblock wup_mem_99059, struct memblock wval_mem_99060, struct memblock wvoc_mem_99061, struct memblock tokens_mem_99062, struct memblock mask_mem_99063);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_101435, struct memblock *mem_out_p_101436, struct memblock *mem_out_p_101437, struct memblock *mem_out_p_101438, struct memblock *mem_out_p_101439, struct memblock *mem_out_p_101440, struct memblock *mem_out_p_101441, struct memblock *mem_out_p_101442, struct memblock *mem_out_p_101443, struct memblock wte_mem_99053, struct memblock wpe_mem_99054, struct memblock wqry_mem_99055, struct memblock wkey_mem_99056, struct memblock wval_mem_99057, struct memblock wout_mem_99058, struct memblock wup_mem_99059, struct memblock wdown_mem_99060, struct memblock wvoc_mem_99061);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_101444, struct memblock *mem_out_p_101445, struct memblock *mem_out_p_101446, struct memblock *mem_out_p_101447, struct memblock *mem_out_p_101448, struct memblock *mem_out_p_101449, struct memblock *mem_out_p_101450, struct memblock *mem_out_p_101451, struct memblock *mem_out_p_101452, struct memblock *mem_out_p_101453, struct memblock *mem_out_p_101454, struct memblock *mem_out_p_101455, struct memblock *mem_out_p_101456, struct memblock *mem_out_p_101457, struct memblock *mem_out_p_101458, struct memblock *mem_out_p_101459, struct memblock *mem_out_p_101460, struct memblock *mem_out_p_101461, struct memblock *mem_out_p_101462, struct memblock *mem_out_p_101463, struct memblock *mem_out_p_101464, struct memblock *mem_out_p_101465, struct memblock *mem_out_p_101466, struct memblock *mem_out_p_101467, struct memblock *mem_out_p_101468, struct memblock *mem_out_p_101469, struct memblock *mem_out_p_101470, struct memblock wdown_mem_99053, struct memblock wkey_mem_99054, struct memblock wout_mem_99055, struct memblock wpe_mem_99056, struct memblock wqry_mem_99057, struct memblock wte_mem_99058, struct memblock wup_mem_99059, struct memblock wval_mem_99060, struct memblock wvoc_mem_99061, struct memblock wdown_mem_99062, struct memblock wkey_mem_99063, struct memblock wout_mem_99064, struct memblock wpe_mem_99065, struct memblock wqry_mem_99066, struct memblock wte_mem_99067, struct memblock wup_mem_99068, struct memblock wval_mem_99069, struct memblock wvoc_mem_99070, struct memblock wdown_mem_99071, struct memblock wkey_mem_99072, struct memblock wout_mem_99073, struct memblock wpe_mem_99074, struct memblock wqry_mem_99075, struct memblock wte_mem_99076, struct memblock wup_mem_99077, struct memblock wval_mem_99078, struct memblock wvoc_mem_99079, struct memblock masks_mem_99080, struct memblock dls_mem_99081, struct memblock seqs_mem_99082);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_101659, struct memblock *mem_out_p_101660, struct memblock *mem_out_p_101661, struct memblock *mem_out_p_101662, struct memblock *mem_out_p_101663, struct memblock *mem_out_p_101664, struct memblock *mem_out_p_101665, struct memblock *mem_out_p_101666, struct memblock *mem_out_p_101667);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_99044 (ctx->constants->mem_99044)
    #define mem_99045 (ctx->constants->mem_99045)
    #define mem_99046 (ctx->constants->mem_99046)
    #define mem_99047 (ctx->constants->mem_99047)
    #define mem_99048 (ctx->constants->mem_99048)
    #define mem_99049 (ctx->constants->mem_99049)
    #define mem_99050 (ctx->constants->mem_99050)
    #define mem_99051 (ctx->constants->mem_99051)
    #define mem_99052 (ctx->constants->mem_99052)
    mem_99044.references = NULL;
    mem_99045.references = NULL;
    mem_99046.references = NULL;
    mem_99047.references = NULL;
    mem_99048.references = NULL;
    mem_99049.references = NULL;
    mem_99050.references = NULL;
    mem_99051.references = NULL;
    mem_99052.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99044, (int64_t) 3456, "mem_99044")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101353 = 0; nest_i_101353 < (int64_t) 27; nest_i_101353++) {
        for (int64_t nest_i_101354 = 0; nest_i_101354 < (int64_t) 16; nest_i_101354++) {
            ((double *) mem_99044.mem)[nest_i_101353 * (int64_t) 16 + nest_i_101354] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99045, (int64_t) 2048, "mem_99045")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101355 = 0; nest_i_101355 < (int64_t) 16; nest_i_101355++) {
        for (int64_t nest_i_101356 = 0; nest_i_101356 < (int64_t) 16; nest_i_101356++) {
            ((double *) mem_99045.mem)[nest_i_101355 * (int64_t) 16 + nest_i_101356] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99046, (int64_t) 2048, "mem_99046")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101357 = 0; nest_i_101357 < (int64_t) 16; nest_i_101357++) {
        for (int64_t nest_i_101358 = 0; nest_i_101358 < (int64_t) 16; nest_i_101358++) {
            ((double *) mem_99046.mem)[nest_i_101357 * (int64_t) 16 + nest_i_101358] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99047, (int64_t) 2048, "mem_99047")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101359 = 0; nest_i_101359 < (int64_t) 16; nest_i_101359++) {
        for (int64_t nest_i_101360 = 0; nest_i_101360 < (int64_t) 16; nest_i_101360++) {
            ((double *) mem_99047.mem)[nest_i_101359 * (int64_t) 16 + nest_i_101360] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99048, (int64_t) 2048, "mem_99048")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101361 = 0; nest_i_101361 < (int64_t) 16; nest_i_101361++) {
        for (int64_t nest_i_101362 = 0; nest_i_101362 < (int64_t) 16; nest_i_101362++) {
            ((double *) mem_99048.mem)[nest_i_101361 * (int64_t) 16 + nest_i_101362] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99049, (int64_t) 2048, "mem_99049")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101363 = 0; nest_i_101363 < (int64_t) 16; nest_i_101363++) {
        for (int64_t nest_i_101364 = 0; nest_i_101364 < (int64_t) 16; nest_i_101364++) {
            ((double *) mem_99049.mem)[nest_i_101363 * (int64_t) 16 + nest_i_101364] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99050, (int64_t) 8192, "mem_99050")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101365 = 0; nest_i_101365 < (int64_t) 64; nest_i_101365++) {
        for (int64_t nest_i_101366 = 0; nest_i_101366 < (int64_t) 16; nest_i_101366++) {
            ((double *) mem_99050.mem)[nest_i_101365 * (int64_t) 16 + nest_i_101366] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99051, (int64_t) 8192, "mem_99051")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101367 = 0; nest_i_101367 < (int64_t) 16; nest_i_101367++) {
        for (int64_t nest_i_101368 = 0; nest_i_101368 < (int64_t) 64; nest_i_101368++) {
            ((double *) mem_99051.mem)[nest_i_101367 * (int64_t) 64 + nest_i_101368] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99052, (int64_t) 3456, "mem_99052")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_101369 = 0; nest_i_101369 < (int64_t) 27; nest_i_101369++) {
        for (int64_t nest_i_101370 = 0; nest_i_101370 < (int64_t) 16; nest_i_101370++) {
            ((double *) mem_99052.mem)[nest_i_101369 * (int64_t) 16 + nest_i_101370] = 0.0;
        }
    }
    #undef mem_99044
    #undef mem_99045
    #undef mem_99046
    #undef mem_99047
    #undef mem_99048
    #undef mem_99049
    #undef mem_99050
    #undef mem_99051
    #undef mem_99052
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_99044, "ctx->constants->mem_99044") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99045, "ctx->constants->mem_99045") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99046, "ctx->constants->mem_99046") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99047, "ctx->constants->mem_99047") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99048, "ctx->constants->mem_99048") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99049, "ctx->constants->mem_99049") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99050, "ctx->constants->mem_99050") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99051, "ctx->constants->mem_99051") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_99052, "ctx->constants->mem_99052") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10560(struct futhark_context *ctx, struct memblock *mem_out_p_101371, struct memblock *mem_out_p_101372, struct memblock *mem_out_p_101373, struct memblock w_mem_99053, struct memblock mw_mem_99054, struct memblock vw_mem_99055, struct memblock dw_mem_99056, int64_t n_69613, int64_t m_69614, int64_t step_69619, double lt_r_69620)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_99097_cached_sizze_101374 = 0;
    unsigned char *mem_99097 = NULL;
    int64_t mem_99100_cached_sizze_101375 = 0;
    unsigned char *mem_99100 = NULL;
    struct memblock mem_99135;
    
    mem_99135.references = NULL;
    
    struct memblock mem_99062;
    
    mem_99062.references = NULL;
    
    struct memblock mem_99059;
    
    mem_99059.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_99057 = (int64_t) 8 * n_69613;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_99058 = m_69614 * binop_x_99057;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99059, bytes_99058, "mem_99059")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99062, bytes_99058, "mem_99062")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98061 = 0; i_98061 < n_69613; i_98061++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98054 = 0; i_98054 < m_69614; i_98054++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93813 = ((double *) mw_mem_99054.mem)[i_98061 * m_69614 + i_98054];
            
            // futhark/microgpt.fut:396:10-20
            
            double zp_lhs_93814 = 0.85 * zt_rhs_93813;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93815 = ((double *) dw_mem_99056.mem)[i_98061 * m_69614 + i_98054];
            
            // futhark/microgpt.fut:396:35-45
            
            double zp_rhs_93816 = 0.15000000000000002 * zt_rhs_93815;
            
            // futhark/microgpt.fut:396:21-45
            
            double lifted_lambda_res_93817 = zp_lhs_93814 + zp_rhs_93816;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93824 = ((double *) vw_mem_99055.mem)[i_98061 * m_69614 + i_98054];
            
            // futhark/microgpt.fut:398:10-20
            
            double zp_lhs_93825 = 0.99 * zt_rhs_93824;
            
            // futhark/microgpt.fut:398:35-45
            
            double zt_lhs_93827 = 1.0000000000000009e-2 * zt_rhs_93815;
            
            // futhark/microgpt.fut:398:46-56
            
            double zp_rhs_93828 = zt_rhs_93815 * zt_lhs_93827;
            
            // futhark/microgpt.fut:398:21-56
            
            double lifted_lambda_res_93829 = zp_lhs_93825 + zp_rhs_93828;
            
            ((double *) mem_99059.mem)[i_98061 * m_69614 + i_98054] = lifted_lambda_res_93829;
            ((double *) mem_99062.mem)[i_98061 * m_69614 + i_98054] = lifted_lambda_res_93817;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_74823 = sitofp_i64_f64(step_69619);
    
    // futhark/microgpt.fut:400:54-57
    
    double ztzt_rhs_74824 = 1.0 + i64_res_74823;
    
    // futhark/microgpt.fut:400:30-57
    
    double zm_rhs_74825 = fpow64(0.85, ztzt_rhs_74824);
    
    // futhark/microgpt.fut:400:23-57
    
    double zs_rhs_74826 = 1.0 - zm_rhs_74825;
    
    // futhark/microgpt.fut:402:31-58
    
    double zm_rhs_74864 = fpow64(0.99, ztzt_rhs_74824);
    
    // futhark/microgpt.fut:402:23-58
    
    double zs_rhs_74865 = 1.0 - zm_rhs_74864;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_99097_cached_sizze_101374 < bytes_99058) {
        err = lexical_realloc(ctx, &mem_99097, &mem_99097_cached_sizze_101374, bytes_99058);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99100_cached_sizze_101375 < bytes_99058) {
        err = lexical_realloc(ctx, &mem_99100, &mem_99100_cached_sizze_101375, bytes_99058);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98075 = 0; i_98075 < n_69613; i_98075++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98068 = 0; i_98068 < m_69614; i_98068++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_93849 = ((double *) mem_99062.mem)[i_98075 * m_69614 + i_98068];
            
            // futhark/microgpt.fut:400:18-57
            
            double lifted_lambda_res_93850 = zs_lhs_93849 / zs_rhs_74826;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_93857 = ((double *) mem_99059.mem)[i_98075 * m_69614 + i_98068];
            
            // futhark/microgpt.fut:402:18-58
            
            double lifted_lambda_res_93858 = zs_lhs_93857 / zs_rhs_74865;
            
            ((double *) mem_99097)[i_98075 * m_69614 + i_98068] = lifted_lambda_res_93858;
            ((double *) mem_99100)[i_98075 * m_69614 + i_98068] = lifted_lambda_res_93850;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99135, bytes_99058, "mem_99135")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98084 = 0; i_98084 < n_69613; i_98084++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98080 = 0; i_98080 < m_69614; i_98080++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_73878 = ((double *) w_mem_99053.mem)[i_98084 * m_69614 + i_98080];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_73879 = ((double *) mem_99100)[i_98084 * m_69614 + i_98080];
            
            // futhark/microgpt.fut:404:21-34
            
            double zs_lhs_73880 = lt_r_69620 * zt_rhs_73879;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_73881 = ((double *) mem_99097)[i_98084 * m_69614 + i_98080];
            
            // futhark/microgpt.fut:404:51-57
            
            double zp_lhs_73882 = fpow64(ztzt_lhs_73881, 0.5);
            
            // futhark/microgpt.fut:404:59-71
            
            double zs_rhs_73883 = 1.0e-8 + zp_lhs_73882;
            
            // futhark/microgpt.fut:404:35-71
            
            double zm_rhs_73884 = zs_lhs_73880 / zs_rhs_73883;
            
            // futhark/microgpt.fut:404:13-71
            
            double lifted_lambda_res_73885 = zm_lhs_73878 - zm_rhs_73884;
            
            ((double *) mem_99135.mem)[i_98084 * m_69614 + i_98080] = lifted_lambda_res_73885;
        }
    }
    if (memblock_set(ctx, &mem_out_101017, &mem_99135, "mem_99135") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101018, &mem_99062, "mem_99062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101019, &mem_99059, "mem_99059") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101371, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101372, &mem_out_101018, "mem_out_101018") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101373, &mem_out_101019, "mem_out_101019") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_99097);
        free(mem_99100);
        if (memblock_unref(ctx, &mem_99135, "mem_99135") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_99062, "mem_99062") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_99059, "mem_99059") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101019, "mem_out_101019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101018, "mem_out_101018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10561(struct futhark_context *ctx, struct memblock *mem_out_p_101376, struct memblock *mem_out_p_101377, struct memblock *mem_out_p_101378, struct memblock w_mem_99053, struct memblock mw_mem_99054, struct memblock vw_mem_99055, struct memblock dw_mem_99056, int64_t n_70646, int64_t m_70647, int64_t step_70652, double lt_r_70653)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_99097_cached_sizze_101379 = 0;
    unsigned char *mem_99097 = NULL;
    int64_t mem_99100_cached_sizze_101380 = 0;
    unsigned char *mem_99100 = NULL;
    struct memblock mem_99135;
    
    mem_99135.references = NULL;
    
    struct memblock mem_99062;
    
    mem_99062.references = NULL;
    
    struct memblock mem_99059;
    
    mem_99059.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_99057 = (int64_t) 8 * n_70646;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_99058 = m_70647 * binop_x_99057;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99059, bytes_99058, "mem_99059")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99062, bytes_99058, "mem_99062")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98061 = 0; i_98061 < n_70646; i_98061++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98054 = 0; i_98054 < m_70647; i_98054++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93813 = ((double *) mw_mem_99054.mem)[i_98061 * m_70647 + i_98054];
            
            // futhark/microgpt.fut:396:10-20
            
            double zp_lhs_93814 = 0.85 * zt_rhs_93813;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93815 = ((double *) dw_mem_99056.mem)[i_98061 * m_70647 + i_98054];
            
            // futhark/microgpt.fut:396:35-45
            
            double zp_rhs_93816 = 0.15000000000000002 * zt_rhs_93815;
            
            // futhark/microgpt.fut:396:21-45
            
            double lifted_lambda_res_93817 = zp_lhs_93814 + zp_rhs_93816;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_93824 = ((double *) vw_mem_99055.mem)[i_98061 * m_70647 + i_98054];
            
            // futhark/microgpt.fut:398:10-20
            
            double zp_lhs_93825 = 0.99 * zt_rhs_93824;
            
            // futhark/microgpt.fut:398:35-45
            
            double zt_lhs_93827 = 1.0000000000000009e-2 * zt_rhs_93815;
            
            // futhark/microgpt.fut:398:46-56
            
            double zp_rhs_93828 = zt_rhs_93815 * zt_lhs_93827;
            
            // futhark/microgpt.fut:398:21-56
            
            double lifted_lambda_res_93829 = zp_lhs_93825 + zp_rhs_93828;
            
            ((double *) mem_99059.mem)[i_98061 * m_70647 + i_98054] = lifted_lambda_res_93829;
            ((double *) mem_99062.mem)[i_98061 * m_70647 + i_98054] = lifted_lambda_res_93817;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_74823 = sitofp_i64_f64(step_70652);
    
    // futhark/microgpt.fut:400:54-57
    
    double ztzt_rhs_74824 = 1.0 + i64_res_74823;
    
    // futhark/microgpt.fut:400:30-57
    
    double zm_rhs_74825 = fpow64(0.85, ztzt_rhs_74824);
    
    // futhark/microgpt.fut:400:23-57
    
    double zs_rhs_74826 = 1.0 - zm_rhs_74825;
    
    // futhark/microgpt.fut:402:31-58
    
    double zm_rhs_74864 = fpow64(0.99, ztzt_rhs_74824);
    
    // futhark/microgpt.fut:402:23-58
    
    double zs_rhs_74865 = 1.0 - zm_rhs_74864;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_99097_cached_sizze_101379 < bytes_99058) {
        err = lexical_realloc(ctx, &mem_99097, &mem_99097_cached_sizze_101379, bytes_99058);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99100_cached_sizze_101380 < bytes_99058) {
        err = lexical_realloc(ctx, &mem_99100, &mem_99100_cached_sizze_101380, bytes_99058);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98075 = 0; i_98075 < n_70646; i_98075++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98068 = 0; i_98068 < m_70647; i_98068++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_93849 = ((double *) mem_99062.mem)[i_98075 * m_70647 + i_98068];
            
            // futhark/microgpt.fut:400:18-57
            
            double lifted_lambda_res_93850 = zs_lhs_93849 / zs_rhs_74826;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_93857 = ((double *) mem_99059.mem)[i_98075 * m_70647 + i_98068];
            
            // futhark/microgpt.fut:402:18-58
            
            double lifted_lambda_res_93858 = zs_lhs_93857 / zs_rhs_74865;
            
            ((double *) mem_99097)[i_98075 * m_70647 + i_98068] = lifted_lambda_res_93858;
            ((double *) mem_99100)[i_98075 * m_70647 + i_98068] = lifted_lambda_res_93850;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99135, bytes_99058, "mem_99135")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98084 = 0; i_98084 < n_70646; i_98084++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98080 = 0; i_98080 < m_70647; i_98080++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_73878 = ((double *) w_mem_99053.mem)[i_98084 * m_70647 + i_98080];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_73879 = ((double *) mem_99100)[i_98084 * m_70647 + i_98080];
            
            // futhark/microgpt.fut:404:21-34
            
            double zs_lhs_73880 = lt_r_70653 * zt_rhs_73879;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_73881 = ((double *) mem_99097)[i_98084 * m_70647 + i_98080];
            
            // futhark/microgpt.fut:404:51-57
            
            double zp_lhs_73882 = fpow64(ztzt_lhs_73881, 0.5);
            
            // futhark/microgpt.fut:404:59-71
            
            double zs_rhs_73883 = 1.0e-8 + zp_lhs_73882;
            
            // futhark/microgpt.fut:404:35-71
            
            double zm_rhs_73884 = zs_lhs_73880 / zs_rhs_73883;
            
            // futhark/microgpt.fut:404:13-71
            
            double lifted_lambda_res_73885 = zm_lhs_73878 - zm_rhs_73884;
            
            ((double *) mem_99135.mem)[i_98084 * m_70647 + i_98080] = lifted_lambda_res_73885;
        }
    }
    if (memblock_set(ctx, &mem_out_101017, &mem_99135, "mem_99135") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101018, &mem_99062, "mem_99062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101019, &mem_99059, "mem_99059") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101376, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101377, &mem_out_101018, "mem_out_101018") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101378, &mem_out_101019, "mem_out_101019") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_99097);
        free(mem_99100);
        if (memblock_unref(ctx, &mem_99135, "mem_99135") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_99062, "mem_99062") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_99059, "mem_99059") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101019, "mem_out_101019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101018, "mem_out_101018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_101381, struct memblock wdown_mem_99053, struct memblock wkey_mem_99054, struct memblock wout_mem_99055, struct memblock wpe_mem_99056, struct memblock wqry_mem_99057, struct memblock wte_mem_99058, struct memblock wup_mem_99059, struct memblock wval_mem_99060, struct memblock wvoc_mem_99061, struct memblock tokens_mem_99062, struct memblock mask_mem_99063)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_99064_cached_sizze_101382 = 0;
    unsigned char *mem_99064 = NULL;
    int64_t mem_99069_cached_sizze_101383 = 0;
    unsigned char *mem_99069 = NULL;
    int64_t mem_99080_cached_sizze_101384 = 0;
    unsigned char *mem_99080 = NULL;
    int64_t mem_99085_cached_sizze_101385 = 0;
    unsigned char *mem_99085 = NULL;
    int64_t mem_99096_cached_sizze_101386 = 0;
    unsigned char *mem_99096 = NULL;
    int64_t mem_99101_cached_sizze_101387 = 0;
    unsigned char *mem_99101 = NULL;
    int64_t mem_99108_cached_sizze_101388 = 0;
    unsigned char *mem_99108 = NULL;
    int64_t mem_99119_cached_sizze_101389 = 0;
    unsigned char *mem_99119 = NULL;
    int64_t mem_99124_cached_sizze_101390 = 0;
    unsigned char *mem_99124 = NULL;
    int64_t mem_99131_cached_sizze_101391 = 0;
    unsigned char *mem_99131 = NULL;
    int64_t mem_99142_cached_sizze_101392 = 0;
    unsigned char *mem_99142 = NULL;
    int64_t mem_99143_cached_sizze_101393 = 0;
    unsigned char *mem_99143 = NULL;
    int64_t mem_99144_cached_sizze_101394 = 0;
    unsigned char *mem_99144 = NULL;
    int64_t mem_99157_cached_sizze_101395 = 0;
    unsigned char *mem_99157 = NULL;
    int64_t mem_99158_cached_sizze_101396 = 0;
    unsigned char *mem_99158 = NULL;
    int64_t mem_99159_cached_sizze_101397 = 0;
    unsigned char *mem_99159 = NULL;
    int64_t mem_99190_cached_sizze_101398 = 0;
    unsigned char *mem_99190 = NULL;
    int64_t mem_99191_cached_sizze_101399 = 0;
    unsigned char *mem_99191 = NULL;
    int64_t mem_99192_cached_sizze_101400 = 0;
    unsigned char *mem_99192 = NULL;
    int64_t mem_99208_cached_sizze_101401 = 0;
    unsigned char *mem_99208 = NULL;
    int64_t mem_99209_cached_sizze_101402 = 0;
    unsigned char *mem_99209 = NULL;
    int64_t mem_99210_cached_sizze_101403 = 0;
    unsigned char *mem_99210 = NULL;
    int64_t mem_99223_cached_sizze_101404 = 0;
    unsigned char *mem_99223 = NULL;
    int64_t mem_99224_cached_sizze_101405 = 0;
    unsigned char *mem_99224 = NULL;
    int64_t mem_99225_cached_sizze_101406 = 0;
    unsigned char *mem_99225 = NULL;
    int64_t mem_99271_cached_sizze_101407 = 0;
    unsigned char *mem_99271 = NULL;
    int64_t mem_99277_cached_sizze_101408 = 0;
    unsigned char *mem_99277 = NULL;
    int64_t mem_99282_cached_sizze_101409 = 0;
    unsigned char *mem_99282 = NULL;
    int64_t mem_99293_cached_sizze_101410 = 0;
    unsigned char *mem_99293 = NULL;
    int64_t mem_99298_cached_sizze_101411 = 0;
    unsigned char *mem_99298 = NULL;
    int64_t mem_99309_cached_sizze_101412 = 0;
    unsigned char *mem_99309 = NULL;
    int64_t mem_99314_cached_sizze_101413 = 0;
    unsigned char *mem_99314 = NULL;
    int64_t mem_99321_cached_sizze_101414 = 0;
    unsigned char *mem_99321 = NULL;
    int64_t mem_99332_cached_sizze_101415 = 0;
    unsigned char *mem_99332 = NULL;
    int64_t mem_99337_cached_sizze_101416 = 0;
    unsigned char *mem_99337 = NULL;
    int64_t mem_99353_cached_sizze_101417 = 0;
    unsigned char *mem_99353 = NULL;
    int64_t mem_99358_cached_sizze_101418 = 0;
    unsigned char *mem_99358 = NULL;
    int64_t mem_99369_cached_sizze_101419 = 0;
    unsigned char *mem_99369 = NULL;
    int64_t mem_99374_cached_sizze_101420 = 0;
    unsigned char *mem_99374 = NULL;
    int64_t mem_99385_cached_sizze_101421 = 0;
    unsigned char *mem_99385 = NULL;
    int64_t mem_99390_cached_sizze_101422 = 0;
    unsigned char *mem_99390 = NULL;
    int64_t mem_99401_cached_sizze_101423 = 0;
    unsigned char *mem_99401 = NULL;
    int64_t mem_99406_cached_sizze_101424 = 0;
    unsigned char *mem_99406 = NULL;
    int64_t mem_99413_cached_sizze_101425 = 0;
    unsigned char *mem_99413 = NULL;
    int64_t mem_99424_cached_sizze_101426 = 0;
    unsigned char *mem_99424 = NULL;
    int64_t mem_99429_cached_sizze_101427 = 0;
    unsigned char *mem_99429 = NULL;
    int64_t mem_99440_cached_sizze_101428 = 0;
    unsigned char *mem_99440 = NULL;
    int64_t mem_99445_cached_sizze_101429 = 0;
    unsigned char *mem_99445 = NULL;
    int64_t mem_99456_cached_sizze_101430 = 0;
    unsigned char *mem_99456 = NULL;
    int64_t mem_99461_cached_sizze_101431 = 0;
    unsigned char *mem_99461 = NULL;
    int64_t mem_99472_cached_sizze_101432 = 0;
    unsigned char *mem_99472 = NULL;
    int64_t mem_99477_cached_sizze_101433 = 0;
    unsigned char *mem_99477 = NULL;
    int64_t mem_99493_cached_sizze_101434 = 0;
    unsigned char *mem_99493 = NULL;
    struct memblock mem_99488;
    
    mem_99488.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_99064_cached_sizze_101382 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99064, &mem_99064_cached_sizze_101382, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99069_cached_sizze_101383 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99069, &mem_99069_cached_sizze_101383, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98056 = 0; i_98056 < (int64_t) 16; i_98056++) {
        // futhark/microgpt.fut:381:41-50
        
        int64_t tmp_93154 = ((int64_t *) tokens_mem_99062.mem)[i_98056];
        
        // futhark/microgpt.fut:381:37-51
        
        bool x_93155 = sle64((int64_t) 0, tmp_93154);
        
        // futhark/microgpt.fut:381:37-51
        
        bool y_93156 = slt64(tmp_93154, (int64_t) 27);
        
        // futhark/microgpt.fut:381:37-51
        
        bool bounds_check_93157 = x_93155 && y_93156;
        
        // futhark/microgpt.fut:381:37-51
        
        bool index_certs_93158;
        
        if (!bounds_check_93157) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93154, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:381:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:381:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98052 = 0; i_98052 < (int64_t) 16; i_98052++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_93165 = ((double *) wte_mem_99058.mem)[tmp_93154 * (int64_t) 16 + i_98052];
            
            ((double *) mem_99069)[i_98052] = lifted_lambda_res_93165;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99064, i_98056 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99069, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99080_cached_sizze_101384 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99080, &mem_99080_cached_sizze_101384, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99085_cached_sizze_101385 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99085, &mem_99085_cached_sizze_101385, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98064 = 0; i_98064 < (int64_t) 16; i_98064++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98060 = 0; i_98060 < (int64_t) 16; i_98060++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_93197 = ((double *) wpe_mem_99056.mem)[i_98064 * (int64_t) 16 + i_98060];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_93198 = ((double *) mem_99064)[i_98064 * (int64_t) 16 + i_98060];
            
            // futhark/microgpt.fut:158:46-86
            
            double zp_res_93199 = zp_lhs_93197 + zp_rhs_93198;
            
            ((double *) mem_99085)[i_98060] = zp_res_93199;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99080, i_98064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99085, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99096_cached_sizze_101386 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99096, &mem_99096_cached_sizze_101386, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99101_cached_sizze_101387 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99101, &mem_99101_cached_sizze_101387, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99108_cached_sizze_101388 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99108, &mem_99108_cached_sizze_101388, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98076 = 0; i_98076 < (int64_t) 16; i_98076++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98068 = 0; i_98068 < (int64_t) 16; i_98068++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93214 = ((double *) mem_99080)[i_98076 * (int64_t) 16 + i_98068];
            
            // futhark/microgpt.fut:159:77-114
            
            double zt_res_93215 = zt_lhs_93214 * zt_lhs_93214;
            
            ((double *) mem_99101)[i_98068] = zt_res_93215;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_93217;
        double r_93219 = 0.0;
        
        for (int64_t i_93218 = 0; i_93218 < (int64_t) 16; i_93218++) {
            // futhark/microgpt.fut:160:37-47
            
            double lifted_lambda_res_93220 = ((double *) mem_99101)[i_93218];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_93221 = r_93219 + lifted_lambda_res_93220;
            double r_tmp_101024 = zp_res_93221;
            
            r_93219 = r_tmp_101024;
        }
        defunc_0_lifted_lambda_res_93217 = r_93219;
        // futhark/microgpt.fut:160:17-64
        
        double zs_res_93222 = defunc_0_lifted_lambda_res_93217 / 16.0;
        
        // futhark/microgpt.fut:161:24-55
        
        double zp_res_93223 = 1.0e-5 + zs_res_93222;
        
        // futhark/microgpt.fut:161:16-55
        
        double sqrt_res_93224 = futrts_sqrt64(zp_res_93223);
        
        // futhark/microgpt.fut:162:27-38
        
        double zs_res_93225 = 1.0 / sqrt_res_93224;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98072 = 0; i_98072 < (int64_t) 16; i_98072++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93232 = ((double *) mem_99080)[i_98076 * (int64_t) 16 + i_98072];
            
            // futhark/microgpt.fut:162:5-38
            
            double zt_res_93233 = zs_res_93225 * zt_lhs_93232;
            
            ((double *) mem_99108)[i_98072] = zt_res_93233;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99096, i_98076 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99108, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99119_cached_sizze_101389 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99119, &mem_99119_cached_sizze_101389, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99124_cached_sizze_101390 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99124, &mem_99124_cached_sizze_101390, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99131_cached_sizze_101391 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99131, &mem_99131_cached_sizze_101391, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98088 = 0; i_98088 < (int64_t) 16; i_98088++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98080 = 0; i_98080 < (int64_t) 16; i_98080++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93248 = ((double *) mem_99096)[i_98088 * (int64_t) 16 + i_98080];
            
            // futhark/microgpt.fut:163:77-114
            
            double zt_res_93249 = zt_lhs_93248 * zt_lhs_93248;
            
            ((double *) mem_99124)[i_98080] = zt_res_93249;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_93251;
        double r_93253 = 0.0;
        
        for (int64_t i_93252 = 0; i_93252 < (int64_t) 16; i_93252++) {
            // futhark/microgpt.fut:164:37-47
            
            double lifted_lambda_res_93254 = ((double *) mem_99124)[i_93252];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_93255 = r_93253 + lifted_lambda_res_93254;
            double r_tmp_101028 = zp_res_93255;
            
            r_93253 = r_tmp_101028;
        }
        defunc_0_lifted_lambda_res_93251 = r_93253;
        // futhark/microgpt.fut:164:17-64
        
        double zs_res_93256 = defunc_0_lifted_lambda_res_93251 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_93257 = 1.0e-5 + zs_res_93256;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_93258 = futrts_sqrt64(zp_res_93257);
        
        // futhark/microgpt.fut:166:27-38
        
        double zs_res_93259 = 1.0 / sqrt_res_93258;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98084 = 0; i_98084 < (int64_t) 16; i_98084++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93266 = ((double *) mem_99096)[i_98088 * (int64_t) 16 + i_98084];
            
            // futhark/microgpt.fut:166:5-38
            
            double zt_res_93267 = zs_res_93259 * zt_lhs_93266;
            
            ((double *) mem_99131)[i_98084] = zt_res_93267;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99119, i_98088 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99131, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99142_cached_sizze_101392 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99142, &mem_99142_cached_sizze_101392, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99143_cached_sizze_101393 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99143, &mem_99143_cached_sizze_101393, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99144_cached_sizze_101394 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99144, &mem_99144_cached_sizze_101394, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99157_cached_sizze_101395 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99157, &mem_99157_cached_sizze_101395, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99158_cached_sizze_101396 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99158, &mem_99158_cached_sizze_101396, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99159_cached_sizze_101397 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99159, &mem_99159_cached_sizze_101397, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98106 = 0; i_98106 < (int64_t) 16; i_98106++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98096 = 0; i_98096 < (int64_t) 16; i_98096++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94032;
            double r_94034 = 0.0;
            
            for (int64_t i_94033 = 0; i_94033 < (int64_t) 16; i_94033++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_94035 = ((double *) wqry_mem_99057.mem)[i_98096 * (int64_t) 16 + i_94033];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_94036 = ((double *) mem_99119)[i_98106 * (int64_t) 16 + i_94033];
                
                // futhark/microgpt.fut:167:66-105
                
                double zt_res_94037 = zt_lhs_94035 * zt_rhs_94036;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94038 = r_94034 + zt_res_94037;
                double r_tmp_101036 = zp_res_94038;
                
                r_94034 = r_tmp_101036;
            }
            defunc_0_lifted_lambda_res_94032 = r_94034;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94045;
            double r_94047 = 0.0;
            
            for (int64_t i_94046 = 0; i_94046 < (int64_t) 16; i_94046++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_94048 = ((double *) wkey_mem_99054.mem)[i_98096 * (int64_t) 16 + i_94046];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_94049 = ((double *) mem_99119)[i_98106 * (int64_t) 16 + i_94046];
                
                // futhark/microgpt.fut:168:66-105
                
                double zt_res_94050 = zt_lhs_94048 * zt_rhs_94049;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94051 = r_94047 + zt_res_94050;
                double r_tmp_101037 = zp_res_94051;
                
                r_94047 = r_tmp_101037;
            }
            defunc_0_lifted_lambda_res_94045 = r_94047;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94061;
            double r_94063 = 0.0;
            
            for (int64_t i_94062 = 0; i_94062 < (int64_t) 16; i_94062++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_94064 = ((double *) wval_mem_99060.mem)[i_98096 * (int64_t) 16 + i_94062];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_94065 = ((double *) mem_99119)[i_98106 * (int64_t) 16 + i_94062];
                
                // futhark/microgpt.fut:169:66-105
                
                double zt_res_94066 = zt_lhs_94064 * zt_rhs_94065;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94067 = r_94063 + zt_res_94066;
                double r_tmp_101038 = zp_res_94067;
                
                r_94063 = r_tmp_101038;
            }
            defunc_0_lifted_lambda_res_94061 = r_94063;
            ((double *) mem_99157)[i_98096] = defunc_0_lifted_lambda_res_94061;
            ((double *) mem_99158)[i_98096] = defunc_0_lifted_lambda_res_94045;
            ((double *) mem_99159)[i_98096] = defunc_0_lifted_lambda_res_94032;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99142, i_98106 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99157, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99143, i_98106 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99158, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99144, i_98106 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99159, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99190_cached_sizze_101398 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99190, &mem_99190_cached_sizze_101398, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99191_cached_sizze_101399 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99191, &mem_99191_cached_sizze_101399, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99192_cached_sizze_101400 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99192, &mem_99192_cached_sizze_101400, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99208_cached_sizze_101401 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99208, &mem_99208_cached_sizze_101401, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99209_cached_sizze_101402 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99209, &mem_99209_cached_sizze_101402, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99210_cached_sizze_101403 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99210, &mem_99210_cached_sizze_101403, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99223_cached_sizze_101404 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99223, &mem_99223_cached_sizze_101404, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99224_cached_sizze_101405 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99224, &mem_99224_cached_sizze_101405, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99225_cached_sizze_101406 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99225, &mem_99225_cached_sizze_101406, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98136 = 0; i_98136 < (int64_t) 4; i_98136++) {
        // futhark/microgpt.fut:170:69-72
        
        int64_t zp_lhs_93907 = mul64((int64_t) 4, i_98136);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98126 = 0; i_98126 < (int64_t) 16; i_98126++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98116 = 0; i_98116 < (int64_t) 4; i_98116++) {
                // futhark/microgpt.fut:170:74-81
                
                int64_t tmp_94225 = add64(zp_lhs_93907, i_98116);
                
                // futhark/microgpt.fut:170:51-83
                
                bool x_94226 = sle64((int64_t) 0, tmp_94225);
                
                // futhark/microgpt.fut:170:51-83
                
                bool y_94227 = slt64(tmp_94225, (int64_t) 16);
                
                // futhark/microgpt.fut:170:51-83
                
                bool bounds_check_94228 = x_94226 && y_94227;
                
                // futhark/microgpt.fut:170:51-83
                
                bool index_certs_94229;
                
                if (!bounds_check_94228) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_94225, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:170:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:170:15-84\n   #9  futhark/microgpt.fut:382:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94230 = ((double *) mem_99144)[i_98126 * (int64_t) 16 + tmp_94225];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94238 = ((double *) mem_99143)[i_98126 * (int64_t) 16 + tmp_94225];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94249 = ((double *) mem_99142)[i_98126 * (int64_t) 16 + tmp_94225];
                
                ((double *) mem_99223)[i_98116] = lifted_lambda_res_94249;
                ((double *) mem_99224)[i_98116] = lifted_lambda_res_94238;
                ((double *) mem_99225)[i_98116] = lifted_lambda_res_94230;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99208, i_98126 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99223, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99209, i_98126 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99224, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99210, i_98126 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99225, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_99190, i_98136 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99208, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_99191, i_98136 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99209, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_99192, i_98136 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99210, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99271_cached_sizze_101407 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99271, &mem_99271_cached_sizze_101407, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99277_cached_sizze_101408 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99277, &mem_99277_cached_sizze_101408, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99282_cached_sizze_101409 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99282, &mem_99282_cached_sizze_101409, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99293_cached_sizze_101410 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99293, &mem_99293_cached_sizze_101410, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99298_cached_sizze_101411 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99298, &mem_99298_cached_sizze_101411, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99309_cached_sizze_101412 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99309, &mem_99309_cached_sizze_101412, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99314_cached_sizze_101413 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99314, &mem_99314_cached_sizze_101413, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99321_cached_sizze_101414 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99321, &mem_99321_cached_sizze_101414, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99332_cached_sizze_101415 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99332, &mem_99332_cached_sizze_101415, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99337_cached_sizze_101416 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99337, &mem_99337_cached_sizze_101416, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98181 = 0; i_98181 < (int64_t) 4; i_98181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98146 = 0; i_98146 < (int64_t) 16; i_98146++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98142 = 0; i_98142 < (int64_t) 16; i_98142++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93412;
                double r_93414 = 0.0;
                
                for (int64_t i_93413 = 0; i_93413 < (int64_t) 4; i_93413++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93415 = ((double *) mem_99192)[i_98181 * (int64_t) 64 + i_98146 * (int64_t) 4 + i_93413];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93416 = ((double *) mem_99191)[i_98181 * (int64_t) 64 + i_98142 * (int64_t) 4 + i_93413];
                    
                    // futhark/microgpt.fut:173:113-164
                    
                    double zt_res_93417 = zt_lhs_93415 * zt_rhs_93416;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93418 = r_93414 + zt_res_93417;
                    double r_tmp_101051 = zp_res_93418;
                    
                    r_93414 = r_tmp_101051;
                }
                defunc_0_lifted_lambda_res_93412 = r_93414;
                ((double *) mem_99282)[i_98142] = defunc_0_lifted_lambda_res_93412;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99277, i_98146 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99282, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98154 = 0; i_98154 < (int64_t) 16; i_98154++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98150 = 0; i_98150 < (int64_t) 16; i_98150++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_93433 = ((double *) mem_99277)[i_98154 * (int64_t) 16 + i_98150];
                
                // futhark/microgpt.fut:174:47-78
                
                double zs_res_93434 = zs_lhs_93433 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_93435 = ((double *) mask_mem_99063.mem)[i_98154 * (int64_t) 16 + i_98150];
                
                // futhark/microgpt.fut:174:65-102
                
                double zp_res_93436 = zs_res_93434 + zp_rhs_93435;
                
                ((double *) mem_99298)[i_98150] = zp_res_93436;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99293, i_98154 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99298, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98169 = 0; i_98169 < (int64_t) 16; i_98169++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_94328;
            int64_t defunc_0_reduce_res_94329;
            double redout_98156;
            int64_t redout_98157;
            
            redout_98156 = -INFINITY;
            redout_98157 = (int64_t) 16;
            for (int64_t i_98158 = 0; i_98158 < (int64_t) 16; i_98158++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94279 = ((double *) mem_99293)[i_98169 * (int64_t) 16 + i_98158];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_93461 = lifted_lambda_res_94279 < redout_98156;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_93462;
                
                if (zg_res_93461) {
                    lifted_lambda_res_93462 = redout_98156;
                } else {
                    lifted_lambda_res_93462 = lifted_lambda_res_94279;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_93463;
                
                if (zg_res_93461) {
                    lifted_lambda_res_93463 = redout_98157;
                } else {
                    lifted_lambda_res_93463 = i_98158;
                }
                
                double redout_tmp_101055 = lifted_lambda_res_93462;
                int64_t redout_tmp_101056 = lifted_lambda_res_93463;
                
                redout_98156 = redout_tmp_101055;
                redout_98157 = redout_tmp_101056;
            }
            defunc_0_reduce_res_94328 = redout_98156;
            defunc_0_reduce_res_94329 = redout_98157;
            // futhark/microgpt.fut:175:56-112
            
            bool x_93464 = sle64((int64_t) 0, defunc_0_reduce_res_94329);
            
            // futhark/microgpt.fut:175:56-112
            
            bool y_93465 = slt64(defunc_0_reduce_res_94329, (int64_t) 16);
            
            // futhark/microgpt.fut:175:56-112
            
            bool bounds_check_93466 = x_93464 && y_93465;
            
            // futhark/microgpt.fut:175:56-112
            
            bool index_certs_93467;
            
            if (!bounds_check_93466) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_94329, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:175:56-112\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:175:16-178:38\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:27-39\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:9:13-40\n   #10 futhark/microgpt.fut:15:29-44\n   #11 futhark/microgpt.fut:4:11-25\n   #12 futhark/microgpt.fut:15:15-45\n   #13 futhark/microgpt.fut:173:15-179:78\n   #14 futhark/microgpt.fut:382:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double x49_93468 = ((double *) mem_99293)[i_98169 * (int64_t) 16 + defunc_0_reduce_res_94329];
            
            // futhark/microgpt.fut:176:67-76
            
            double neg_res_93469 = -x49_93468;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98161 = 0; i_98161 < (int64_t) 16; i_98161++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_93476 = ((double *) mem_99293)[i_98169 * (int64_t) 16 + i_98161];
                
                // futhark/microgpt.fut:176:44-76
                
                double zp_res_93477 = neg_res_93469 + zp_lhs_93476;
                
                // futhark/microgpt.fut:176:37-76
                
                double exp_res_93478 = futrts_exp64(zp_res_93477);
                
                ((double *) mem_99314)[i_98161] = exp_res_93478;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93480;
            double r_93482 = 0.0;
            
            for (int64_t i_93481 = 0; i_93481 < (int64_t) 16; i_93481++) {
                // futhark/microgpt.fut:177:36-46
                
                double lifted_lambda_res_93483 = ((double *) mem_99314)[i_93481];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93484 = r_93482 + lifted_lambda_res_93483;
                double r_tmp_101058 = zp_res_93484;
                
                r_93482 = r_tmp_101058;
            }
            defunc_0_lifted_lambda_res_93480 = r_93482;
            // futhark/microgpt.fut:178:21-32
            
            double zs_res_93485 = 1.0 / defunc_0_lifted_lambda_res_93480;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98165 = 0; i_98165 < (int64_t) 16; i_98165++) {
                // futhark/microgpt.fut:178:5-15
                
                double zt_lhs_93492 = ((double *) mem_99314)[i_98165];
                
                // futhark/microgpt.fut:178:5-32
                
                double zt_res_93493 = zs_res_93485 * zt_lhs_93492;
                
                ((double *) mem_99321)[i_98165] = zt_res_93493;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99309, i_98169 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99321, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98177 = 0; i_98177 < (int64_t) 16; i_98177++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98173 = 0; i_98173 < (int64_t) 4; i_98173++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93508;
                double r_93510 = 0.0;
                
                for (int64_t i_93509 = 0; i_93509 < (int64_t) 16; i_93509++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93511 = ((double *) mem_99309)[i_98177 * (int64_t) 16 + i_93509];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93512 = ((double *) mem_99190)[i_98181 * (int64_t) 64 + i_93509 * (int64_t) 4 + i_98173];
                    
                    // futhark/microgpt.fut:179:26-71
                    
                    double zt_res_93513 = zt_lhs_93511 * zt_rhs_93512;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93514 = r_93510 + zt_res_93513;
                    double r_tmp_101062 = zp_res_93514;
                    
                    r_93510 = r_tmp_101062;
                }
                defunc_0_lifted_lambda_res_93508 = r_93510;
                ((double *) mem_99337)[i_98173] = defunc_0_lifted_lambda_res_93508;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99332, i_98177 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99337, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_99271, i_98181 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99332, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99353_cached_sizze_101417 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99353, &mem_99353_cached_sizze_101417, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99358_cached_sizze_101418 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99358, &mem_99358_cached_sizze_101418, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98189 = 0; i_98189 < (int64_t) 16; i_98189++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98185 = 0; i_98185 < (int64_t) 16; i_98185++) {
            // futhark/microgpt.fut:180:55-58
            
            int64_t tmp_93526 = sdiv64(i_98185, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-60
            
            bool x_93527 = sle64((int64_t) 0, tmp_93526);
            
            // futhark/microgpt.fut:180:45-60
            
            bool y_93528 = slt64(tmp_93526, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-60
            
            bool bounds_check_93529 = x_93527 && y_93528;
            
            // futhark/microgpt.fut:180:45-60
            
            bool index_certs_93530;
            
            if (!bounds_check_93529) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93526, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:180:45-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:180:16-81\n   #6  futhark/microgpt.fut:382:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:180:75-78
            
            int64_t tmp_93531 = smod64(i_98185, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-80
            
            bool x_93532 = sle64((int64_t) 0, tmp_93531);
            
            // futhark/microgpt.fut:180:45-80
            
            bool y_93533 = slt64(tmp_93531, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-80
            
            bool bounds_check_93534 = x_93532 && y_93533;
            
            // futhark/microgpt.fut:180:45-80
            
            bool index_certs_93535;
            
            if (!bounds_check_93534) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93531, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:180:45-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:180:16-81\n   #6  futhark/microgpt.fut:382:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_93536 = ((double *) mem_99271)[tmp_93526 * (int64_t) 64 + i_98189 * (int64_t) 4 + tmp_93531];
            
            ((double *) mem_99358)[i_98185] = lifted_lambda_res_93536;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99353, i_98189 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99358, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99369_cached_sizze_101419 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99369, &mem_99369_cached_sizze_101419, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99374_cached_sizze_101420 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99374, &mem_99374_cached_sizze_101420, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98197 = 0; i_98197 < (int64_t) 16; i_98197++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98193 = 0; i_98193 < (int64_t) 16; i_98193++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93551;
            double r_93553 = 0.0;
            
            for (int64_t i_93552 = 0; i_93552 < (int64_t) 16; i_93552++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93554 = ((double *) wout_mem_99055.mem)[i_98193 * (int64_t) 16 + i_93552];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_93555 = ((double *) mem_99353)[i_98197 * (int64_t) 16 + i_93552];
                
                // futhark/microgpt.fut:181:67-107
                
                double zt_res_93556 = zt_lhs_93554 * zt_rhs_93555;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93557 = r_93553 + zt_res_93556;
                double r_tmp_101067 = zp_res_93557;
                
                r_93553 = r_tmp_101067;
            }
            defunc_0_lifted_lambda_res_93551 = r_93553;
            ((double *) mem_99374)[i_98193] = defunc_0_lifted_lambda_res_93551;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99369, i_98197 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99374, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99385_cached_sizze_101421 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99385, &mem_99385_cached_sizze_101421, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99390_cached_sizze_101422 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99390, &mem_99390_cached_sizze_101422, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98205 = 0; i_98205 < (int64_t) 16; i_98205++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98201 = 0; i_98201 < (int64_t) 16; i_98201++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_93572 = ((double *) mem_99369)[i_98205 * (int64_t) 16 + i_98201];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_93573 = ((double *) mem_99096)[i_98205 * (int64_t) 16 + i_98201];
            
            // futhark/microgpt.fut:182:46-84
            
            double zp_res_93574 = zp_lhs_93572 + zp_rhs_93573;
            
            ((double *) mem_99390)[i_98201] = zp_res_93574;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99385, i_98205 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99390, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99401_cached_sizze_101423 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99401, &mem_99401_cached_sizze_101423, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99406_cached_sizze_101424 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99406, &mem_99406_cached_sizze_101424, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99413_cached_sizze_101425 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99413, &mem_99413_cached_sizze_101425, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98217 = 0; i_98217 < (int64_t) 16; i_98217++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98209 = 0; i_98209 < (int64_t) 16; i_98209++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93589 = ((double *) mem_99385)[i_98217 * (int64_t) 16 + i_98209];
            
            // futhark/microgpt.fut:183:78-117
            
            double zt_res_93590 = zt_lhs_93589 * zt_lhs_93589;
            
            ((double *) mem_99406)[i_98209] = zt_res_93590;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_93592;
        double r_93594 = 0.0;
        
        for (int64_t i_93593 = 0; i_93593 < (int64_t) 16; i_93593++) {
            // futhark/microgpt.fut:184:37-47
            
            double lifted_lambda_res_93595 = ((double *) mem_99406)[i_93593];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_93596 = r_93594 + lifted_lambda_res_93595;
            double r_tmp_101072 = zp_res_93596;
            
            r_93594 = r_tmp_101072;
        }
        defunc_0_lifted_lambda_res_93592 = r_93594;
        // futhark/microgpt.fut:184:17-64
        
        double zs_res_93597 = defunc_0_lifted_lambda_res_93592 / 16.0;
        
        // futhark/microgpt.fut:185:24-55
        
        double zp_res_93598 = 1.0e-5 + zs_res_93597;
        
        // futhark/microgpt.fut:185:16-55
        
        double sqrt_res_93599 = futrts_sqrt64(zp_res_93598);
        
        // futhark/microgpt.fut:186:28-39
        
        double zs_res_93600 = 1.0 / sqrt_res_93599;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98213 = 0; i_98213 < (int64_t) 16; i_98213++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_93607 = ((double *) mem_99385)[i_98217 * (int64_t) 16 + i_98213];
            
            // futhark/microgpt.fut:186:5-39
            
            double zt_res_93608 = zs_res_93600 * zt_lhs_93607;
            
            ((double *) mem_99413)[i_98213] = zt_res_93608;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99401, i_98217 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99413, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99424_cached_sizze_101426 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99424, &mem_99424_cached_sizze_101426, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99429_cached_sizze_101427 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99429, &mem_99429_cached_sizze_101427, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98225 = 0; i_98225 < (int64_t) 16; i_98225++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98221 = 0; i_98221 < (int64_t) 64; i_98221++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93624;
            double r_93626 = 0.0;
            
            for (int64_t i_93625 = 0; i_93625 < (int64_t) 16; i_93625++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93627 = ((double *) wup_mem_99059.mem)[i_98221 * (int64_t) 16 + i_93625];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_93628 = ((double *) mem_99401)[i_98225 * (int64_t) 16 + i_93625];
                
                // futhark/microgpt.fut:187:67-106
                
                double zt_res_93629 = zt_lhs_93627 * zt_rhs_93628;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93630 = r_93626 + zt_res_93629;
                double r_tmp_101076 = zp_res_93630;
                
                r_93626 = r_tmp_101076;
            }
            defunc_0_lifted_lambda_res_93624 = r_93626;
            ((double *) mem_99429)[i_98221] = defunc_0_lifted_lambda_res_93624;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99424, i_98225 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99429, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99440_cached_sizze_101428 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99440, &mem_99440_cached_sizze_101428, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99445_cached_sizze_101429 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99445, &mem_99445_cached_sizze_101429, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98233 = 0; i_98233 < (int64_t) 16; i_98233++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98229 = 0; i_98229 < (int64_t) 64; i_98229++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_93645 = ((double *) mem_99424)[i_98233 * (int64_t) 64 + i_98229];
            
            // futhark/microgpt.fut:188:45-73
            
            double max_res_93646 = fmax64(0.0, max_arg0_93645);
            
            ((double *) mem_99445)[i_98229] = max_res_93646;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99440, i_98233 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99445, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99456_cached_sizze_101430 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99456, &mem_99456_cached_sizze_101430, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99461_cached_sizze_101431 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99461, &mem_99461_cached_sizze_101431, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98241 = 0; i_98241 < (int64_t) 16; i_98241++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98237 = 0; i_98237 < (int64_t) 16; i_98237++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93661;
            double r_93663 = 0.0;
            
            for (int64_t i_93662 = 0; i_93662 < (int64_t) 64; i_93662++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93664 = ((double *) wdown_mem_99053.mem)[i_98237 * (int64_t) 64 + i_93662];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_93665 = ((double *) mem_99440)[i_98241 * (int64_t) 64 + i_93662];
                
                // futhark/microgpt.fut:189:67-108
                
                double zt_res_93666 = zt_lhs_93664 * zt_rhs_93665;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93667 = r_93663 + zt_res_93666;
                double r_tmp_101081 = zp_res_93667;
                
                r_93663 = r_tmp_101081;
            }
            defunc_0_lifted_lambda_res_93661 = r_93663;
            ((double *) mem_99461)[i_98237] = defunc_0_lifted_lambda_res_93661;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99456, i_98241 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99461, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99472_cached_sizze_101432 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99472, &mem_99472_cached_sizze_101432, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99477_cached_sizze_101433 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99477, &mem_99477_cached_sizze_101433, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98249 = 0; i_98249 < (int64_t) 16; i_98249++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98245 = 0; i_98245 < (int64_t) 16; i_98245++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_93682 = ((double *) mem_99456)[i_98249 * (int64_t) 16 + i_98245];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_93683 = ((double *) mem_99385)[i_98249 * (int64_t) 16 + i_98245];
            
            // futhark/microgpt.fut:190:46-85
            
            double zp_res_93684 = zp_lhs_93682 + zp_rhs_93683;
            
            ((double *) mem_99477)[i_98245] = zp_res_93684;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99472, i_98249 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99477, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_99488, (int64_t) 3456, "mem_99488")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99493_cached_sizze_101434 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99493, &mem_99493_cached_sizze_101434, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_98257 = 0; i_98257 < (int64_t) 16; i_98257++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98253 = 0; i_98253 < (int64_t) 27; i_98253++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93700;
            double r_93702 = 0.0;
            
            for (int64_t i_93701 = 0; i_93701 < (int64_t) 16; i_93701++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93703 = ((double *) wvoc_mem_99061.mem)[i_98253 * (int64_t) 16 + i_93701];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_93704 = ((double *) mem_99472)[i_98257 * (int64_t) 16 + i_93701];
                
                // futhark/microgpt.fut:191:56-96
                
                double zt_res_93705 = zt_lhs_93703 * zt_rhs_93704;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93706 = r_93702 + zt_res_93705;
                double r_tmp_101086 = zp_res_93706;
                
                r_93702 = r_tmp_101086;
            }
            defunc_0_lifted_lambda_res_93700 = r_93702;
            ((double *) mem_99493)[i_98253] = defunc_0_lifted_lambda_res_93700;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_99488.mem, i_98257 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99493, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_101017, &mem_99488, "mem_99488") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101381, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_99064);
        free(mem_99069);
        free(mem_99080);
        free(mem_99085);
        free(mem_99096);
        free(mem_99101);
        free(mem_99108);
        free(mem_99119);
        free(mem_99124);
        free(mem_99131);
        free(mem_99142);
        free(mem_99143);
        free(mem_99144);
        free(mem_99157);
        free(mem_99158);
        free(mem_99159);
        free(mem_99190);
        free(mem_99191);
        free(mem_99192);
        free(mem_99208);
        free(mem_99209);
        free(mem_99210);
        free(mem_99223);
        free(mem_99224);
        free(mem_99225);
        free(mem_99271);
        free(mem_99277);
        free(mem_99282);
        free(mem_99293);
        free(mem_99298);
        free(mem_99309);
        free(mem_99314);
        free(mem_99321);
        free(mem_99332);
        free(mem_99337);
        free(mem_99353);
        free(mem_99358);
        free(mem_99369);
        free(mem_99374);
        free(mem_99385);
        free(mem_99390);
        free(mem_99401);
        free(mem_99406);
        free(mem_99413);
        free(mem_99424);
        free(mem_99429);
        free(mem_99440);
        free(mem_99445);
        free(mem_99456);
        free(mem_99461);
        free(mem_99472);
        free(mem_99477);
        free(mem_99493);
        if (memblock_unref(ctx, &mem_99488, "mem_99488") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_101435, struct memblock *mem_out_p_101436, struct memblock *mem_out_p_101437, struct memblock *mem_out_p_101438, struct memblock *mem_out_p_101439, struct memblock *mem_out_p_101440, struct memblock *mem_out_p_101441, struct memblock *mem_out_p_101442, struct memblock *mem_out_p_101443, struct memblock wte_mem_99053, struct memblock wpe_mem_99054, struct memblock wqry_mem_99055, struct memblock wkey_mem_99056, struct memblock wval_mem_99057, struct memblock wout_mem_99058, struct memblock wup_mem_99059, struct memblock wdown_mem_99060, struct memblock wvoc_mem_99061)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    if (memblock_set(ctx, &mem_out_101017, &wdown_mem_99060, "wdown_mem_99060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101018, &wkey_mem_99056, "wkey_mem_99056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101019, &wout_mem_99058, "wout_mem_99058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101020, &wpe_mem_99054, "wpe_mem_99054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101021, &wqry_mem_99055, "wqry_mem_99055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101022, &wte_mem_99053, "wte_mem_99053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101023, &wup_mem_99059, "wup_mem_99059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101024, &wval_mem_99057, "wval_mem_99057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101025, &wvoc_mem_99061, "wvoc_mem_99061") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101435, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101436, &mem_out_101018, "mem_out_101018") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101437, &mem_out_101019, "mem_out_101019") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101438, &mem_out_101020, "mem_out_101020") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101439, &mem_out_101021, "mem_out_101021") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101440, &mem_out_101022, "mem_out_101022") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101441, &mem_out_101023, "mem_out_101023") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101442, &mem_out_101024, "mem_out_101024") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101443, &mem_out_101025, "mem_out_101025") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_101025, "mem_out_101025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101024, "mem_out_101024") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101023, "mem_out_101023") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101022, "mem_out_101022") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101021, "mem_out_101021") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101020, "mem_out_101020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101019, "mem_out_101019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101018, "mem_out_101018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_101444, struct memblock *mem_out_p_101445, struct memblock *mem_out_p_101446, struct memblock *mem_out_p_101447, struct memblock *mem_out_p_101448, struct memblock *mem_out_p_101449, struct memblock *mem_out_p_101450, struct memblock *mem_out_p_101451, struct memblock *mem_out_p_101452, struct memblock *mem_out_p_101453, struct memblock *mem_out_p_101454, struct memblock *mem_out_p_101455, struct memblock *mem_out_p_101456, struct memblock *mem_out_p_101457, struct memblock *mem_out_p_101458, struct memblock *mem_out_p_101459, struct memblock *mem_out_p_101460, struct memblock *mem_out_p_101461, struct memblock *mem_out_p_101462, struct memblock *mem_out_p_101463, struct memblock *mem_out_p_101464, struct memblock *mem_out_p_101465, struct memblock *mem_out_p_101466, struct memblock *mem_out_p_101467, struct memblock *mem_out_p_101468, struct memblock *mem_out_p_101469, struct memblock *mem_out_p_101470, struct memblock wdown_mem_99053, struct memblock wkey_mem_99054, struct memblock wout_mem_99055, struct memblock wpe_mem_99056, struct memblock wqry_mem_99057, struct memblock wte_mem_99058, struct memblock wup_mem_99059, struct memblock wval_mem_99060, struct memblock wvoc_mem_99061, struct memblock wdown_mem_99062, struct memblock wkey_mem_99063, struct memblock wout_mem_99064, struct memblock wpe_mem_99065, struct memblock wqry_mem_99066, struct memblock wte_mem_99067, struct memblock wup_mem_99068, struct memblock wval_mem_99069, struct memblock wvoc_mem_99070, struct memblock wdown_mem_99071, struct memblock wkey_mem_99072, struct memblock wout_mem_99073, struct memblock wpe_mem_99074, struct memblock wqry_mem_99075, struct memblock wte_mem_99076, struct memblock wup_mem_99077, struct memblock wval_mem_99078, struct memblock wvoc_mem_99079, struct memblock masks_mem_99080, struct memblock dls_mem_99081, struct memblock seqs_mem_99082)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_99191_cached_sizze_101471 = 0;
    unsigned char *mem_99191 = NULL;
    int64_t mem_99192_cached_sizze_101472 = 0;
    unsigned char *mem_99192 = NULL;
    int64_t mem_99201_cached_sizze_101473 = 0;
    unsigned char *mem_99201 = NULL;
    int64_t mem_99208_cached_sizze_101474 = 0;
    unsigned char *mem_99208 = NULL;
    int64_t mem_99223_cached_sizze_101475 = 0;
    unsigned char *mem_99223 = NULL;
    int64_t mem_99224_cached_sizze_101476 = 0;
    unsigned char *mem_99224 = NULL;
    int64_t mem_99233_cached_sizze_101477 = 0;
    unsigned char *mem_99233 = NULL;
    int64_t mem_99240_cached_sizze_101478 = 0;
    unsigned char *mem_99240 = NULL;
    int64_t mem_99255_cached_sizze_101479 = 0;
    unsigned char *mem_99255 = NULL;
    int64_t mem_99256_cached_sizze_101480 = 0;
    unsigned char *mem_99256 = NULL;
    int64_t mem_99265_cached_sizze_101481 = 0;
    unsigned char *mem_99265 = NULL;
    int64_t mem_99266_cached_sizze_101482 = 0;
    unsigned char *mem_99266 = NULL;
    int64_t mem_99287_cached_sizze_101483 = 0;
    unsigned char *mem_99287 = NULL;
    int64_t mem_99288_cached_sizze_101484 = 0;
    unsigned char *mem_99288 = NULL;
    int64_t mem_99289_cached_sizze_101485 = 0;
    unsigned char *mem_99289 = NULL;
    int64_t mem_99301_cached_sizze_101486 = 0;
    unsigned char *mem_99301 = NULL;
    int64_t mem_99302_cached_sizze_101487 = 0;
    unsigned char *mem_99302 = NULL;
    int64_t mem_99326_cached_sizze_101488 = 0;
    unsigned char *mem_99326 = NULL;
    int64_t mem_99327_cached_sizze_101489 = 0;
    unsigned char *mem_99327 = NULL;
    int64_t mem_99328_cached_sizze_101490 = 0;
    unsigned char *mem_99328 = NULL;
    int64_t mem_99329_cached_sizze_101491 = 0;
    unsigned char *mem_99329 = NULL;
    int64_t mem_99330_cached_sizze_101492 = 0;
    unsigned char *mem_99330 = NULL;
    int64_t mem_99349_cached_sizze_101493 = 0;
    unsigned char *mem_99349 = NULL;
    int64_t mem_99350_cached_sizze_101494 = 0;
    unsigned char *mem_99350 = NULL;
    int64_t mem_99351_cached_sizze_101495 = 0;
    unsigned char *mem_99351 = NULL;
    int64_t mem_99388_cached_sizze_101496 = 0;
    unsigned char *mem_99388 = NULL;
    int64_t mem_99389_cached_sizze_101497 = 0;
    unsigned char *mem_99389 = NULL;
    int64_t mem_99390_cached_sizze_101498 = 0;
    unsigned char *mem_99390 = NULL;
    int64_t mem_99406_cached_sizze_101499 = 0;
    unsigned char *mem_99406 = NULL;
    int64_t mem_99407_cached_sizze_101500 = 0;
    unsigned char *mem_99407 = NULL;
    int64_t mem_99408_cached_sizze_101501 = 0;
    unsigned char *mem_99408 = NULL;
    int64_t mem_99421_cached_sizze_101502 = 0;
    unsigned char *mem_99421 = NULL;
    int64_t mem_99422_cached_sizze_101503 = 0;
    unsigned char *mem_99422 = NULL;
    int64_t mem_99423_cached_sizze_101504 = 0;
    unsigned char *mem_99423 = NULL;
    int64_t mem_99469_cached_sizze_101505 = 0;
    unsigned char *mem_99469 = NULL;
    int64_t mem_99470_cached_sizze_101506 = 0;
    unsigned char *mem_99470 = NULL;
    int64_t mem_99481_cached_sizze_101507 = 0;
    unsigned char *mem_99481 = NULL;
    int64_t mem_99482_cached_sizze_101508 = 0;
    unsigned char *mem_99482 = NULL;
    int64_t mem_99491_cached_sizze_101509 = 0;
    unsigned char *mem_99491 = NULL;
    int64_t mem_99492_cached_sizze_101510 = 0;
    unsigned char *mem_99492 = NULL;
    int64_t mem_99513_cached_sizze_101511 = 0;
    unsigned char *mem_99513 = NULL;
    int64_t mem_99518_cached_sizze_101512 = 0;
    unsigned char *mem_99518 = NULL;
    int64_t mem_99525_cached_sizze_101513 = 0;
    unsigned char *mem_99525 = NULL;
    int64_t mem_99546_cached_sizze_101514 = 0;
    unsigned char *mem_99546 = NULL;
    int64_t mem_99547_cached_sizze_101515 = 0;
    unsigned char *mem_99547 = NULL;
    int64_t mem_99555_cached_sizze_101516 = 0;
    unsigned char *mem_99555 = NULL;
    int64_t mem_99569_cached_sizze_101517 = 0;
    unsigned char *mem_99569 = NULL;
    int64_t mem_99574_cached_sizze_101518 = 0;
    unsigned char *mem_99574 = NULL;
    int64_t mem_99585_cached_sizze_101519 = 0;
    unsigned char *mem_99585 = NULL;
    int64_t mem_99590_cached_sizze_101520 = 0;
    unsigned char *mem_99590 = NULL;
    int64_t mem_99601_cached_sizze_101521 = 0;
    unsigned char *mem_99601 = NULL;
    int64_t mem_99602_cached_sizze_101522 = 0;
    unsigned char *mem_99602 = NULL;
    int64_t mem_99611_cached_sizze_101523 = 0;
    unsigned char *mem_99611 = NULL;
    int64_t mem_99612_cached_sizze_101524 = 0;
    unsigned char *mem_99612 = NULL;
    int64_t mem_99633_cached_sizze_101525 = 0;
    unsigned char *mem_99633 = NULL;
    int64_t mem_99634_cached_sizze_101526 = 0;
    unsigned char *mem_99634 = NULL;
    int64_t mem_99642_cached_sizze_101527 = 0;
    unsigned char *mem_99642 = NULL;
    int64_t mem_99656_cached_sizze_101528 = 0;
    unsigned char *mem_99656 = NULL;
    int64_t mem_99657_cached_sizze_101529 = 0;
    unsigned char *mem_99657 = NULL;
    int64_t mem_99665_cached_sizze_101530 = 0;
    unsigned char *mem_99665 = NULL;
    int64_t mem_99679_cached_sizze_101531 = 0;
    unsigned char *mem_99679 = NULL;
    int64_t mem_99684_cached_sizze_101532 = 0;
    unsigned char *mem_99684 = NULL;
    int64_t mem_99695_cached_sizze_101533 = 0;
    unsigned char *mem_99695 = NULL;
    int64_t mem_99700_cached_sizze_101534 = 0;
    unsigned char *mem_99700 = NULL;
    int64_t mem_99711_cached_sizze_101535 = 0;
    unsigned char *mem_99711 = NULL;
    int64_t mem_99716_cached_sizze_101536 = 0;
    unsigned char *mem_99716 = NULL;
    int64_t mem_99727_cached_sizze_101537 = 0;
    unsigned char *mem_99727 = NULL;
    int64_t mem_99734_cached_sizze_101538 = 0;
    unsigned char *mem_99734 = NULL;
    int64_t mem_99739_cached_sizze_101539 = 0;
    unsigned char *mem_99739 = NULL;
    int64_t mem_99750_cached_sizze_101540 = 0;
    unsigned char *mem_99750 = NULL;
    int64_t mem_99755_cached_sizze_101541 = 0;
    unsigned char *mem_99755 = NULL;
    int64_t mem_99766_cached_sizze_101542 = 0;
    unsigned char *mem_99766 = NULL;
    int64_t mem_99773_cached_sizze_101543 = 0;
    unsigned char *mem_99773 = NULL;
    int64_t mem_99777_cached_sizze_101544 = 0;
    unsigned char *mem_99777 = NULL;
    int64_t mem_99787_cached_sizze_101545 = 0;
    unsigned char *mem_99787 = NULL;
    int64_t mem_99792_cached_sizze_101546 = 0;
    unsigned char *mem_99792 = NULL;
    int64_t mem_99799_cached_sizze_101547 = 0;
    unsigned char *mem_99799 = NULL;
    int64_t mem_99810_cached_sizze_101548 = 0;
    unsigned char *mem_99810 = NULL;
    int64_t mem_99815_cached_sizze_101549 = 0;
    unsigned char *mem_99815 = NULL;
    int64_t mem_99826_cached_sizze_101550 = 0;
    unsigned char *mem_99826 = NULL;
    int64_t mem_99833_cached_sizze_101551 = 0;
    unsigned char *mem_99833 = NULL;
    int64_t mem_99838_cached_sizze_101552 = 0;
    unsigned char *mem_99838 = NULL;
    int64_t mem_99849_cached_sizze_101553 = 0;
    unsigned char *mem_99849 = NULL;
    int64_t mem_99854_cached_sizze_101554 = 0;
    unsigned char *mem_99854 = NULL;
    int64_t mem_99865_cached_sizze_101555 = 0;
    unsigned char *mem_99865 = NULL;
    int64_t mem_99870_cached_sizze_101556 = 0;
    unsigned char *mem_99870 = NULL;
    int64_t mem_99881_cached_sizze_101557 = 0;
    unsigned char *mem_99881 = NULL;
    int64_t mem_99882_cached_sizze_101558 = 0;
    unsigned char *mem_99882 = NULL;
    int64_t mem_99891_cached_sizze_101559 = 0;
    unsigned char *mem_99891 = NULL;
    int64_t mem_99892_cached_sizze_101560 = 0;
    unsigned char *mem_99892 = NULL;
    int64_t mem_99913_cached_sizze_101561 = 0;
    unsigned char *mem_99913 = NULL;
    int64_t mem_99918_cached_sizze_101562 = 0;
    unsigned char *mem_99918 = NULL;
    int64_t mem_99929_cached_sizze_101563 = 0;
    unsigned char *mem_99929 = NULL;
    int64_t mem_99934_cached_sizze_101564 = 0;
    unsigned char *mem_99934 = NULL;
    int64_t mem_99945_cached_sizze_101565 = 0;
    unsigned char *mem_99945 = NULL;
    int64_t mem_99952_cached_sizze_101566 = 0;
    unsigned char *mem_99952 = NULL;
    int64_t mem_99959_cached_sizze_101567 = 0;
    unsigned char *mem_99959 = NULL;
    int64_t mem_99969_cached_sizze_101568 = 0;
    unsigned char *mem_99969 = NULL;
    int64_t mem_99974_cached_sizze_101569 = 0;
    unsigned char *mem_99974 = NULL;
    int64_t mem_99985_cached_sizze_101570 = 0;
    unsigned char *mem_99985 = NULL;
    int64_t mem_99986_cached_sizze_101571 = 0;
    unsigned char *mem_99986 = NULL;
    int64_t mem_99995_cached_sizze_101572 = 0;
    unsigned char *mem_99995 = NULL;
    int64_t mem_99996_cached_sizze_101573 = 0;
    unsigned char *mem_99996 = NULL;
    int64_t mem_100017_cached_sizze_101574 = 0;
    unsigned char *mem_100017 = NULL;
    int64_t mem_100018_cached_sizze_101575 = 0;
    unsigned char *mem_100018 = NULL;
    int64_t mem_100029_cached_sizze_101576 = 0;
    unsigned char *mem_100029 = NULL;
    int64_t mem_100030_cached_sizze_101577 = 0;
    unsigned char *mem_100030 = NULL;
    int64_t mem_100039_cached_sizze_101578 = 0;
    unsigned char *mem_100039 = NULL;
    int64_t mem_100046_cached_sizze_101579 = 0;
    unsigned char *mem_100046 = NULL;
    int64_t mem_100071_cached_sizze_101580 = 0;
    unsigned char *mem_100071 = NULL;
    int64_t mem_100072_cached_sizze_101581 = 0;
    unsigned char *mem_100072 = NULL;
    int64_t mem_100073_cached_sizze_101582 = 0;
    unsigned char *mem_100073 = NULL;
    int64_t mem_100088_cached_sizze_101583 = 0;
    unsigned char *mem_100088 = NULL;
    int64_t mem_100089_cached_sizze_101584 = 0;
    unsigned char *mem_100089 = NULL;
    int64_t mem_100090_cached_sizze_101585 = 0;
    unsigned char *mem_100090 = NULL;
    int64_t mem_100102_cached_sizze_101586 = 0;
    unsigned char *mem_100102 = NULL;
    int64_t mem_100109_cached_sizze_101587 = 0;
    unsigned char *mem_100109 = NULL;
    int64_t mem_100116_cached_sizze_101588 = 0;
    unsigned char *mem_100116 = NULL;
    int64_t mem_100148_cached_sizze_101589 = 0;
    unsigned char *mem_100148 = NULL;
    int64_t mem_100149_cached_sizze_101590 = 0;
    unsigned char *mem_100149 = NULL;
    int64_t mem_100160_cached_sizze_101591 = 0;
    unsigned char *mem_100160 = NULL;
    int64_t mem_100161_cached_sizze_101592 = 0;
    unsigned char *mem_100161 = NULL;
    int64_t mem_100170_cached_sizze_101593 = 0;
    unsigned char *mem_100170 = NULL;
    int64_t mem_100177_cached_sizze_101594 = 0;
    unsigned char *mem_100177 = NULL;
    int64_t mem_100202_cached_sizze_101595 = 0;
    unsigned char *mem_100202 = NULL;
    int64_t mem_100208_cached_sizze_101596 = 0;
    unsigned char *mem_100208 = NULL;
    int64_t mem_100213_cached_sizze_101597 = 0;
    unsigned char *mem_100213 = NULL;
    int64_t mem_100229_cached_sizze_101598 = 0;
    unsigned char *mem_100229 = NULL;
    int64_t mem_100234_cached_sizze_101599 = 0;
    unsigned char *mem_100234 = NULL;
    int64_t mem_100245_cached_sizze_101600 = 0;
    unsigned char *mem_100245 = NULL;
    int64_t mem_100250_cached_sizze_101601 = 0;
    unsigned char *mem_100250 = NULL;
    int64_t mem_100261_cached_sizze_101602 = 0;
    unsigned char *mem_100261 = NULL;
    int64_t mem_100267_cached_sizze_101603 = 0;
    unsigned char *mem_100267 = NULL;
    int64_t mem_100272_cached_sizze_101604 = 0;
    unsigned char *mem_100272 = NULL;
    int64_t mem_100288_cached_sizze_101605 = 0;
    unsigned char *mem_100288 = NULL;
    int64_t mem_100294_cached_sizze_101606 = 0;
    unsigned char *mem_100294 = NULL;
    int64_t mem_100299_cached_sizze_101607 = 0;
    unsigned char *mem_100299 = NULL;
    int64_t mem_100315_cached_sizze_101608 = 0;
    unsigned char *mem_100315 = NULL;
    int64_t mem_100320_cached_sizze_101609 = 0;
    unsigned char *mem_100320 = NULL;
    int64_t mem_100331_cached_sizze_101610 = 0;
    unsigned char *mem_100331 = NULL;
    int64_t mem_100337_cached_sizze_101611 = 0;
    unsigned char *mem_100337 = NULL;
    int64_t mem_100342_cached_sizze_101612 = 0;
    unsigned char *mem_100342 = NULL;
    int64_t mem_100358_cached_sizze_101613 = 0;
    unsigned char *mem_100358 = NULL;
    int64_t mem_100364_cached_sizze_101614 = 0;
    unsigned char *mem_100364 = NULL;
    int64_t mem_100369_cached_sizze_101615 = 0;
    unsigned char *mem_100369 = NULL;
    int64_t mem_100385_cached_sizze_101616 = 0;
    unsigned char *mem_100385 = NULL;
    int64_t mem_100391_cached_sizze_101617 = 0;
    unsigned char *mem_100391 = NULL;
    int64_t mem_100396_cached_sizze_101618 = 0;
    unsigned char *mem_100396 = NULL;
    int64_t mem_100412_cached_sizze_101619 = 0;
    unsigned char *mem_100412 = NULL;
    int64_t mem_100413_cached_sizze_101620 = 0;
    unsigned char *mem_100413 = NULL;
    int64_t mem_100424_cached_sizze_101621 = 0;
    unsigned char *mem_100424 = NULL;
    int64_t mem_100425_cached_sizze_101622 = 0;
    unsigned char *mem_100425 = NULL;
    int64_t mem_100434_cached_sizze_101623 = 0;
    unsigned char *mem_100434 = NULL;
    int64_t mem_100435_cached_sizze_101624 = 0;
    unsigned char *mem_100435 = NULL;
    int64_t mem_100466_cached_sizze_101625 = 0;
    unsigned char *mem_100466 = NULL;
    int64_t mem_100467_cached_sizze_101626 = 0;
    unsigned char *mem_100467 = NULL;
    int64_t mem_100468_cached_sizze_101627 = 0;
    unsigned char *mem_100468 = NULL;
    int64_t mem_100481_cached_sizze_101628 = 0;
    unsigned char *mem_100481 = NULL;
    int64_t mem_100482_cached_sizze_101629 = 0;
    unsigned char *mem_100482 = NULL;
    int64_t mem_100483_cached_sizze_101630 = 0;
    unsigned char *mem_100483 = NULL;
    int64_t mem_100514_cached_sizze_101631 = 0;
    unsigned char *mem_100514 = NULL;
    int64_t mem_100515_cached_sizze_101632 = 0;
    unsigned char *mem_100515 = NULL;
    int64_t mem_100516_cached_sizze_101633 = 0;
    unsigned char *mem_100516 = NULL;
    int64_t mem_100517_cached_sizze_101634 = 0;
    unsigned char *mem_100517 = NULL;
    int64_t mem_100534_cached_sizze_101635 = 0;
    unsigned char *mem_100534 = NULL;
    int64_t mem_100535_cached_sizze_101636 = 0;
    unsigned char *mem_100535 = NULL;
    int64_t mem_100536_cached_sizze_101637 = 0;
    unsigned char *mem_100536 = NULL;
    int64_t mem_100537_cached_sizze_101638 = 0;
    unsigned char *mem_100537 = NULL;
    int64_t mem_100578_cached_sizze_101639 = 0;
    unsigned char *mem_100578 = NULL;
    int64_t mem_100585_cached_sizze_101640 = 0;
    unsigned char *mem_100585 = NULL;
    int64_t mem_100592_cached_sizze_101641 = 0;
    unsigned char *mem_100592 = NULL;
    int64_t mem_100602_cached_sizze_101642 = 0;
    unsigned char *mem_100602 = NULL;
    int64_t mem_100607_cached_sizze_101643 = 0;
    unsigned char *mem_100607 = NULL;
    int64_t mem_100618_cached_sizze_101644 = 0;
    unsigned char *mem_100618 = NULL;
    int64_t mem_100625_cached_sizze_101645 = 0;
    unsigned char *mem_100625 = NULL;
    int64_t mem_100632_cached_sizze_101646 = 0;
    unsigned char *mem_100632 = NULL;
    int64_t mem_100642_cached_sizze_101647 = 0;
    unsigned char *mem_100642 = NULL;
    int64_t mem_100647_cached_sizze_101648 = 0;
    unsigned char *mem_100647 = NULL;
    int64_t mem_100658_cached_sizze_101649 = 0;
    unsigned char *mem_100658 = NULL;
    int64_t mem_100659_cached_sizze_101650 = 0;
    unsigned char *mem_100659 = NULL;
    int64_t mem_100668_cached_sizze_101651 = 0;
    unsigned char *mem_100668 = NULL;
    int64_t mem_100669_cached_sizze_101652 = 0;
    unsigned char *mem_100669 = NULL;
    int64_t mem_100690_cached_sizze_101653 = 0;
    unsigned char *mem_100690 = NULL;
    int64_t mem_100695_cached_sizze_101654 = 0;
    unsigned char *mem_100695 = NULL;
    int64_t mem_100706_cached_sizze_101655 = 0;
    unsigned char *mem_100706 = NULL;
    int64_t mem_100707_cached_sizze_101656 = 0;
    unsigned char *mem_100707 = NULL;
    int64_t mem_100716_cached_sizze_101657 = 0;
    unsigned char *mem_100716 = NULL;
    int64_t mem_100717_cached_sizze_101658 = 0;
    unsigned char *mem_100717 = NULL;
    struct memblock mem_param_tmp_101070;
    
    mem_param_tmp_101070.references = NULL;
    
    struct memblock mem_param_tmp_101069;
    
    mem_param_tmp_101069.references = NULL;
    
    struct memblock mem_param_tmp_101068;
    
    mem_param_tmp_101068.references = NULL;
    
    struct memblock mem_param_tmp_101067;
    
    mem_param_tmp_101067.references = NULL;
    
    struct memblock mem_param_tmp_101066;
    
    mem_param_tmp_101066.references = NULL;
    
    struct memblock mem_param_tmp_101065;
    
    mem_param_tmp_101065.references = NULL;
    
    struct memblock mem_param_tmp_101064;
    
    mem_param_tmp_101064.references = NULL;
    
    struct memblock mem_param_tmp_101063;
    
    mem_param_tmp_101063.references = NULL;
    
    struct memblock mem_param_tmp_101062;
    
    mem_param_tmp_101062.references = NULL;
    
    struct memblock mem_param_tmp_101061;
    
    mem_param_tmp_101061.references = NULL;
    
    struct memblock mem_param_tmp_101060;
    
    mem_param_tmp_101060.references = NULL;
    
    struct memblock mem_param_tmp_101059;
    
    mem_param_tmp_101059.references = NULL;
    
    struct memblock mem_param_tmp_101058;
    
    mem_param_tmp_101058.references = NULL;
    
    struct memblock mem_param_tmp_101057;
    
    mem_param_tmp_101057.references = NULL;
    
    struct memblock mem_param_tmp_101056;
    
    mem_param_tmp_101056.references = NULL;
    
    struct memblock mem_param_tmp_101055;
    
    mem_param_tmp_101055.references = NULL;
    
    struct memblock mem_param_tmp_101054;
    
    mem_param_tmp_101054.references = NULL;
    
    struct memblock mem_param_tmp_101053;
    
    mem_param_tmp_101053.references = NULL;
    
    struct memblock mem_param_tmp_101052;
    
    mem_param_tmp_101052.references = NULL;
    
    struct memblock mem_param_tmp_101051;
    
    mem_param_tmp_101051.references = NULL;
    
    struct memblock mem_param_tmp_101050;
    
    mem_param_tmp_101050.references = NULL;
    
    struct memblock mem_param_tmp_101049;
    
    mem_param_tmp_101049.references = NULL;
    
    struct memblock mem_param_tmp_101048;
    
    mem_param_tmp_101048.references = NULL;
    
    struct memblock mem_param_tmp_101047;
    
    mem_param_tmp_101047.references = NULL;
    
    struct memblock mem_param_tmp_101046;
    
    mem_param_tmp_101046.references = NULL;
    
    struct memblock mem_param_tmp_101045;
    
    mem_param_tmp_101045.references = NULL;
    
    struct memblock mem_param_tmp_101044;
    
    mem_param_tmp_101044.references = NULL;
    
    struct memblock ext_mem_100834;
    
    ext_mem_100834.references = NULL;
    
    struct memblock ext_mem_100835;
    
    ext_mem_100835.references = NULL;
    
    struct memblock ext_mem_100836;
    
    ext_mem_100836.references = NULL;
    
    struct memblock mem_100832;
    
    mem_100832.references = NULL;
    
    struct memblock mem_100830;
    
    mem_100830.references = NULL;
    
    struct memblock mem_100828;
    
    mem_100828.references = NULL;
    
    struct memblock mem_100826;
    
    mem_100826.references = NULL;
    
    struct memblock ext_mem_100823;
    
    ext_mem_100823.references = NULL;
    
    struct memblock ext_mem_100824;
    
    ext_mem_100824.references = NULL;
    
    struct memblock ext_mem_100825;
    
    ext_mem_100825.references = NULL;
    
    struct memblock mem_100821;
    
    mem_100821.references = NULL;
    
    struct memblock mem_100819;
    
    mem_100819.references = NULL;
    
    struct memblock mem_100817;
    
    mem_100817.references = NULL;
    
    struct memblock mem_100815;
    
    mem_100815.references = NULL;
    
    struct memblock ext_mem_100812;
    
    ext_mem_100812.references = NULL;
    
    struct memblock ext_mem_100813;
    
    ext_mem_100813.references = NULL;
    
    struct memblock ext_mem_100814;
    
    ext_mem_100814.references = NULL;
    
    struct memblock mem_100810;
    
    mem_100810.references = NULL;
    
    struct memblock mem_100808;
    
    mem_100808.references = NULL;
    
    struct memblock mem_100806;
    
    mem_100806.references = NULL;
    
    struct memblock mem_100804;
    
    mem_100804.references = NULL;
    
    struct memblock ext_mem_100801;
    
    ext_mem_100801.references = NULL;
    
    struct memblock ext_mem_100802;
    
    ext_mem_100802.references = NULL;
    
    struct memblock ext_mem_100803;
    
    ext_mem_100803.references = NULL;
    
    struct memblock mem_100799;
    
    mem_100799.references = NULL;
    
    struct memblock mem_100797;
    
    mem_100797.references = NULL;
    
    struct memblock mem_100795;
    
    mem_100795.references = NULL;
    
    struct memblock mem_100793;
    
    mem_100793.references = NULL;
    
    struct memblock ext_mem_100790;
    
    ext_mem_100790.references = NULL;
    
    struct memblock ext_mem_100791;
    
    ext_mem_100791.references = NULL;
    
    struct memblock ext_mem_100792;
    
    ext_mem_100792.references = NULL;
    
    struct memblock mem_100788;
    
    mem_100788.references = NULL;
    
    struct memblock mem_100786;
    
    mem_100786.references = NULL;
    
    struct memblock mem_100784;
    
    mem_100784.references = NULL;
    
    struct memblock mem_100782;
    
    mem_100782.references = NULL;
    
    struct memblock ext_mem_100779;
    
    ext_mem_100779.references = NULL;
    
    struct memblock ext_mem_100780;
    
    ext_mem_100780.references = NULL;
    
    struct memblock ext_mem_100781;
    
    ext_mem_100781.references = NULL;
    
    struct memblock mem_100777;
    
    mem_100777.references = NULL;
    
    struct memblock mem_100775;
    
    mem_100775.references = NULL;
    
    struct memblock mem_100773;
    
    mem_100773.references = NULL;
    
    struct memblock mem_100771;
    
    mem_100771.references = NULL;
    
    struct memblock ext_mem_100768;
    
    ext_mem_100768.references = NULL;
    
    struct memblock ext_mem_100769;
    
    ext_mem_100769.references = NULL;
    
    struct memblock ext_mem_100770;
    
    ext_mem_100770.references = NULL;
    
    struct memblock mem_100766;
    
    mem_100766.references = NULL;
    
    struct memblock mem_100764;
    
    mem_100764.references = NULL;
    
    struct memblock mem_100762;
    
    mem_100762.references = NULL;
    
    struct memblock mem_100760;
    
    mem_100760.references = NULL;
    
    struct memblock ext_mem_100757;
    
    ext_mem_100757.references = NULL;
    
    struct memblock ext_mem_100758;
    
    ext_mem_100758.references = NULL;
    
    struct memblock ext_mem_100759;
    
    ext_mem_100759.references = NULL;
    
    struct memblock mem_100755;
    
    mem_100755.references = NULL;
    
    struct memblock mem_100753;
    
    mem_100753.references = NULL;
    
    struct memblock mem_100751;
    
    mem_100751.references = NULL;
    
    struct memblock mem_100749;
    
    mem_100749.references = NULL;
    
    struct memblock ext_mem_100746;
    
    ext_mem_100746.references = NULL;
    
    struct memblock ext_mem_100747;
    
    ext_mem_100747.references = NULL;
    
    struct memblock ext_mem_100748;
    
    ext_mem_100748.references = NULL;
    
    struct memblock mem_100744;
    
    mem_100744.references = NULL;
    
    struct memblock mem_100742;
    
    mem_100742.references = NULL;
    
    struct memblock mem_100740;
    
    mem_100740.references = NULL;
    
    struct memblock mem_100738;
    
    mem_100738.references = NULL;
    
    struct memblock mem_param_99190;
    
    mem_param_99190.references = NULL;
    
    struct memblock mem_param_99186;
    
    mem_param_99186.references = NULL;
    
    struct memblock mem_param_99182;
    
    mem_param_99182.references = NULL;
    
    struct memblock mem_param_99178;
    
    mem_param_99178.references = NULL;
    
    struct memblock mem_param_99174;
    
    mem_param_99174.references = NULL;
    
    struct memblock mem_param_99170;
    
    mem_param_99170.references = NULL;
    
    struct memblock mem_param_99166;
    
    mem_param_99166.references = NULL;
    
    struct memblock mem_param_99162;
    
    mem_param_99162.references = NULL;
    
    struct memblock mem_param_99158;
    
    mem_param_99158.references = NULL;
    
    struct memblock mem_param_99154;
    
    mem_param_99154.references = NULL;
    
    struct memblock mem_param_99150;
    
    mem_param_99150.references = NULL;
    
    struct memblock mem_param_99146;
    
    mem_param_99146.references = NULL;
    
    struct memblock mem_param_99142;
    
    mem_param_99142.references = NULL;
    
    struct memblock mem_param_99138;
    
    mem_param_99138.references = NULL;
    
    struct memblock mem_param_99134;
    
    mem_param_99134.references = NULL;
    
    struct memblock mem_param_99130;
    
    mem_param_99130.references = NULL;
    
    struct memblock mem_param_99126;
    
    mem_param_99126.references = NULL;
    
    struct memblock mem_param_99122;
    
    mem_param_99122.references = NULL;
    
    struct memblock mem_param_99118;
    
    mem_param_99118.references = NULL;
    
    struct memblock mem_param_99114;
    
    mem_param_99114.references = NULL;
    
    struct memblock mem_param_99110;
    
    mem_param_99110.references = NULL;
    
    struct memblock mem_param_99106;
    
    mem_param_99106.references = NULL;
    
    struct memblock mem_param_99102;
    
    mem_param_99102.references = NULL;
    
    struct memblock mem_param_99098;
    
    mem_param_99098.references = NULL;
    
    struct memblock mem_param_99094;
    
    mem_param_99094.references = NULL;
    
    struct memblock mem_param_99090;
    
    mem_param_99090.references = NULL;
    
    struct memblock mem_param_99086;
    
    mem_param_99086.references = NULL;
    
    struct memblock ext_mem_100918;
    
    ext_mem_100918.references = NULL;
    
    struct memblock ext_mem_100919;
    
    ext_mem_100919.references = NULL;
    
    struct memblock ext_mem_100920;
    
    ext_mem_100920.references = NULL;
    
    struct memblock ext_mem_100921;
    
    ext_mem_100921.references = NULL;
    
    struct memblock ext_mem_100922;
    
    ext_mem_100922.references = NULL;
    
    struct memblock ext_mem_100923;
    
    ext_mem_100923.references = NULL;
    
    struct memblock ext_mem_100924;
    
    ext_mem_100924.references = NULL;
    
    struct memblock ext_mem_100925;
    
    ext_mem_100925.references = NULL;
    
    struct memblock ext_mem_100926;
    
    ext_mem_100926.references = NULL;
    
    struct memblock ext_mem_100927;
    
    ext_mem_100927.references = NULL;
    
    struct memblock ext_mem_100928;
    
    ext_mem_100928.references = NULL;
    
    struct memblock ext_mem_100929;
    
    ext_mem_100929.references = NULL;
    
    struct memblock ext_mem_100930;
    
    ext_mem_100930.references = NULL;
    
    struct memblock ext_mem_100931;
    
    ext_mem_100931.references = NULL;
    
    struct memblock ext_mem_100932;
    
    ext_mem_100932.references = NULL;
    
    struct memblock ext_mem_100933;
    
    ext_mem_100933.references = NULL;
    
    struct memblock ext_mem_100934;
    
    ext_mem_100934.references = NULL;
    
    struct memblock ext_mem_100935;
    
    ext_mem_100935.references = NULL;
    
    struct memblock ext_mem_100936;
    
    ext_mem_100936.references = NULL;
    
    struct memblock ext_mem_100937;
    
    ext_mem_100937.references = NULL;
    
    struct memblock ext_mem_100938;
    
    ext_mem_100938.references = NULL;
    
    struct memblock ext_mem_100939;
    
    ext_mem_100939.references = NULL;
    
    struct memblock ext_mem_100940;
    
    ext_mem_100940.references = NULL;
    
    struct memblock ext_mem_100941;
    
    ext_mem_100941.references = NULL;
    
    struct memblock ext_mem_100942;
    
    ext_mem_100942.references = NULL;
    
    struct memblock ext_mem_100943;
    
    ext_mem_100943.references = NULL;
    
    struct memblock ext_mem_100944;
    
    ext_mem_100944.references = NULL;
    
    struct memblock mem_out_101043;
    
    mem_out_101043.references = NULL;
    
    struct memblock mem_out_101042;
    
    mem_out_101042.references = NULL;
    
    struct memblock mem_out_101041;
    
    mem_out_101041.references = NULL;
    
    struct memblock mem_out_101040;
    
    mem_out_101040.references = NULL;
    
    struct memblock mem_out_101039;
    
    mem_out_101039.references = NULL;
    
    struct memblock mem_out_101038;
    
    mem_out_101038.references = NULL;
    
    struct memblock mem_out_101037;
    
    mem_out_101037.references = NULL;
    
    struct memblock mem_out_101036;
    
    mem_out_101036.references = NULL;
    
    struct memblock mem_out_101035;
    
    mem_out_101035.references = NULL;
    
    struct memblock mem_out_101034;
    
    mem_out_101034.references = NULL;
    
    struct memblock mem_out_101033;
    
    mem_out_101033.references = NULL;
    
    struct memblock mem_out_101032;
    
    mem_out_101032.references = NULL;
    
    struct memblock mem_out_101031;
    
    mem_out_101031.references = NULL;
    
    struct memblock mem_out_101030;
    
    mem_out_101030.references = NULL;
    
    struct memblock mem_out_101029;
    
    mem_out_101029.references = NULL;
    
    struct memblock mem_out_101028;
    
    mem_out_101028.references = NULL;
    
    struct memblock mem_out_101027;
    
    mem_out_101027.references = NULL;
    
    struct memblock mem_out_101026;
    
    mem_out_101026.references = NULL;
    
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_99191_cached_sizze_101471 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99191, &mem_99191_cached_sizze_101471, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99192_cached_sizze_101472 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99192, &mem_99192_cached_sizze_101472, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99201_cached_sizze_101473 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99201, &mem_99201_cached_sizze_101473, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99208_cached_sizze_101474 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99208, &mem_99208_cached_sizze_101474, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99223_cached_sizze_101475 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99223, &mem_99223_cached_sizze_101475, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99224_cached_sizze_101476 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99224, &mem_99224_cached_sizze_101476, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99233_cached_sizze_101477 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99233, &mem_99233_cached_sizze_101477, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99240_cached_sizze_101478 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99240, &mem_99240_cached_sizze_101478, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99255_cached_sizze_101479 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99255, &mem_99255_cached_sizze_101479, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99256_cached_sizze_101480 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99256, &mem_99256_cached_sizze_101480, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99265_cached_sizze_101481 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99265, &mem_99265_cached_sizze_101481, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99266_cached_sizze_101482 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99266, &mem_99266_cached_sizze_101482, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99287_cached_sizze_101483 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99287, &mem_99287_cached_sizze_101483, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99288_cached_sizze_101484 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99288, &mem_99288_cached_sizze_101484, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99289_cached_sizze_101485 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99289, &mem_99289_cached_sizze_101485, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99301_cached_sizze_101486 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99301, &mem_99301_cached_sizze_101486, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99302_cached_sizze_101487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99302, &mem_99302_cached_sizze_101487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99326_cached_sizze_101488 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99326, &mem_99326_cached_sizze_101488, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99327_cached_sizze_101489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99327, &mem_99327_cached_sizze_101489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99328_cached_sizze_101490 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99328, &mem_99328_cached_sizze_101490, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99329_cached_sizze_101491 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99329, &mem_99329_cached_sizze_101491, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99330_cached_sizze_101492 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99330, &mem_99330_cached_sizze_101492, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99349_cached_sizze_101493 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99349, &mem_99349_cached_sizze_101493, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99350_cached_sizze_101494 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99350, &mem_99350_cached_sizze_101494, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99351_cached_sizze_101495 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99351, &mem_99351_cached_sizze_101495, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99388_cached_sizze_101496 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99388, &mem_99388_cached_sizze_101496, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99389_cached_sizze_101497 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99389, &mem_99389_cached_sizze_101497, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99390_cached_sizze_101498 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99390, &mem_99390_cached_sizze_101498, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99406_cached_sizze_101499 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99406, &mem_99406_cached_sizze_101499, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99407_cached_sizze_101500 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99407, &mem_99407_cached_sizze_101500, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99408_cached_sizze_101501 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99408, &mem_99408_cached_sizze_101501, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99421_cached_sizze_101502 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99421, &mem_99421_cached_sizze_101502, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99422_cached_sizze_101503 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99422, &mem_99422_cached_sizze_101503, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99423_cached_sizze_101504 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99423, &mem_99423_cached_sizze_101504, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99469_cached_sizze_101505 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99469, &mem_99469_cached_sizze_101505, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99470_cached_sizze_101506 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99470, &mem_99470_cached_sizze_101506, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99481_cached_sizze_101507 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99481, &mem_99481_cached_sizze_101507, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99482_cached_sizze_101508 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99482, &mem_99482_cached_sizze_101508, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99491_cached_sizze_101509 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99491, &mem_99491_cached_sizze_101509, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99492_cached_sizze_101510 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99492, &mem_99492_cached_sizze_101510, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99513_cached_sizze_101511 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99513, &mem_99513_cached_sizze_101511, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99518_cached_sizze_101512 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99518, &mem_99518_cached_sizze_101512, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99525_cached_sizze_101513 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_99525, &mem_99525_cached_sizze_101513, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99546_cached_sizze_101514 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99546, &mem_99546_cached_sizze_101514, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99547_cached_sizze_101515 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99547, &mem_99547_cached_sizze_101515, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99555_cached_sizze_101516 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99555, &mem_99555_cached_sizze_101516, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99569_cached_sizze_101517 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99569, &mem_99569_cached_sizze_101517, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99574_cached_sizze_101518 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99574, &mem_99574_cached_sizze_101518, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99585_cached_sizze_101519 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99585, &mem_99585_cached_sizze_101519, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99590_cached_sizze_101520 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99590, &mem_99590_cached_sizze_101520, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99601_cached_sizze_101521 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99601, &mem_99601_cached_sizze_101521, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99602_cached_sizze_101522 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99602, &mem_99602_cached_sizze_101522, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99611_cached_sizze_101523 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99611, &mem_99611_cached_sizze_101523, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99612_cached_sizze_101524 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99612, &mem_99612_cached_sizze_101524, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99633_cached_sizze_101525 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99633, &mem_99633_cached_sizze_101525, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99634_cached_sizze_101526 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99634, &mem_99634_cached_sizze_101526, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99642_cached_sizze_101527 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99642, &mem_99642_cached_sizze_101527, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99656_cached_sizze_101528 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99656, &mem_99656_cached_sizze_101528, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99657_cached_sizze_101529 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99657, &mem_99657_cached_sizze_101529, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99665_cached_sizze_101530 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99665, &mem_99665_cached_sizze_101530, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99679_cached_sizze_101531 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99679, &mem_99679_cached_sizze_101531, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99684_cached_sizze_101532 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99684, &mem_99684_cached_sizze_101532, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99695_cached_sizze_101533 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99695, &mem_99695_cached_sizze_101533, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99700_cached_sizze_101534 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99700, &mem_99700_cached_sizze_101534, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99711_cached_sizze_101535 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99711, &mem_99711_cached_sizze_101535, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99716_cached_sizze_101536 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99716, &mem_99716_cached_sizze_101536, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99727_cached_sizze_101537 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99727, &mem_99727_cached_sizze_101537, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99734_cached_sizze_101538 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99734, &mem_99734_cached_sizze_101538, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99739_cached_sizze_101539 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99739, &mem_99739_cached_sizze_101539, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99750_cached_sizze_101540 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99750, &mem_99750_cached_sizze_101540, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99755_cached_sizze_101541 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99755, &mem_99755_cached_sizze_101541, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99766_cached_sizze_101542 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99766, &mem_99766_cached_sizze_101542, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99773_cached_sizze_101543 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99773, &mem_99773_cached_sizze_101543, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99777_cached_sizze_101544 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99777, &mem_99777_cached_sizze_101544, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99787_cached_sizze_101545 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99787, &mem_99787_cached_sizze_101545, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99792_cached_sizze_101546 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99792, &mem_99792_cached_sizze_101546, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99799_cached_sizze_101547 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99799, &mem_99799_cached_sizze_101547, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99810_cached_sizze_101548 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99810, &mem_99810_cached_sizze_101548, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99815_cached_sizze_101549 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99815, &mem_99815_cached_sizze_101549, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99826_cached_sizze_101550 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99826, &mem_99826_cached_sizze_101550, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99833_cached_sizze_101551 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99833, &mem_99833_cached_sizze_101551, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99838_cached_sizze_101552 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99838, &mem_99838_cached_sizze_101552, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99849_cached_sizze_101553 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_99849, &mem_99849_cached_sizze_101553, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99854_cached_sizze_101554 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_99854, &mem_99854_cached_sizze_101554, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99865_cached_sizze_101555 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99865, &mem_99865_cached_sizze_101555, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99870_cached_sizze_101556 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99870, &mem_99870_cached_sizze_101556, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99881_cached_sizze_101557 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99881, &mem_99881_cached_sizze_101557, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99882_cached_sizze_101558 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99882, &mem_99882_cached_sizze_101558, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99891_cached_sizze_101559 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99891, &mem_99891_cached_sizze_101559, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99892_cached_sizze_101560 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99892, &mem_99892_cached_sizze_101560, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99913_cached_sizze_101561 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_99913, &mem_99913_cached_sizze_101561, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99918_cached_sizze_101562 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_99918, &mem_99918_cached_sizze_101562, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99929_cached_sizze_101563 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99929, &mem_99929_cached_sizze_101563, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99934_cached_sizze_101564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99934, &mem_99934_cached_sizze_101564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99945_cached_sizze_101565 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99945, &mem_99945_cached_sizze_101565, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99952_cached_sizze_101566 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99952, &mem_99952_cached_sizze_101566, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99959_cached_sizze_101567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99959, &mem_99959_cached_sizze_101567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99969_cached_sizze_101568 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99969, &mem_99969_cached_sizze_101568, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99974_cached_sizze_101569 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99974, &mem_99974_cached_sizze_101569, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99985_cached_sizze_101570 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99985, &mem_99985_cached_sizze_101570, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99986_cached_sizze_101571 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_99986, &mem_99986_cached_sizze_101571, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99995_cached_sizze_101572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99995, &mem_99995_cached_sizze_101572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_99996_cached_sizze_101573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_99996, &mem_99996_cached_sizze_101573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100017_cached_sizze_101574 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100017, &mem_100017_cached_sizze_101574, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100018_cached_sizze_101575 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100018, &mem_100018_cached_sizze_101575, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100029_cached_sizze_101576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100029, &mem_100029_cached_sizze_101576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100030_cached_sizze_101577 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100030, &mem_100030_cached_sizze_101577, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100039_cached_sizze_101578 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_100039, &mem_100039_cached_sizze_101578, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100046_cached_sizze_101579 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100046, &mem_100046_cached_sizze_101579, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100071_cached_sizze_101580 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100071, &mem_100071_cached_sizze_101580, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100072_cached_sizze_101581 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100072, &mem_100072_cached_sizze_101581, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100073_cached_sizze_101582 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100073, &mem_100073_cached_sizze_101582, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100088_cached_sizze_101583 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100088, &mem_100088_cached_sizze_101583, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100089_cached_sizze_101584 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100089, &mem_100089_cached_sizze_101584, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100090_cached_sizze_101585 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100090, &mem_100090_cached_sizze_101585, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:5-117:48
    if (mem_100102_cached_sizze_101586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100102, &mem_100102_cached_sizze_101586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100109_cached_sizze_101587 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100109, &mem_100109_cached_sizze_101587, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100116_cached_sizze_101588 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100116, &mem_100116_cached_sizze_101588, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100148_cached_sizze_101589 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100148, &mem_100148_cached_sizze_101589, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100149_cached_sizze_101590 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100149, &mem_100149_cached_sizze_101590, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100160_cached_sizze_101591 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100160, &mem_100160_cached_sizze_101591, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100161_cached_sizze_101592 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100161, &mem_100161_cached_sizze_101592, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100170_cached_sizze_101593 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100170, &mem_100170_cached_sizze_101593, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100177_cached_sizze_101594 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_100177, &mem_100177_cached_sizze_101594, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100202_cached_sizze_101595 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100202, &mem_100202_cached_sizze_101595, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100208_cached_sizze_101596 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100208, &mem_100208_cached_sizze_101596, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100213_cached_sizze_101597 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100213, &mem_100213_cached_sizze_101597, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100229_cached_sizze_101598 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100229, &mem_100229_cached_sizze_101598, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100234_cached_sizze_101599 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100234, &mem_100234_cached_sizze_101599, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100245_cached_sizze_101600 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100245, &mem_100245_cached_sizze_101600, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100250_cached_sizze_101601 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100250, &mem_100250_cached_sizze_101601, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100261_cached_sizze_101602 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100261, &mem_100261_cached_sizze_101602, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100267_cached_sizze_101603 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100267, &mem_100267_cached_sizze_101603, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100272_cached_sizze_101604 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100272, &mem_100272_cached_sizze_101604, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100288_cached_sizze_101605 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100288, &mem_100288_cached_sizze_101605, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100294_cached_sizze_101606 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100294, &mem_100294_cached_sizze_101606, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100299_cached_sizze_101607 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100299, &mem_100299_cached_sizze_101607, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100315_cached_sizze_101608 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100315, &mem_100315_cached_sizze_101608, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100320_cached_sizze_101609 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100320, &mem_100320_cached_sizze_101609, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100331_cached_sizze_101610 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100331, &mem_100331_cached_sizze_101610, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100337_cached_sizze_101611 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100337, &mem_100337_cached_sizze_101611, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100342_cached_sizze_101612 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100342, &mem_100342_cached_sizze_101612, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100358_cached_sizze_101613 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100358, &mem_100358_cached_sizze_101613, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100364_cached_sizze_101614 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100364, &mem_100364_cached_sizze_101614, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100369_cached_sizze_101615 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100369, &mem_100369_cached_sizze_101615, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100385_cached_sizze_101616 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100385, &mem_100385_cached_sizze_101616, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100391_cached_sizze_101617 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100391, &mem_100391_cached_sizze_101617, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100396_cached_sizze_101618 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100396, &mem_100396_cached_sizze_101618, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100412_cached_sizze_101619 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100412, &mem_100412_cached_sizze_101619, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100413_cached_sizze_101620 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100413, &mem_100413_cached_sizze_101620, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100424_cached_sizze_101621 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100424, &mem_100424_cached_sizze_101621, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100425_cached_sizze_101622 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_100425, &mem_100425_cached_sizze_101622, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100434_cached_sizze_101623 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_100434, &mem_100434_cached_sizze_101623, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100435_cached_sizze_101624 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_100435, &mem_100435_cached_sizze_101624, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100466_cached_sizze_101625 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100466, &mem_100466_cached_sizze_101625, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100467_cached_sizze_101626 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100467, &mem_100467_cached_sizze_101626, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100468_cached_sizze_101627 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100468, &mem_100468_cached_sizze_101627, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100481_cached_sizze_101628 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100481, &mem_100481_cached_sizze_101628, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100482_cached_sizze_101629 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100482, &mem_100482_cached_sizze_101629, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100483_cached_sizze_101630 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100483, &mem_100483_cached_sizze_101630, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100514_cached_sizze_101631 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100514, &mem_100514_cached_sizze_101631, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100515_cached_sizze_101632 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100515, &mem_100515_cached_sizze_101632, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100516_cached_sizze_101633 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100516, &mem_100516_cached_sizze_101633, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100517_cached_sizze_101634 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100517, &mem_100517_cached_sizze_101634, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100534_cached_sizze_101635 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100534, &mem_100534_cached_sizze_101635, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100535_cached_sizze_101636 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100535, &mem_100535_cached_sizze_101636, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100536_cached_sizze_101637 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100536, &mem_100536_cached_sizze_101637, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100537_cached_sizze_101638 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100537, &mem_100537_cached_sizze_101638, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100578_cached_sizze_101639 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100578, &mem_100578_cached_sizze_101639, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100585_cached_sizze_101640 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100585, &mem_100585_cached_sizze_101640, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100592_cached_sizze_101641 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100592, &mem_100592_cached_sizze_101641, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100602_cached_sizze_101642 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100602, &mem_100602_cached_sizze_101642, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100607_cached_sizze_101643 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100607, &mem_100607_cached_sizze_101643, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100618_cached_sizze_101644 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100618, &mem_100618_cached_sizze_101644, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100625_cached_sizze_101645 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100625, &mem_100625_cached_sizze_101645, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100632_cached_sizze_101646 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100632, &mem_100632_cached_sizze_101646, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100642_cached_sizze_101647 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100642, &mem_100642_cached_sizze_101647, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100647_cached_sizze_101648 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100647, &mem_100647_cached_sizze_101648, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100658_cached_sizze_101649 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100658, &mem_100658_cached_sizze_101649, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100659_cached_sizze_101650 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_100659, &mem_100659_cached_sizze_101650, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100668_cached_sizze_101651 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100668, &mem_100668_cached_sizze_101651, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100669_cached_sizze_101652 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100669, &mem_100669_cached_sizze_101652, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100690_cached_sizze_101653 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_100690, &mem_100690_cached_sizze_101653, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100695_cached_sizze_101654 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100695, &mem_100695_cached_sizze_101654, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100706_cached_sizze_101655 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_100706, &mem_100706_cached_sizze_101655, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100707_cached_sizze_101656 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_100707, &mem_100707_cached_sizze_101656, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100716_cached_sizze_101657 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100716, &mem_100716_cached_sizze_101657, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_100717_cached_sizze_101658 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_100717, &mem_100717_cached_sizze_101658, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:475:5-480:51
    if (memblock_set(ctx, &mem_param_99086, &wdown_mem_99053, "wdown_mem_99053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99090, &wkey_mem_99054, "wkey_mem_99054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99094, &wout_mem_99055, "wout_mem_99055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99098, &wpe_mem_99056, "wpe_mem_99056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99102, &wqry_mem_99057, "wqry_mem_99057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99106, &wte_mem_99058, "wte_mem_99058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99110, &wup_mem_99059, "wup_mem_99059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99114, &wval_mem_99060, "wval_mem_99060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99118, &wvoc_mem_99061, "wvoc_mem_99061") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99122, &wdown_mem_99062, "wdown_mem_99062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99126, &wkey_mem_99063, "wkey_mem_99063") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99130, &wout_mem_99064, "wout_mem_99064") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99134, &wpe_mem_99065, "wpe_mem_99065") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99138, &wqry_mem_99066, "wqry_mem_99066") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99142, &wte_mem_99067, "wte_mem_99067") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99146, &wup_mem_99068, "wup_mem_99068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99150, &wval_mem_99069, "wval_mem_99069") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99154, &wvoc_mem_99070, "wvoc_mem_99070") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99158, &wdown_mem_99071, "wdown_mem_99071") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99162, &wkey_mem_99072, "wkey_mem_99072") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99166, &wout_mem_99073, "wout_mem_99073") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99170, &wpe_mem_99074, "wpe_mem_99074") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99174, &wqry_mem_99075, "wqry_mem_99075") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99178, &wte_mem_99076, "wte_mem_99076") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99182, &wup_mem_99077, "wup_mem_99077") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99186, &wval_mem_99078, "wval_mem_99078") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_99190, &wvoc_mem_99079, "wvoc_mem_99079") != 0)
        return 1;
    for (int64_t step_89818 = 0; step_89818 < (int64_t) 500; step_89818++) {
        // futhark/microgpt.fut:477:16-25
        
        int64_t dl_89846 = ((int64_t *) dls_mem_99081.mem)[step_89818];
        
        // futhark/microgpt.fut:390:37-40
        
        int64_t zl_rhs_89851 = sub64(dl_89846, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98062 = 0; i_98062 < (int64_t) 16; i_98062++) {
            // futhark/microgpt.fut:390:25-81
            
            bool cond_93720 = slt64(i_98062, zl_rhs_89851);
            
            // futhark/microgpt.fut:390:56-59
            
            int64_t zeze_lhs_93721 = add64((int64_t) 1, i_98062);
            
            // futhark/microgpt.fut:390:47-60
            
            bool x_93722 = sle64((int64_t) 0, zeze_lhs_93721);
            
            // futhark/microgpt.fut:390:47-60
            
            bool y_93723 = slt64(zeze_lhs_93721, (int64_t) 16);
            
            // futhark/microgpt.fut:390:47-60
            
            bool bounds_check_93724 = x_93722 && y_93723;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_93725 = !cond_93720;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_93726 = bounds_check_93724 || loop_not_taken_93725;
            
            // futhark/microgpt.fut:390:47-60
            
            bool index_certs_93727;
            
            if (!protect_assert_disj_93726) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_93721, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:390:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:390:3-83\n   #6  futhark/microgpt.fut:448:18-38\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_93742 = ((int64_t *) seqs_mem_99082.mem)[step_89818 * (int64_t) 16 + i_98062];
            
            // futhark/microgpt.fut:450:37-51
            
            bool x_93743 = sle64((int64_t) 0, tmp_93742);
            
            // futhark/microgpt.fut:450:37-51
            
            bool y_93744 = slt64(tmp_93742, (int64_t) 27);
            
            // futhark/microgpt.fut:450:37-51
            
            bool bounds_check_93745 = x_93743 && y_93744;
            
            // futhark/microgpt.fut:450:37-51
            
            bool index_certs_93746;
            
            if (!bounds_check_93745) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93742, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:450:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:450:16-55\n   #6  futhark/microgpt.fut:458:26-464:31\n   #7  futhark/microgpt.fut:480:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:390:47-60
            
            int64_t zeze_lhs_93728;
            
            if (cond_93720) {
                int64_t x_97825 = ((int64_t *) seqs_mem_99082.mem)[step_89818 * (int64_t) 16 + zeze_lhs_93721];
                
                zeze_lhs_93728 = x_97825;
            } else {
                zeze_lhs_93728 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98052 = 0; i_98052 < (int64_t) 27; i_98052++) {
                // futhark/microgpt.fut:390:61-65
                
                bool cond_t_res_93732 = zeze_lhs_93728 == i_98052;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_93733 = cond_93720 && cond_t_res_93732;
                
                // futhark/microgpt.fut:390:25-81
                
                double lifted_lambda_res_93734;
                
                if (x_93733) {
                    lifted_lambda_res_93734 = 1.0;
                } else {
                    lifted_lambda_res_93734 = 0.0;
                }
                ((double *) mem_99201)[i_98052] = lifted_lambda_res_93734;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98056 = 0; i_98056 < (int64_t) 16; i_98056++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93753 = ((double *) mem_param_99106.mem)[tmp_93742 * (int64_t) 16 + i_98056];
                
                ((double *) mem_99208)[i_98056] = lifted_lambda_res_93753;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99191, i_98062 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99192, i_98062 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99201, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98077 = 0; i_98077 < (int64_t) 16; i_98077++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98067 = 0; i_98067 < (int64_t) 16; i_98067++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_93778 = ((double *) mem_param_99098.mem)[i_98077 * (int64_t) 16 + i_98067];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_93779 = ((double *) mem_99191)[i_98077 * (int64_t) 16 + i_98067];
                
                // futhark/microgpt.fut:240:39-75
                
                double zp_res_93780 = zp_lhs_93778 + zp_rhs_93779;
                
                ((double *) mem_99233)[i_98067] = zp_res_93780;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98071 = 0; i_98071 < (int64_t) 27; i_98071++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_93794 = ((double *) mem_99192)[i_98077 * (int64_t) 27 + i_98071];
                
                // futhark/microgpt.fut:264:43-85
                
                double zt_res_93795 = -6.25e-2 * zt_rhs_93794;
                
                ((double *) mem_99240)[i_98071] = zt_res_93795;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99223, i_98077 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99240, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99224, i_98077 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99233, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98091 = 0; i_98091 < (int64_t) 16; i_98091++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93814;
            double r_93816 = 0.0;
            
            for (int64_t i_93815 = 0; i_93815 < (int64_t) 16; i_93815++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93817 = ((double *) mem_99224)[i_98091 * (int64_t) 16 + i_93815];
                
                // futhark/microgpt.fut:241:79-112
                
                double zt_res_93818 = zt_lhs_93817 * zt_lhs_93817;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93819 = r_93816 + zt_res_93818;
                double r_tmp_101108 = zp_res_93819;
                
                r_93816 = r_tmp_101108;
            }
            defunc_0_lifted_lambda_res_93814 = r_93816;
            // futhark/microgpt.fut:241:59-130
            
            double zs_res_93820 = defunc_0_lifted_lambda_res_93814 / 16.0;
            
            // futhark/microgpt.fut:241:116-159
            
            double zp_res_93821 = 1.0e-5 + zs_res_93820;
            
            // futhark/microgpt.fut:241:49-159
            
            double sqrt_res_93822 = futrts_sqrt64(zp_res_93821);
            
            // futhark/microgpt.fut:241:40-161
            
            double zs_res_93823 = 1.0 / sqrt_res_93822;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98084 = 0; i_98084 < (int64_t) 16; i_98084++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_95830 = ((double *) mem_99224)[i_98091 * (int64_t) 16 + i_98084];
                
                // futhark/microgpt.fut:241:44-182
                
                double zt_res_95831 = zs_res_93823 * zt_rhs_95830;
                
                // futhark/microgpt.fut:320:45-86
                
                double zt_res_95839 = zt_rhs_95830 * zt_rhs_95830;
                
                ((double *) mem_99265)[i_98084] = zt_res_95839;
                ((double *) mem_99266)[i_98084] = zt_res_95831;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99255, i_98091 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99265, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99256, i_98091 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99266, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98107 = 0; i_98107 < (int64_t) 16; i_98107++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93922;
            double r_93924 = 0.0;
            
            for (int64_t i_93923 = 0; i_93923 < (int64_t) 16; i_93923++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_93925 = ((double *) mem_99256)[i_98107 * (int64_t) 16 + i_93923];
                
                // futhark/microgpt.fut:242:79-112
                
                double zt_res_93926 = zt_lhs_93925 * zt_lhs_93925;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93927 = r_93924 + zt_res_93926;
                double r_tmp_101114 = zp_res_93927;
                
                r_93924 = r_tmp_101114;
            }
            defunc_0_lifted_lambda_res_93922 = r_93924;
            // futhark/microgpt.fut:242:59-130
            
            double zs_res_93928 = defunc_0_lifted_lambda_res_93922 / 16.0;
            
            // futhark/microgpt.fut:242:116-159
            
            double zp_res_93929 = 1.0e-5 + zs_res_93928;
            
            // futhark/microgpt.fut:242:49-159
            
            double sqrt_res_93930 = futrts_sqrt64(zp_res_93929);
            
            // futhark/microgpt.fut:242:40-161
            
            double zs_res_93931 = 1.0 / sqrt_res_93930;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98098 = 0; i_98098 < (int64_t) 16; i_98098++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_95859 = ((double *) mem_99256)[i_98107 * (int64_t) 16 + i_98098];
                
                // futhark/microgpt.fut:242:44-182
                
                double zt_res_95860 = zs_res_93931 * zt_rhs_95859;
                
                // futhark/microgpt.fut:313:45-86
                
                double zt_res_95868 = zt_rhs_95859 * zt_rhs_95859;
                
                ((double *) mem_99301)[i_98098] = zt_res_95868;
                ((double *) mem_99302)[i_98098] = zt_res_95860;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_93965;
            double r_93967 = 0.0;
            
            for (int64_t i_93966 = 0; i_93966 < (int64_t) 16; i_93966++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_93968 = ((double *) mem_99255)[i_98107 * (int64_t) 16 + i_93966];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_93969 = r_93967 + lifted_lambda_res_93968;
                double r_tmp_101117 = zp_res_93969;
                
                r_93967 = r_tmp_101117;
            }
            defunc_0_lifted_lambda_res_93965 = r_93967;
            // futhark/microgpt.fut:321:36-94
            
            double zs_res_93970 = defunc_0_lifted_lambda_res_93965 / 16.0;
            
            ((double *) mem_99287)[i_98107] = zs_res_93970;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99288, i_98107 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99301, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99289, i_98107 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99302, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98131 = 0; i_98131 < (int64_t) 16; i_98131++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98117 = 0; i_98117 < (int64_t) 16; i_98117++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_95931;
                double r_95933 = 0.0;
                
                for (int64_t i_95932 = 0; i_95932 < (int64_t) 16; i_95932++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_95934 = ((double *) mem_param_99102.mem)[i_98117 * (int64_t) 16 + i_95932];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_95935 = ((double *) mem_99289)[i_98131 * (int64_t) 16 + i_95932];
                    
                    // futhark/microgpt.fut:243:60-97
                    
                    double zt_res_95936 = zt_lhs_95934 * zt_rhs_95935;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_95937 = r_95933 + zt_res_95936;
                    double r_tmp_101126 = zp_res_95937;
                    
                    r_95933 = r_tmp_101126;
                }
                defunc_0_lifted_lambda_res_95931 = r_95933;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_95944;
                double r_95946 = 0.0;
                
                for (int64_t i_95945 = 0; i_95945 < (int64_t) 16; i_95945++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_95947 = ((double *) mem_param_99090.mem)[i_98117 * (int64_t) 16 + i_95945];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_95948 = ((double *) mem_99289)[i_98131 * (int64_t) 16 + i_95945];
                    
                    // futhark/microgpt.fut:244:63-102
                    
                    double zt_res_95949 = zt_lhs_95947 * zt_rhs_95948;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_95950 = r_95946 + zt_res_95949;
                    double r_tmp_101127 = zp_res_95950;
                    
                    r_95946 = r_tmp_101127;
                }
                defunc_0_lifted_lambda_res_95944 = r_95946;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_95960;
                double r_95962 = 0.0;
                
                for (int64_t i_95961 = 0; i_95961 < (int64_t) 16; i_95961++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_95963 = ((double *) mem_param_99114.mem)[i_98117 * (int64_t) 16 + i_95961];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_95964 = ((double *) mem_99289)[i_98131 * (int64_t) 16 + i_95961];
                    
                    // futhark/microgpt.fut:245:63-102
                    
                    double zt_res_95965 = zt_lhs_95963 * zt_rhs_95964;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_95966 = r_95962 + zt_res_95965;
                    double r_tmp_101128 = zp_res_95966;
                    
                    r_95962 = r_tmp_101128;
                }
                defunc_0_lifted_lambda_res_95960 = r_95962;
                ((double *) mem_99349)[i_98117] = defunc_0_lifted_lambda_res_95960;
                ((double *) mem_99350)[i_98117] = defunc_0_lifted_lambda_res_95944;
                ((double *) mem_99351)[i_98117] = defunc_0_lifted_lambda_res_95931;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94312;
            double r_94314 = 0.0;
            
            for (int64_t i_94313 = 0; i_94313 < (int64_t) 16; i_94313++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_94315 = ((double *) mem_99288)[i_98131 * (int64_t) 16 + i_94313];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94316 = r_94314 + lifted_lambda_res_94315;
                double r_tmp_101129 = zp_res_94316;
                
                r_94314 = r_tmp_101129;
            }
            defunc_0_lifted_lambda_res_94312 = r_94314;
            // futhark/microgpt.fut:314:36-94
            
            double zs_res_94317 = defunc_0_lifted_lambda_res_94312 / 16.0;
            
            // futhark/microgpt.fut:322:43-55
            
            double zp_lhs_94331 = ((double *) mem_99287)[i_98131];
            
            // futhark/microgpt.fut:322:43-83
            
            double zp_res_94332 = 1.0e-5 + zp_lhs_94331;
            
            // futhark/microgpt.fut:322:35-83
            
            double sqrt_res_94333 = futrts_sqrt64(zp_res_94332);
            
            ((double *) mem_99326)[i_98131] = sqrt_res_94333;
            ((double *) mem_99327)[i_98131] = zs_res_94317;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99328, i_98131 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99349, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99329, i_98131 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99350, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99330, i_98131 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99351, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98163 = 0; i_98163 < (int64_t) 4; i_98163++) {
            // futhark/microgpt.fut:246:66-69
            
            int64_t zp_lhs_94405 = mul64((int64_t) 4, i_98163);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98153 = 0; i_98153 < (int64_t) 16; i_98153++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98143 = 0; i_98143 < (int64_t) 4; i_98143++) {
                    // futhark/microgpt.fut:246:71-78
                    
                    int64_t tmp_96124 = add64(zp_lhs_94405, i_98143);
                    
                    // futhark/microgpt.fut:246:48-80
                    
                    bool x_96125 = sle64((int64_t) 0, tmp_96124);
                    
                    // futhark/microgpt.fut:246:48-80
                    
                    bool y_96126 = slt64(tmp_96124, (int64_t) 16);
                    
                    // futhark/microgpt.fut:246:48-80
                    
                    bool bounds_check_96127 = x_96125 && y_96126;
                    
                    // futhark/microgpt.fut:246:48-80
                    
                    bool index_certs_96128;
                    
                    if (!bounds_check_96127) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_96124, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:246:48-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:246:12-81\n   #9  futhark/microgpt.fut:453:5-76\n   #10 futhark/microgpt.fut:458:26-464:31\n   #11 futhark/microgpt.fut:480:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96129 = ((double *) mem_99330)[i_98153 * (int64_t) 16 + tmp_96124];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96137 = ((double *) mem_99329)[i_98153 * (int64_t) 16 + tmp_96124];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96148 = ((double *) mem_99328)[i_98153 * (int64_t) 16 + tmp_96124];
                    
                    ((double *) mem_99421)[i_98143] = lifted_lambda_res_96148;
                    ((double *) mem_99422)[i_98143] = lifted_lambda_res_96137;
                    ((double *) mem_99423)[i_98143] = lifted_lambda_res_96129;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99406, i_98153 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99421, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99407, i_98153 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99422, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99408, i_98153 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99423, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_99388, i_98163 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99406, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_99389, i_98163 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99407, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_99390, i_98163 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99408, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98200 = 0; i_98200 < (int64_t) 4; i_98200++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98178 = 0; i_98178 < (int64_t) 16; i_98178++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98171 = 0; i_98171 < (int64_t) 16; i_98171++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_96230;
                    double r_96232 = 0.0;
                    
                    for (int64_t i_96231 = 0; i_96231 < (int64_t) 4; i_96231++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_96233 = ((double *) mem_99390)[i_98200 * (int64_t) 64 + i_98178 * (int64_t) 4 + i_96231];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_96234 = ((double *) mem_99389)[i_98200 * (int64_t) 64 + i_98171 * (int64_t) 4 + i_96231];
                        
                        // futhark/microgpt.fut:249:112-165
                        
                        double zt_res_96235 = zt_lhs_96233 * zt_rhs_96234;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_96236 = r_96232 + zt_res_96235;
                        double r_tmp_101145 = zp_res_96236;
                        
                        r_96232 = r_tmp_101145;
                    }
                    defunc_0_lifted_lambda_res_96230 = r_96232;
                    // futhark/microgpt.fut:249:92-182
                    
                    double zs_res_96237 = defunc_0_lifted_lambda_res_96230 / 2.0;
                    double zp_rhs_96238 = ((double *) masks_mem_99080.mem)[step_89818 * (int64_t) 256 + i_98178 * (int64_t) 16 + i_98171];
                    
                    // futhark/microgpt.fut:249:169-206
                    
                    double zp_res_96239 = zs_res_96237 + zp_rhs_96238;
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_96246;
                    double r_96248 = 0.0;
                    
                    for (int64_t i_96247 = 0; i_96247 < (int64_t) 4; i_96247++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_96249 = ((double *) mem_99390)[i_98200 * (int64_t) 64 + i_98178 * (int64_t) 4 + i_96247];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_96250 = ((double *) mem_99389)[i_98200 * (int64_t) 64 + i_98171 * (int64_t) 4 + i_96247];
                        
                        // futhark/microgpt.fut:290:75-134
                        
                        double zt_res_96251 = zt_lhs_96249 * zt_rhs_96250;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_96252 = r_96248 + zt_res_96251;
                        double r_tmp_101146 = zp_res_96252;
                        
                        r_96248 = r_tmp_101146;
                    }
                    defunc_0_lifted_lambda_res_96246 = r_96248;
                    ((double *) mem_99491)[i_98171] = defunc_0_lifted_lambda_res_96246;
                    ((double *) mem_99492)[i_98171] = zp_res_96239;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99481, i_98178 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99491, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99482, i_98178 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99492, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98194 = 0; i_98194 < (int64_t) 16; i_98194++) {
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_97845;
                int64_t defunc_0_reduce_res_97846;
                double redout_98181;
                int64_t redout_98182;
                
                redout_98181 = -INFINITY;
                redout_98182 = (int64_t) 16;
                for (int64_t i_98183 = 0; i_98183 < (int64_t) 16; i_98183++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96272 = ((double *) mem_99482)[i_98194 * (int64_t) 16 + i_98183];
                    
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_94527 = lifted_lambda_res_96272 < redout_98181;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_94528;
                    
                    if (zg_res_94527) {
                        lifted_lambda_res_94528 = redout_98181;
                    } else {
                        lifted_lambda_res_94528 = lifted_lambda_res_96272;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_94529;
                    
                    if (zg_res_94527) {
                        lifted_lambda_res_94529 = redout_98182;
                    } else {
                        lifted_lambda_res_94529 = i_98183;
                    }
                    
                    double redout_tmp_101148 = lifted_lambda_res_94528;
                    int64_t redout_tmp_101149 = lifted_lambda_res_94529;
                    
                    redout_98181 = redout_tmp_101148;
                    redout_98182 = redout_tmp_101149;
                }
                defunc_0_reduce_res_97845 = redout_98181;
                defunc_0_reduce_res_97846 = redout_98182;
                // futhark/microgpt.fut:250:73-129
                
                bool x_94530 = sle64((int64_t) 0, defunc_0_reduce_res_97846);
                
                // futhark/microgpt.fut:250:73-129
                
                bool y_94531 = slt64(defunc_0_reduce_res_97846, (int64_t) 16);
                
                // futhark/microgpt.fut:250:73-129
                
                bool bounds_check_94532 = x_94530 && y_94531;
                
                // futhark/microgpt.fut:250:73-129
                
                bool index_certs_94533;
                
                if (!bounds_check_94532) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_97846, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:250:73-129\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:250:16-133\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:15:29-44\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:15:15-45\n   #11 futhark/microgpt.fut:249:12-251:121\n   #12 futhark/microgpt.fut:453:5-76\n   #13 futhark/microgpt.fut:458:26-464:31\n   #14 futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_94534 = ((double *) mem_99482)[i_98194 * (int64_t) 16 + defunc_0_reduce_res_97846];
                
                // futhark/microgpt.fut:250:67-129
                
                double neg_res_94535 = -neg_arg0_94534;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98186 = 0; i_98186 < (int64_t) 16; i_98186++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_94542 = ((double *) mem_99482)[i_98194 * (int64_t) 16 + i_98186];
                    
                    // futhark/microgpt.fut:250:44-129
                    
                    double zp_res_94543 = neg_res_94535 + zp_lhs_94542;
                    
                    // futhark/microgpt.fut:250:37-129
                    
                    double exp_res_94544 = futrts_exp64(zp_res_94543);
                    
                    ((double *) mem_99518)[i_98186] = exp_res_94544;
                }
                // futhark/microgpt.fut:4:11-25
                
                double x_94546;
                double r_94548 = 0.0;
                
                for (int64_t i_94547 = 0; i_94547 < (int64_t) 16; i_94547++) {
                    // futhark/microgpt.fut:251:57-67
                    
                    double lifted_lambda_res_94549 = ((double *) mem_99518)[i_94547];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94550 = r_94548 + lifted_lambda_res_94549;
                    double r_tmp_101151 = zp_res_94550;
                    
                    r_94548 = r_tmp_101151;
                }
                x_94546 = r_94548;
                // futhark/microgpt.fut:251:28-68
                
                double zs_res_94551 = 1.0 / x_94546;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98190 = 0; i_98190 < (int64_t) 4; i_98190++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_94558;
                    double r_94560 = 0.0;
                    
                    for (int64_t i_94559 = 0; i_94559 < (int64_t) 16; i_94559++) {
                        // futhark/microgpt.fut:251:75-85
                        
                        double zt_rhs_94561 = ((double *) mem_99518)[i_94559];
                        
                        // futhark/microgpt.fut:251:32-85
                        
                        double zt_res_94562 = zs_res_94551 * zt_rhs_94561;
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_94563 = ((double *) mem_99388)[i_98200 * (int64_t) 64 + i_94559 * (int64_t) 4 + i_98190];
                        
                        // futhark/microgpt.fut:251:71-115
                        
                        double zt_res_94564 = zt_res_94562 * zt_rhs_94563;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_94565 = r_94560 + zt_res_94564;
                        double r_tmp_101153 = zp_res_94565;
                        
                        r_94560 = r_tmp_101153;
                    }
                    defunc_0_lifted_lambda_res_94558 = r_94560;
                    ((double *) mem_99525)[i_98190] = defunc_0_lifted_lambda_res_94558;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_99513, i_98194 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99525, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_99469, i_98200 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_99481, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_99470, i_98200 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_99513, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98211 = 0; i_98211 < (int64_t) 16; i_98211++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98205 = 0; i_98205 < (int64_t) 16; i_98205++) {
                // futhark/microgpt.fut:252:52-55
                
                int64_t tmp_94614 = sdiv64(i_98205, (int64_t) 4);
                
                // futhark/microgpt.fut:252:41-57
                
                bool x_94615 = sle64((int64_t) 0, tmp_94614);
                
                // futhark/microgpt.fut:252:41-57
                
                bool y_94616 = slt64(tmp_94614, (int64_t) 4);
                
                // futhark/microgpt.fut:252:41-57
                
                bool bounds_check_94617 = x_94615 && y_94616;
                
                // futhark/microgpt.fut:252:41-57
                
                bool index_certs_94618;
                
                if (!bounds_check_94617) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_94614, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:252:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:252:12-78\n   #6  futhark/microgpt.fut:453:5-76\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:252:72-75
                
                int64_t tmp_94619 = smod64(i_98205, (int64_t) 4);
                
                // futhark/microgpt.fut:252:41-77
                
                bool x_94620 = sle64((int64_t) 0, tmp_94619);
                
                // futhark/microgpt.fut:252:41-77
                
                bool y_94621 = slt64(tmp_94619, (int64_t) 4);
                
                // futhark/microgpt.fut:252:41-77
                
                bool bounds_check_94622 = x_94620 && y_94621;
                
                // futhark/microgpt.fut:252:41-77
                
                bool index_certs_94623;
                
                if (!bounds_check_94622) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_94619, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:252:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:252:12-78\n   #6  futhark/microgpt.fut:453:5-76\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94624 = ((double *) mem_99470)[tmp_94614 * (int64_t) 64 + i_98211 * (int64_t) 4 + tmp_94619];
                
                ((double *) mem_99555)[i_98205] = lifted_lambda_res_94624;
            }
            // futhark/microgpt.fut:315:43-55
            
            double zp_lhs_94632 = ((double *) mem_99327)[i_98211];
            
            // futhark/microgpt.fut:315:43-83
            
            double zp_res_94633 = 1.0e-5 + zp_lhs_94632;
            
            // futhark/microgpt.fut:315:35-83
            
            double sqrt_res_94634 = futrts_sqrt64(zp_res_94633);
            
            ((double *) mem_99546)[i_98211] = sqrt_res_94634;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99547, i_98211 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99555, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98220 = 0; i_98220 < (int64_t) 16; i_98220++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98216 = 0; i_98216 < (int64_t) 16; i_98216++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90211;
                double r_90213 = 0.0;
                
                for (int64_t i_90212 = 0; i_90212 < (int64_t) 16; i_90212++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90214 = ((double *) mem_param_99094.mem)[i_98216 * (int64_t) 16 + i_90212];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90215 = ((double *) mem_99547)[i_98220 * (int64_t) 16 + i_90212];
                    
                    // futhark/microgpt.fut:253:63-103
                    
                    double zt_res_90216 = zt_lhs_90214 * zt_rhs_90215;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90217 = r_90213 + zt_res_90216;
                    double r_tmp_101159 = zp_res_90217;
                    
                    r_90213 = r_tmp_101159;
                }
                defunc_0_lifted_lambda_res_90211 = r_90213;
                ((double *) mem_99574)[i_98216] = defunc_0_lifted_lambda_res_90211;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99569, i_98220 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99574, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98228 = 0; i_98228 < (int64_t) 16; i_98228++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98224 = 0; i_98224 < (int64_t) 16; i_98224++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90232 = ((double *) mem_99569)[i_98228 * (int64_t) 16 + i_98224];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_90233 = ((double *) mem_99256)[i_98228 * (int64_t) 16 + i_98224];
                
                // futhark/microgpt.fut:254:42-80
                
                double zp_res_90234 = zp_lhs_90232 + zp_rhs_90233;
                
                ((double *) mem_99590)[i_98224] = zp_res_90234;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99585, i_98228 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99590, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98241 = 0; i_98241 < (int64_t) 16; i_98241++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94652;
            double r_94654 = 0.0;
            
            for (int64_t i_94653 = 0; i_94653 < (int64_t) 16; i_94653++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_94655 = ((double *) mem_99585)[i_98241 * (int64_t) 16 + i_94653];
                
                // futhark/microgpt.fut:255:83-122
                
                double zt_res_94656 = zt_lhs_94655 * zt_lhs_94655;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94657 = r_94654 + zt_res_94656;
                double r_tmp_101164 = zp_res_94657;
                
                r_94654 = r_tmp_101164;
            }
            defunc_0_lifted_lambda_res_94652 = r_94654;
            // futhark/microgpt.fut:255:62-140
            
            double zs_res_94658 = defunc_0_lifted_lambda_res_94652 / 16.0;
            
            // futhark/microgpt.fut:255:126-169
            
            double zp_res_94659 = 1.0e-5 + zs_res_94658;
            
            // futhark/microgpt.fut:255:52-169
            
            double sqrt_res_94660 = futrts_sqrt64(zp_res_94659);
            
            // futhark/microgpt.fut:255:43-171
            
            double zs_res_94661 = 1.0 / sqrt_res_94660;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98234 = 0; i_98234 < (int64_t) 16; i_98234++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_96309 = ((double *) mem_99585)[i_98241 * (int64_t) 16 + i_98234];
                
                // futhark/microgpt.fut:255:47-195
                
                double zt_res_96310 = zs_res_94661 * zt_rhs_96309;
                
                // futhark/microgpt.fut:281:45-88
                
                double zt_res_96318 = zt_rhs_96309 * zt_rhs_96309;
                
                ((double *) mem_99611)[i_98234] = zt_res_96318;
                ((double *) mem_99612)[i_98234] = zt_res_96310;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99601, i_98241 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99611, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99602, i_98241 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99612, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98252 = 0; i_98252 < (int64_t) 16; i_98252++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98246 = 0; i_98246 < (int64_t) 64; i_98246++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94709;
                double r_94711 = 0.0;
                
                for (int64_t i_94710 = 0; i_94710 < (int64_t) 16; i_94710++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94712 = ((double *) mem_param_99110.mem)[i_98246 * (int64_t) 16 + i_94710];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94713 = ((double *) mem_99602)[i_98252 * (int64_t) 16 + i_94710];
                    
                    // futhark/microgpt.fut:256:63-102
                    
                    double zt_res_94714 = zt_lhs_94712 * zt_rhs_94713;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94715 = r_94711 + zt_res_94714;
                    double r_tmp_101170 = zp_res_94715;
                    
                    r_94711 = r_tmp_101170;
                }
                defunc_0_lifted_lambda_res_94709 = r_94711;
                ((double *) mem_99642)[i_98246] = defunc_0_lifted_lambda_res_94709;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_94723;
            double r_94725 = 0.0;
            
            for (int64_t i_94724 = 0; i_94724 < (int64_t) 16; i_94724++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_94726 = ((double *) mem_99601)[i_98252 * (int64_t) 16 + i_94724];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_94727 = r_94725 + lifted_lambda_res_94726;
                double r_tmp_101171 = zp_res_94727;
                
                r_94725 = r_tmp_101171;
            }
            defunc_0_lifted_lambda_res_94723 = r_94725;
            // futhark/microgpt.fut:282:36-94
            
            double zs_res_94728 = defunc_0_lifted_lambda_res_94723 / 16.0;
            
            ((double *) mem_99633)[i_98252] = zs_res_94728;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99634, i_98252 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99642, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98263 = 0; i_98263 < (int64_t) 16; i_98263++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98257 = 0; i_98257 < (int64_t) 64; i_98257++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_94752 = ((double *) mem_99634)[i_98263 * (int64_t) 64 + i_98257];
                
                // futhark/microgpt.fut:257:41-69
                
                double max_res_94753 = fmax64(0.0, max_arg0_94752);
                
                ((double *) mem_99665)[i_98257] = max_res_94753;
            }
            // futhark/microgpt.fut:283:43-55
            
            double zp_lhs_94761 = ((double *) mem_99633)[i_98263];
            
            // futhark/microgpt.fut:283:43-83
            
            double zp_res_94762 = 1.0e-5 + zp_lhs_94761;
            
            // futhark/microgpt.fut:283:35-83
            
            double sqrt_res_94763 = futrts_sqrt64(zp_res_94762);
            
            ((double *) mem_99656)[i_98263] = sqrt_res_94763;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99657, i_98263 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98272 = 0; i_98272 < (int64_t) 16; i_98272++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98268 = 0; i_98268 < (int64_t) 16; i_98268++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90313;
                double r_90315 = 0.0;
                
                for (int64_t i_90314 = 0; i_90314 < (int64_t) 64; i_90314++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90316 = ((double *) mem_param_99086.mem)[i_98268 * (int64_t) 64 + i_90314];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90317 = ((double *) mem_99657)[i_98272 * (int64_t) 64 + i_90314];
                    
                    // futhark/microgpt.fut:258:63-104
                    
                    double zt_res_90318 = zt_lhs_90316 * zt_rhs_90317;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90319 = r_90315 + zt_res_90318;
                    double r_tmp_101177 = zp_res_90319;
                    
                    r_90315 = r_tmp_101177;
                }
                defunc_0_lifted_lambda_res_90313 = r_90315;
                ((double *) mem_99684)[i_98268] = defunc_0_lifted_lambda_res_90313;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99679, i_98272 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99684, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98280 = 0; i_98280 < (int64_t) 16; i_98280++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98276 = 0; i_98276 < (int64_t) 16; i_98276++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90334 = ((double *) mem_99679)[i_98280 * (int64_t) 16 + i_98276];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_90335 = ((double *) mem_99585)[i_98280 * (int64_t) 16 + i_98276];
                
                // futhark/microgpt.fut:259:42-81
                
                double zp_res_90336 = zp_lhs_90334 + zp_rhs_90335;
                
                ((double *) mem_99700)[i_98276] = zp_res_90336;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99695, i_98280 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99700, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98288 = 0; i_98288 < (int64_t) 16; i_98288++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98284 = 0; i_98284 < (int64_t) 27; i_98284++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90351;
                double r_90353 = 0.0;
                
                for (int64_t i_90352 = 0; i_90352 < (int64_t) 16; i_90352++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90354 = ((double *) mem_param_99118.mem)[i_98284 * (int64_t) 16 + i_90352];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90355 = ((double *) mem_99695)[i_98288 * (int64_t) 16 + i_90352];
                    
                    // futhark/microgpt.fut:260:63-103
                    
                    double zt_res_90356 = zt_lhs_90354 * zt_rhs_90355;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90357 = r_90353 + zt_res_90356;
                    double r_tmp_101182 = zp_res_90357;
                    
                    r_90353 = r_tmp_101182;
                }
                defunc_0_lifted_lambda_res_90351 = r_90353;
                ((double *) mem_99716)[i_98284] = defunc_0_lifted_lambda_res_90351;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99711, i_98288 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99716, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98295 = 0; i_98295 < (int64_t) 16; i_98295++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_97862;
            int64_t defunc_0_reduce_res_97863;
            double redout_98290;
            int64_t redout_98291;
            
            redout_98290 = -INFINITY;
            redout_98291 = (int64_t) 27;
            for (int64_t i_98292 = 0; i_98292 < (int64_t) 27; i_98292++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_96343 = ((double *) mem_99711)[i_98295 * (int64_t) 27 + i_98292];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_90398 = lifted_lambda_res_96343 < redout_98290;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_90399;
                
                if (zg_res_90398) {
                    lifted_lambda_res_90399 = redout_98290;
                } else {
                    lifted_lambda_res_90399 = lifted_lambda_res_96343;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_90400;
                
                if (zg_res_90398) {
                    lifted_lambda_res_90400 = redout_98291;
                } else {
                    lifted_lambda_res_90400 = i_98292;
                }
                
                double redout_tmp_101184 = lifted_lambda_res_90399;
                int64_t redout_tmp_101185 = lifted_lambda_res_90400;
                
                redout_98290 = redout_tmp_101184;
                redout_98291 = redout_tmp_101185;
            }
            defunc_0_reduce_res_97862 = redout_98290;
            defunc_0_reduce_res_97863 = redout_98291;
            // futhark/microgpt.fut:265:32-88
            
            bool x_90401 = sle64((int64_t) 0, defunc_0_reduce_res_97863);
            
            // futhark/microgpt.fut:265:32-88
            
            bool y_90402 = slt64(defunc_0_reduce_res_97863, (int64_t) 27);
            
            // futhark/microgpt.fut:265:32-88
            
            bool bounds_check_90403 = x_90401 && y_90402;
            
            // futhark/microgpt.fut:265:32-88
            
            bool index_certs_90404;
            
            if (!bounds_check_90403) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_97863, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:265:32-88\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:265:12-89\n   #4  futhark/microgpt.fut:453:5-76\n   #5  futhark/microgpt.fut:458:26-464:31\n   #6  futhark/microgpt.fut:480:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_90405 = ((double *) mem_99711)[i_98295 * (int64_t) 27 + defunc_0_reduce_res_97863];
            
            ((double *) mem_99727)[i_98295] = lifted_lambda_res_90405;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98303 = 0; i_98303 < (int64_t) 16; i_98303++) {
            // futhark/microgpt.fut:266:71-81
            
            double neg_arg0_90413 = ((double *) mem_99727)[i_98303];
            
            // futhark/microgpt.fut:266:65-81
            
            double neg_res_90414 = -neg_arg0_90413;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98299 = 0; i_98299 < (int64_t) 27; i_98299++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90421 = ((double *) mem_99711)[i_98303 * (int64_t) 27 + i_98299];
                
                // futhark/microgpt.fut:266:42-81
                
                double zp_res_90422 = neg_res_90414 + zp_lhs_90421;
                
                ((double *) mem_99739)[i_98299] = zp_res_90422;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99734, i_98303 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99739, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98311 = 0; i_98311 < (int64_t) 16; i_98311++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98307 = 0; i_98307 < (int64_t) 27; i_98307++) {
                // futhark/microgpt.fut:4:11-25
                
                double exp_arg0_90437 = ((double *) mem_99734)[i_98311 * (int64_t) 27 + i_98307];
                
                // futhark/microgpt.fut:267:42-65
                
                double exp_res_90438 = futrts_exp64(exp_arg0_90437);
                
                ((double *) mem_99755)[i_98307] = exp_res_90438;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99750, i_98311 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99755, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98315 = 0; i_98315 < (int64_t) 16; i_98315++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90447;
            double r_90449 = 0.0;
            
            for (int64_t i_90448 = 0; i_90448 < (int64_t) 27; i_90448++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_90450 = ((double *) mem_99750)[i_98315 * (int64_t) 27 + i_90448];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90451 = r_90449 + lifted_lambda_res_90450;
                double r_tmp_101191 = zp_res_90451;
                
                r_90449 = r_tmp_101191;
            }
            defunc_0_lifted_lambda_res_90447 = r_90449;
            ((double *) mem_99766)[i_98315] = defunc_0_lifted_lambda_res_90447;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98326 = 0; i_98326 < (int64_t) 16; i_98326++) {
            // futhark/microgpt.fut:269:49-59
            
            double zs_rhs_90459 = ((double *) mem_99766)[i_98326];
            
            // futhark/microgpt.fut:269:41-59
            
            double zs_res_90460 = 1.0 / zs_rhs_90459;
            double x_97867;
            int64_t x_97868;
            double redout_98317;
            int64_t redout_98318;
            
            redout_98317 = -INFINITY;
            redout_98318 = (int64_t) 27;
            for (int64_t i_98319 = 0; i_98319 < (int64_t) 27; i_98319++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_96368 = ((double *) mem_99711)[i_98326 * (int64_t) 27 + i_98319];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_90480 = lifted_lambda_res_96368 < redout_98317;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_90481;
                
                if (zg_res_90480) {
                    lifted_lambda_res_90481 = redout_98317;
                } else {
                    lifted_lambda_res_90481 = lifted_lambda_res_96368;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_90482;
                
                if (zg_res_90480) {
                    lifted_lambda_res_90482 = redout_98318;
                } else {
                    lifted_lambda_res_90482 = i_98319;
                }
                
                double redout_tmp_101193 = lifted_lambda_res_90481;
                int64_t redout_tmp_101194 = lifted_lambda_res_90482;
                
                redout_98317 = redout_tmp_101193;
                redout_98318 = redout_tmp_101194;
            }
            x_97867 = redout_98317;
            x_97868 = redout_98318;
            
            double x_96387 = ((double *) mem_99711)[i_98326 * (int64_t) 27 + x_97868];
            
            // futhark/microgpt.fut:269:175-237
            
            double neg_res_90488 = -x_96387;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90461;
            double r_90463 = 0.0;
            
            for (int64_t i_90462 = 0; i_90462 < (int64_t) 27; i_90462++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98322 = 0; i_98322 < (int64_t) 27; i_98322++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_90495 = ((double *) mem_99711)[i_98326 * (int64_t) 27 + i_98322];
                    
                    // futhark/microgpt.fut:269:152-237
                    
                    double zp_res_90496 = neg_res_90488 + zp_lhs_90495;
                    
                    // futhark/microgpt.fut:269:145-237
                    
                    double exp_res_90497 = futrts_exp64(zp_res_90496);
                    
                    ((double *) mem_99777)[i_98322] = exp_res_90497;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90499;
                double r_90501 = 0.0;
                
                for (int64_t i_90500 = 0; i_90500 < (int64_t) 27; i_90500++) {
                    // futhark/microgpt.fut:270:45-55
                    
                    double lifted_lambda_res_90502 = ((double *) mem_99777)[i_90500];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90503 = r_90501 + lifted_lambda_res_90502;
                    double r_tmp_101197 = zp_res_90503;
                    
                    r_90501 = r_tmp_101197;
                }
                defunc_0_lifted_lambda_res_90499 = r_90501;
                // futhark/microgpt.fut:270:16-56
                
                double zs_res_90504 = 1.0 / defunc_0_lifted_lambda_res_90499;
                
                // futhark/microgpt.fut:270:63-73
                
                double zt_rhs_90505 = ((double *) mem_99777)[i_90462];
                
                // futhark/microgpt.fut:270:20-73
                
                double zt_res_90506 = zs_res_90504 * zt_rhs_90505;
                
                // futhark/microgpt.fut:270:6-73
                
                double zs_res_90507 = 1.0 / zt_res_90506;
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_90508 = ((double *) mem_99750)[i_98326 * (int64_t) 27 + i_90462];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_90509 = ((double *) mem_99223)[i_98326 * (int64_t) 27 + i_90462];
                
                // futhark/microgpt.fut:270:81-120
                
                double zt_res_90510 = zt_lhs_90508 * zt_rhs_90509;
                
                // futhark/microgpt.fut:270:10-120
                
                double zt_res_90511 = zs_res_90507 * zt_res_90510;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90512 = r_90463 + zt_res_90511;
                double r_tmp_101195 = zp_res_90512;
                
                r_90463 = r_tmp_101195;
            }
            defunc_0_lifted_lambda_res_90461 = r_90463;
            // futhark/microgpt.fut:269:71-270:124
            
            double zt_res_90513 = zs_res_90460 * defunc_0_lifted_lambda_res_90461;
            
            // futhark/microgpt.fut:269:45-270:124
            
            double zt_res_90514 = zs_res_90460 * zt_res_90513;
            
            // futhark/microgpt.fut:269:33-270:124
            
            double neg_res_90515 = -zt_res_90514;
            
            ((double *) mem_99773)[i_98326] = neg_res_90515;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98341 = 0; i_98341 < (int64_t) 16; i_98341++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_97869;
            int64_t defunc_0_reduce_res_97870;
            double redout_98328;
            int64_t redout_98329;
            
            redout_98328 = -INFINITY;
            redout_98329 = (int64_t) 27;
            for (int64_t i_98330 = 0; i_98330 < (int64_t) 27; i_98330++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_96404 = ((double *) mem_99711)[i_98341 * (int64_t) 27 + i_98330];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_90539 = lifted_lambda_res_96404 < redout_98328;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_90540;
                
                if (zg_res_90539) {
                    lifted_lambda_res_90540 = redout_98328;
                } else {
                    lifted_lambda_res_90540 = lifted_lambda_res_96404;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_90541;
                
                if (zg_res_90539) {
                    lifted_lambda_res_90541 = redout_98329;
                } else {
                    lifted_lambda_res_90541 = i_98330;
                }
                
                double redout_tmp_101199 = lifted_lambda_res_90540;
                int64_t redout_tmp_101200 = lifted_lambda_res_90541;
                
                redout_98328 = redout_tmp_101199;
                redout_98329 = redout_tmp_101200;
            }
            defunc_0_reduce_res_97869 = redout_98328;
            defunc_0_reduce_res_97870 = redout_98329;
            // futhark/microgpt.fut:271:110-166
            
            bool x_90542 = sle64((int64_t) 0, defunc_0_reduce_res_97870);
            
            // futhark/microgpt.fut:271:110-166
            
            bool y_90543 = slt64(defunc_0_reduce_res_97870, (int64_t) 27);
            
            // futhark/microgpt.fut:271:110-166
            
            bool bounds_check_90544 = x_90542 && y_90543;
            
            // futhark/microgpt.fut:271:110-166
            
            bool index_certs_90545;
            
            if (!bounds_check_90544) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_97870, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:271:110-166\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:271:53-170\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:271:12-272:144\n   #9  futhark/microgpt.fut:453:5-76\n   #10 futhark/microgpt.fut:458:26-464:31\n   #11 futhark/microgpt.fut:480:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double neg_arg0_90546 = ((double *) mem_99711)[i_98341 * (int64_t) 27 + defunc_0_reduce_res_97870];
            
            // futhark/microgpt.fut:271:104-166
            
            double neg_res_90547 = -neg_arg0_90546;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98333 = 0; i_98333 < (int64_t) 27; i_98333++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90554 = ((double *) mem_99711)[i_98341 * (int64_t) 27 + i_98333];
                
                // futhark/microgpt.fut:271:81-166
                
                double zp_res_90555 = neg_res_90547 + zp_lhs_90554;
                
                // futhark/microgpt.fut:271:74-166
                
                double exp_res_90556 = futrts_exp64(zp_res_90555);
                
                ((double *) mem_99792)[i_98333] = exp_res_90556;
            }
            // futhark/microgpt.fut:272:15-25
            
            double zs_rhs_90558 = ((double *) mem_99766)[i_98341];
            
            // futhark/microgpt.fut:272:7-25
            
            double zs_res_90559 = 1.0 / zs_rhs_90558;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90560;
            double r_90562 = 0.0;
            
            for (int64_t i_90561 = 0; i_90561 < (int64_t) 27; i_90561++) {
                // futhark/microgpt.fut:272:72-82
                
                double lifted_lambda_res_90563 = ((double *) mem_99792)[i_90561];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90564 = r_90562 + lifted_lambda_res_90563;
                double r_tmp_101202 = zp_res_90564;
                
                r_90562 = r_tmp_101202;
            }
            defunc_0_lifted_lambda_res_90560 = r_90562;
            // futhark/microgpt.fut:272:43-83
            
            double zs_res_90565 = 1.0 / defunc_0_lifted_lambda_res_90560;
            
            // futhark/microgpt.fut:272:131-141
            
            double zp_rhs_90566 = ((double *) mem_99773)[i_98341];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98337 = 0; i_98337 < (int64_t) 27; i_98337++) {
                // futhark/microgpt.fut:272:90-100
                
                double zt_rhs_90573 = ((double *) mem_99792)[i_98337];
                
                // futhark/microgpt.fut:272:47-100
                
                double zt_res_90574 = zs_res_90565 * zt_rhs_90573;
                
                // futhark/microgpt.fut:272:33-100
                
                double zs_res_90575 = 1.0 / zt_res_90574;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90576 = ((double *) mem_99223)[i_98341 * (int64_t) 27 + i_98337];
                
                // futhark/microgpt.fut:272:37-124
                
                double zt_res_90577 = zs_res_90575 * zt_rhs_90576;
                
                // futhark/microgpt.fut:272:11-124
                
                double zt_res_90578 = zs_res_90559 * zt_res_90577;
                
                // futhark/microgpt.fut:272:27-141
                
                double zp_res_90579 = zp_rhs_90566 + zt_res_90578;
                
                ((double *) mem_99799)[i_98337] = zp_res_90579;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99787, i_98341 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99799, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98349 = 0; i_98349 < (int64_t) 16; i_98349++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98345 = 0; i_98345 < (int64_t) 27; i_98345++) {
                // futhark/microgpt.fut:4:11-25
                
                double exp_arg0_90594 = ((double *) mem_99734)[i_98349 * (int64_t) 27 + i_98345];
                
                // futhark/microgpt.fut:273:43-66
                
                double exp_res_90595 = futrts_exp64(exp_arg0_90594);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90596 = ((double *) mem_99787)[i_98349 * (int64_t) 27 + i_98345];
                
                // futhark/microgpt.fut:273:43-89
                
                double zt_res_90597 = exp_res_90595 * zt_rhs_90596;
                
                ((double *) mem_99815)[i_98345] = zt_res_90597;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99810, i_98349 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99815, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98353 = 0; i_98353 < (int64_t) 16; i_98353++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90606;
            double r_90608 = 0.0;
            
            for (int64_t i_90607 = 0; i_90607 < (int64_t) 27; i_90607++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_90609 = ((double *) mem_99810)[i_98353 * (int64_t) 27 + i_90607];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90610 = r_90608 + lifted_lambda_res_90609;
                double r_tmp_101207 = zp_res_90610;
                
                r_90608 = r_tmp_101207;
            }
            defunc_0_lifted_lambda_res_90606 = r_90608;
            // futhark/microgpt.fut:274:33-78
            
            double neg_res_90611 = -defunc_0_lifted_lambda_res_90606;
            
            ((double *) mem_99826)[i_98353] = neg_res_90611;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98364 = 0; i_98364 < (int64_t) 16; i_98364++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_97879;
            int64_t defunc_0_reduce_res_97880;
            double redout_98355;
            int64_t redout_98356;
            
            redout_98355 = -INFINITY;
            redout_98356 = (int64_t) 27;
            for (int64_t i_98357 = 0; i_98357 < (int64_t) 27; i_98357++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_96430 = ((double *) mem_99711)[i_98364 * (int64_t) 27 + i_98357];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_90635 = lifted_lambda_res_96430 < redout_98355;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_90636;
                
                if (zg_res_90635) {
                    lifted_lambda_res_90636 = redout_98355;
                } else {
                    lifted_lambda_res_90636 = lifted_lambda_res_96430;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_90637;
                
                if (zg_res_90635) {
                    lifted_lambda_res_90637 = redout_98356;
                } else {
                    lifted_lambda_res_90637 = i_98357;
                }
                
                double redout_tmp_101209 = lifted_lambda_res_90636;
                int64_t redout_tmp_101210 = lifted_lambda_res_90637;
                
                redout_98355 = redout_tmp_101209;
                redout_98356 = redout_tmp_101210;
            }
            defunc_0_reduce_res_97879 = redout_98355;
            defunc_0_reduce_res_97880 = redout_98356;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98360 = 0; i_98360 < (int64_t) 27; i_98360++) {
                // futhark/microgpt.fut:275:42-128
                
                bool cond_90640 = i_98360 == defunc_0_reduce_res_97880;
                
                // futhark/microgpt.fut:275:42-128
                
                double lifted_lambda_res_90641;
                
                if (cond_90640) {
                    // futhark/microgpt.fut:275:108-118
                    
                    double lifted_lambda_res_t_res_97878 = ((double *) mem_99826)[i_98364];
                    
                    lifted_lambda_res_90641 = lifted_lambda_res_t_res_97878;
                } else {
                    lifted_lambda_res_90641 = 0.0;
                }
                ((double *) mem_99838)[i_98360] = lifted_lambda_res_90641;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99833, i_98364 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99838, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98372 = 0; i_98372 < (int64_t) 16; i_98372++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98368 = 0; i_98368 < (int64_t) 27; i_98368++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90661 = ((double *) mem_99810)[i_98372 * (int64_t) 27 + i_98368];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_90662 = ((double *) mem_99833)[i_98372 * (int64_t) 27 + i_98368];
                
                // futhark/microgpt.fut:276:42-81
                
                double zp_res_90663 = zp_lhs_90661 + zp_rhs_90662;
                
                ((double *) mem_99854)[i_98368] = zp_res_90663;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99849, i_98372 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99854, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98380 = 0; i_98380 < (int64_t) 16; i_98380++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98376 = 0; i_98376 < (int64_t) 16; i_98376++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90678;
                double r_90680 = 0.0;
                
                for (int64_t i_90679 = 0; i_90679 < (int64_t) 27; i_90679++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90681 = ((double *) mem_param_99118.mem)[i_90679 * (int64_t) 16 + i_98376];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90682 = ((double *) mem_99849)[i_98380 * (int64_t) 27 + i_90679];
                    
                    // futhark/microgpt.fut:277:66-110
                    
                    double zt_res_90683 = zt_lhs_90681 * zt_rhs_90682;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90684 = r_90680 + zt_res_90683;
                    double r_tmp_101216 = zp_res_90684;
                    
                    r_90680 = r_tmp_101216;
                }
                defunc_0_lifted_lambda_res_90678 = r_90680;
                ((double *) mem_99870)[i_98376] = defunc_0_lifted_lambda_res_90678;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99865, i_98380 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99870, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98393 = 0; i_98393 < (int64_t) 16; i_98393++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98386 = 0; i_98386 < (int64_t) 64; i_98386++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_96466;
                double r_96468 = 0.0;
                
                for (int64_t i_96467 = 0; i_96467 < (int64_t) 16; i_96467++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_96469 = ((double *) mem_param_99086.mem)[i_96467 * (int64_t) 64 + i_98386];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_96470 = ((double *) mem_99865)[i_98393 * (int64_t) 16 + i_96467];
                    
                    // futhark/microgpt.fut:278:67-112
                    
                    double zt_res_96471 = zt_lhs_96469 * zt_rhs_96470;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_96472 = r_96468 + zt_res_96471;
                    double r_tmp_101221 = zp_res_96472;
                    
                    r_96468 = r_tmp_101221;
                }
                defunc_0_lifted_lambda_res_96466 = r_96468;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_96479;
                double r_96481 = 0.0;
                
                for (int64_t i_96480 = 0; i_96480 < (int64_t) 16; i_96480++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_96482 = ((double *) mem_99865)[i_96480 * (int64_t) 16 + i_98393];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_96483 = ((double *) mem_99657)[i_96480 * (int64_t) 64 + i_98386];
                    
                    // futhark/microgpt.fut:335:69-112
                    
                    double zt_res_96484 = zt_lhs_96482 * zt_rhs_96483;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_96485 = r_96481 + zt_res_96484;
                    double r_tmp_101222 = zp_res_96485;
                    
                    r_96481 = r_tmp_101222;
                }
                defunc_0_lifted_lambda_res_96479 = r_96481;
                ((double *) mem_99891)[i_98386] = defunc_0_lifted_lambda_res_96479;
                ((double *) mem_99892)[i_98386] = defunc_0_lifted_lambda_res_96466;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99881, i_98393 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99891, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99882, i_98393 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98402 = 0; i_98402 < (int64_t) 16; i_98402++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98398 = 0; i_98398 < (int64_t) 64; i_98398++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_90720 = ((double *) mem_99634)[i_98402 * (int64_t) 64 + i_98398];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_90721 = fmax64(0.0, indicatorp_arg0_90720);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_90722 = fsignum64(max_res_90721);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90723 = ((double *) mem_99882)[i_98402 * (int64_t) 64 + i_98398];
                
                // futhark/microgpt.fut:279:46-102
                
                double zt_res_90724 = sgn_res_90722 * zt_rhs_90723;
                
                ((double *) mem_99918)[i_98398] = zt_res_90724;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99913, i_98402 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99918, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98410 = 0; i_98410 < (int64_t) 16; i_98410++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98406 = 0; i_98406 < (int64_t) 16; i_98406++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90739;
                double r_90741 = 0.0;
                
                for (int64_t i_90740 = 0; i_90740 < (int64_t) 64; i_90740++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90742 = ((double *) mem_param_99110.mem)[i_90740 * (int64_t) 16 + i_98406];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90743 = ((double *) mem_99913)[i_98410 * (int64_t) 64 + i_90740];
                    
                    // futhark/microgpt.fut:280:67-111
                    
                    double zt_res_90744 = zt_lhs_90742 * zt_rhs_90743;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90745 = r_90741 + zt_res_90744;
                    double r_tmp_101227 = zp_res_90745;
                    
                    r_90741 = r_tmp_101227;
                }
                defunc_0_lifted_lambda_res_90739 = r_90741;
                ((double *) mem_99934)[i_98406] = defunc_0_lifted_lambda_res_90739;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99929, i_98410 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99934, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98414 = 0; i_98414 < (int64_t) 16; i_98414++) {
            // futhark/microgpt.fut:284:51-63
            
            double zs_rhs_90793 = ((double *) mem_99656)[i_98414];
            
            // futhark/microgpt.fut:284:43-63
            
            double zs_res_90794 = 1.0 / zs_rhs_90793;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90795;
            double r_90797 = 0.0;
            
            for (int64_t i_90796 = 0; i_90796 < (int64_t) 16; i_90796++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_90798 = ((double *) mem_99585)[i_98414 * (int64_t) 16 + i_90796];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_90799 = ((double *) mem_99929)[i_98414 * (int64_t) 16 + i_90796];
                
                // futhark/microgpt.fut:284:120-164
                
                double zt_res_90800 = zt_lhs_90798 * zt_rhs_90799;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90801 = r_90797 + zt_res_90800;
                double r_tmp_101229 = zp_res_90801;
                
                r_90797 = r_tmp_101229;
            }
            defunc_0_lifted_lambda_res_90795 = r_90797;
            // futhark/microgpt.fut:284:75-166
            
            double zt_res_90802 = zs_res_90794 * defunc_0_lifted_lambda_res_90795;
            
            // futhark/microgpt.fut:284:47-166
            
            double zt_res_90803 = zs_res_90794 * zt_res_90802;
            
            // futhark/microgpt.fut:284:35-166
            
            double neg_res_90804 = -zt_res_90803;
            
            ((double *) mem_99945)[i_98414] = neg_res_90804;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98418 = 0; i_98418 < (int64_t) 16; i_98418++) {
            // futhark/microgpt.fut:285:72-84
            
            double zp_lhs_90812 = ((double *) mem_99633)[i_98418];
            
            // futhark/microgpt.fut:285:72-112
            
            double zp_res_90813 = 1.0e-5 + zp_lhs_90812;
            
            // futhark/microgpt.fut:285:64-112
            
            double sqrt_res_90814 = futrts_sqrt64(zp_res_90813);
            
            // futhark/microgpt.fut:285:50-114
            
            double zt_res_90815 = 2.0 * sqrt_res_90814;
            
            // futhark/microgpt.fut:285:36-114
            
            double zs_res_90816 = 1.0 / zt_res_90815;
            
            // futhark/microgpt.fut:285:122-134
            
            double zt_rhs_90817 = ((double *) mem_99945)[i_98418];
            
            // futhark/microgpt.fut:285:40-134
            
            double zt_res_90818 = zs_res_90816 * zt_rhs_90817;
            
            ((double *) mem_99952)[i_98418] = zt_res_90818;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98422 = 0; i_98422 < (int64_t) 16; i_98422++) {
            // futhark/microgpt.fut:286:45-57
            
            double zs_lhs_90826 = ((double *) mem_99952)[i_98422];
            
            // futhark/microgpt.fut:286:45-72
            
            double zs_res_90827 = zs_lhs_90826 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_101232 = 0; nest_i_101232 < (int64_t) 16; nest_i_101232++) {
                ((double *) mem_99959)[i_98422 * (int64_t) 16 + nest_i_101232] = zs_res_90827;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98430 = 0; i_98430 < (int64_t) 16; i_98430++) {
            // futhark/microgpt.fut:287:81-93
            
            double zs_rhs_90836 = ((double *) mem_99656)[i_98430];
            
            // futhark/microgpt.fut:287:73-93
            
            double zs_res_90837 = 1.0 / zs_rhs_90836;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98426 = 0; i_98426 < (int64_t) 16; i_98426++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90844 = ((double *) mem_99865)[i_98430 * (int64_t) 16 + i_98426];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90845 = ((double *) mem_99929)[i_98430 * (int64_t) 16 + i_98426];
                
                // futhark/microgpt.fut:287:77-119
                
                double zt_res_90846 = zs_res_90837 * zt_rhs_90845;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90847 = ((double *) mem_99585)[i_98430 * (int64_t) 16 + i_98426];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90848 = ((double *) mem_99959)[i_98430 * (int64_t) 16 + i_98426];
                
                // futhark/microgpt.fut:287:126-170
                
                double zt_res_90849 = zt_lhs_90847 * zt_rhs_90848;
                
                // futhark/microgpt.fut:287:95-170
                
                double zp_res_90850 = zt_res_90846 + zt_res_90849;
                
                // futhark/microgpt.fut:287:121-222
                
                double zp_res_90851 = zt_res_90849 + zp_res_90850;
                
                // futhark/microgpt.fut:287:45-222
                
                double zp_res_90852 = zp_lhs_90844 + zp_res_90851;
                
                ((double *) mem_99974)[i_98426] = zp_res_90852;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99969, i_98430 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99974, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98443 = 0; i_98443 < (int64_t) 16; i_98443++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98436 = 0; i_98436 < (int64_t) 16; i_98436++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_96508;
                double r_96510 = 0.0;
                
                for (int64_t i_96509 = 0; i_96509 < (int64_t) 16; i_96509++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_96511 = ((double *) mem_param_99094.mem)[i_96509 * (int64_t) 16 + i_98436];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_96512 = ((double *) mem_99969)[i_98443 * (int64_t) 16 + i_96509];
                    
                    // futhark/microgpt.fut:288:67-112
                    
                    double zt_res_96513 = zt_lhs_96511 * zt_rhs_96512;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_96514 = r_96510 + zt_res_96513;
                    double r_tmp_101239 = zp_res_96514;
                    
                    r_96510 = r_tmp_101239;
                }
                defunc_0_lifted_lambda_res_96508 = r_96510;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_96521;
                double r_96523 = 0.0;
                
                for (int64_t i_96522 = 0; i_96522 < (int64_t) 16; i_96522++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_96524 = ((double *) mem_99969)[i_96522 * (int64_t) 16 + i_98443];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_96525 = ((double *) mem_99547)[i_96522 * (int64_t) 16 + i_98436];
                    
                    // futhark/microgpt.fut:333:68-112
                    
                    double zt_res_96526 = zt_lhs_96524 * zt_rhs_96525;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_96527 = r_96523 + zt_res_96526;
                    double r_tmp_101240 = zp_res_96527;
                    
                    r_96523 = r_tmp_101240;
                }
                defunc_0_lifted_lambda_res_96521 = r_96523;
                ((double *) mem_99995)[i_98436] = defunc_0_lifted_lambda_res_96521;
                ((double *) mem_99996)[i_98436] = defunc_0_lifted_lambda_res_96508;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99985, i_98443 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99995, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_99986, i_98443 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_99996, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98465 = 0; i_98465 < (int64_t) 4; i_98465++) {
            // futhark/microgpt.fut:289:74-77
            
            int64_t zp_lhs_94879 = mul64((int64_t) 4, i_98465);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98458 = 0; i_98458 < (int64_t) 16; i_98458++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98448 = 0; i_98448 < (int64_t) 4; i_98448++) {
                    // futhark/microgpt.fut:289:79-87
                    
                    int64_t tmp_96549 = add64(zp_lhs_94879, i_98448);
                    
                    // futhark/microgpt.fut:289:52-89
                    
                    bool x_96550 = sle64((int64_t) 0, tmp_96549);
                    
                    // futhark/microgpt.fut:289:52-89
                    
                    bool y_96551 = slt64(tmp_96549, (int64_t) 16);
                    
                    // futhark/microgpt.fut:289:52-89
                    
                    bool bounds_check_96552 = x_96550 && y_96551;
                    
                    // futhark/microgpt.fut:289:52-89
                    
                    bool index_certs_96553;
                    
                    if (!bounds_check_96552) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_96549, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:289:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:289:13-90\n   #9  futhark/microgpt.fut:453:5-76\n   #10 futhark/microgpt.fut:458:26-464:31\n   #11 futhark/microgpt.fut:480:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96554 = ((double *) mem_99986)[i_98458 * (int64_t) 16 + tmp_96549];
                    
                    ((double *) mem_100039)[i_98448] = lifted_lambda_res_96554;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98452 = 0; i_98452 < (int64_t) 16; i_98452++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_96568 = ((double *) mem_99469)[i_98465 * (int64_t) 256 + i_98458 * (int64_t) 16 + i_98452];
                    
                    // futhark/microgpt.fut:291:55-97
                    
                    double zs_res_96569 = zs_lhs_96568 / 2.0;
                    double zp_rhs_96570 = ((double *) masks_mem_99080.mem)[step_89818 * (int64_t) 256 + i_98458 * (int64_t) 16 + i_98452];
                    
                    // futhark/microgpt.fut:291:84-123
                    
                    double zp_res_96571 = zs_res_96569 + zp_rhs_96570;
                    
                    ((double *) mem_100046)[i_98452] = zp_res_96571;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100029, i_98458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100046, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100030, i_98458 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100039, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100017, i_98465 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100029, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100018, i_98465 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_100030, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98500 = 0; i_98500 < (int64_t) 4; i_98500++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98490 = 0; i_98490 < (int64_t) 16; i_98490++) {
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_97895;
                int64_t defunc_0_reduce_res_97896;
                double defunc_0_reduce_res_97897;
                int64_t defunc_0_reduce_res_97898;
                double redout_98469;
                int64_t redout_98470;
                double redout_98471;
                int64_t redout_98472;
                
                redout_98469 = -INFINITY;
                redout_98470 = (int64_t) 16;
                redout_98471 = -INFINITY;
                redout_98472 = (int64_t) 16;
                for (int64_t i_98474 = 0; i_98474 < (int64_t) 16; i_98474++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_96910 = ((double *) mem_100017)[i_98500 * (int64_t) 256 + i_98490 * (int64_t) 16 + i_98474];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_96922;
                    double r_96924 = 0.0;
                    
                    for (int64_t i_96923 = 0; i_96923 < (int64_t) 4; i_96923++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_96925 = ((double *) mem_100018)[i_98500 * (int64_t) 64 + i_98490 * (int64_t) 4 + i_96923];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_96926 = ((double *) mem_99388)[i_98500 * (int64_t) 64 + i_98474 * (int64_t) 4 + i_96923];
                        
                        // futhark/microgpt.fut:294:75-135
                        
                        double zt_res_96927 = zt_lhs_96925 * zt_rhs_96926;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_96928 = r_96924 + zt_res_96927;
                        double r_tmp_101258 = zp_res_96928;
                        
                        r_96924 = r_tmp_101258;
                    }
                    defunc_0_lifted_lambda_res_96922 = r_96924;
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_96705 = lifted_lambda_res_96910 < redout_98469;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_96706;
                    
                    if (zg_res_96705) {
                        lifted_lambda_res_96706 = redout_98469;
                    } else {
                        lifted_lambda_res_96706 = lifted_lambda_res_96910;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_96707;
                    
                    if (zg_res_96705) {
                        lifted_lambda_res_96707 = redout_98470;
                    } else {
                        lifted_lambda_res_96707 = i_98474;
                    }
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_96784 = lifted_lambda_res_96910 < redout_98471;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_96785;
                    
                    if (zg_res_96784) {
                        lifted_lambda_res_96785 = redout_98471;
                    } else {
                        lifted_lambda_res_96785 = lifted_lambda_res_96910;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_96786;
                    
                    if (zg_res_96784) {
                        lifted_lambda_res_96786 = redout_98472;
                    } else {
                        lifted_lambda_res_96786 = i_98474;
                    }
                    ((double *) mem_100102)[i_98474] = defunc_0_lifted_lambda_res_96922;
                    
                    double redout_tmp_101253 = lifted_lambda_res_96706;
                    int64_t redout_tmp_101254 = lifted_lambda_res_96707;
                    double redout_tmp_101255 = lifted_lambda_res_96785;
                    int64_t redout_tmp_101256 = lifted_lambda_res_96786;
                    
                    redout_98469 = redout_tmp_101253;
                    redout_98470 = redout_tmp_101254;
                    redout_98471 = redout_tmp_101255;
                    redout_98472 = redout_tmp_101256;
                }
                defunc_0_reduce_res_97895 = redout_98469;
                defunc_0_reduce_res_97896 = redout_98470;
                defunc_0_reduce_res_97897 = redout_98471;
                defunc_0_reduce_res_97898 = redout_98472;
                // futhark/microgpt.fut:292:135-213
                
                bool x_96708 = sle64((int64_t) 0, defunc_0_reduce_res_97896);
                
                // futhark/microgpt.fut:292:135-213
                
                bool y_96709 = slt64(defunc_0_reduce_res_97896, (int64_t) 16);
                
                // futhark/microgpt.fut:292:135-213
                
                bool bounds_check_96710 = x_96708 && y_96709;
                
                // futhark/microgpt.fut:292:135-213
                
                bool index_certs_96711;
                
                if (!bounds_check_96710) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_97896, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:292:135-213\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:292:66-217\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:15:29-44\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:15:15-45\n   #11 futhark/microgpt.fut:292:13-293:71\n   #12 futhark/microgpt.fut:453:5-76\n   #13 futhark/microgpt.fut:458:26-464:31\n   #14 futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:295:43-121
                
                bool x_96787 = sle64((int64_t) 0, defunc_0_reduce_res_97898);
                
                // futhark/microgpt.fut:295:43-121
                
                bool y_96788 = slt64(defunc_0_reduce_res_97898, (int64_t) 16);
                
                // futhark/microgpt.fut:295:43-121
                
                bool bounds_check_96789 = x_96787 && y_96788;
                
                // futhark/microgpt.fut:295:43-121
                
                bool index_certs_96790;
                
                if (!bounds_check_96789) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_97898, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:295:43-121\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:295:13-122\n   #6  futhark/microgpt.fut:453:5-76\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_96712 = ((double *) mem_100017)[i_98500 * (int64_t) 256 + i_98490 * (int64_t) 16 + defunc_0_reduce_res_97896];
                
                // futhark/microgpt.fut:292:129-213
                
                double neg_res_96713 = -neg_arg0_96712;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98478 = 0; i_98478 < (int64_t) 16; i_98478++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_96720 = ((double *) mem_100017)[i_98500 * (int64_t) 256 + i_98490 * (int64_t) 16 + i_98478];
                    
                    // futhark/microgpt.fut:292:95-213
                    
                    double zp_res_96721 = neg_res_96713 + zp_lhs_96720;
                    
                    // futhark/microgpt.fut:292:88-213
                    
                    double exp_res_96722 = futrts_exp64(zp_res_96721);
                    
                    ((double *) mem_100109)[i_98478] = exp_res_96722;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_96724;
                double r_96726 = 0.0;
                
                for (int64_t i_96725 = 0; i_96725 < (int64_t) 16; i_96725++) {
                    // futhark/microgpt.fut:293:36-48
                    
                    double lifted_lambda_res_96727 = ((double *) mem_100109)[i_96725];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_96728 = r_96726 + lifted_lambda_res_96727;
                    double r_tmp_101260 = zp_res_96728;
                    
                    r_96726 = r_tmp_101260;
                }
                defunc_0_lifted_lambda_res_96724 = r_96726;
                // futhark/microgpt.fut:293:6-49
                
                double zs_res_96729 = 1.0 / defunc_0_lifted_lambda_res_96724;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98482 = 0; i_98482 < (int64_t) 16; i_98482++) {
                    // futhark/microgpt.fut:293:56-68
                    
                    double zt_rhs_96736 = ((double *) mem_100109)[i_98482];
                    
                    // futhark/microgpt.fut:293:10-68
                    
                    double zt_res_96737 = zs_res_96729 * zt_rhs_96736;
                    
                    ((double *) mem_100116)[i_98482] = zt_res_96737;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_96791 = ((double *) mem_100017)[i_98500 * (int64_t) 256 + i_98490 * (int64_t) 16 + defunc_0_reduce_res_97898];
                
                ((double *) mem_100088)[i_98490] = lifted_lambda_res_96791;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100089, i_98490 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100102, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100090, i_98490 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100116, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100071, i_98500 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100088, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100072, i_98500 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100089, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100073, i_98500 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100090, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98523 = 0; i_98523 < (int64_t) 4; i_98523++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98516 = 0; i_98516 < (int64_t) 16; i_98516++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_96986 = ((double *) mem_100071)[i_98523 * (int64_t) 16 + i_98516];
                
                // futhark/microgpt.fut:296:88-114
                
                double neg_res_96987 = -neg_arg0_96986;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98506 = 0; i_98506 < (int64_t) 16; i_98506++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_96994 = ((double *) mem_100017)[i_98523 * (int64_t) 256 + i_98516 * (int64_t) 16 + i_98506];
                    
                    // futhark/microgpt.fut:296:54-114
                    
                    double zp_res_96995 = neg_res_96987 + zp_lhs_96994;
                    
                    ((double *) mem_100170)[i_98506] = zp_res_96995;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98510 = 0; i_98510 < (int64_t) 4; i_98510++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_97009;
                    double r_97011 = 0.0;
                    
                    for (int64_t i_97010 = 0; i_97010 < (int64_t) 16; i_97010++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_97012 = ((double *) mem_100073)[i_98523 * (int64_t) 256 + i_97010 * (int64_t) 16 + i_98516];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_97013 = ((double *) mem_100018)[i_98523 * (int64_t) 64 + i_97010 * (int64_t) 4 + i_98510];
                        
                        // futhark/microgpt.fut:306:75-136
                        
                        double zt_res_97014 = zt_lhs_97012 * zt_rhs_97013;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_97015 = r_97011 + zt_res_97014;
                        double r_tmp_101268 = zp_res_97015;
                        
                        r_97011 = r_tmp_101268;
                    }
                    defunc_0_lifted_lambda_res_97009 = r_97011;
                    ((double *) mem_100177)[i_98510] = defunc_0_lifted_lambda_res_97009;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100160, i_98516 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100177, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100161, i_98516 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100148, i_98523 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_100160, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100149, i_98523 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100161, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98536 = 0; i_98536 < (int64_t) 4; i_98536++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98532 = 0; i_98532 < (int64_t) 16; i_98532++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98528 = 0; i_98528 < (int64_t) 16; i_98528++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double exp_arg0_91122 = ((double *) mem_100149)[i_98536 * (int64_t) 256 + i_98532 * (int64_t) 16 + i_98528];
                    
                    // futhark/microgpt.fut:297:54-88
                    
                    double exp_res_91123 = futrts_exp64(exp_arg0_91122);
                    
                    ((double *) mem_100213)[i_98528] = exp_res_91123;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100208, i_98532 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100213, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100202, i_98536 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100208, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98544 = 0; i_98544 < (int64_t) 4; i_98544++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98540 = 0; i_98540 < (int64_t) 16; i_98540++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91139;
                double r_91141 = 0.0;
                
                for (int64_t i_91140 = 0; i_91140 < (int64_t) 16; i_91140++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_91142 = ((double *) mem_100202)[i_98544 * (int64_t) 256 + i_98540 * (int64_t) 16 + i_91140];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91143 = r_91141 + lifted_lambda_res_91142;
                    double r_tmp_101274 = zp_res_91143;
                    
                    r_91141 = r_tmp_101274;
                }
                defunc_0_lifted_lambda_res_91139 = r_91141;
                ((double *) mem_100234)[i_98540] = defunc_0_lifted_lambda_res_91139;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100229, i_98544 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100234, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98552 = 0; i_98552 < (int64_t) 4; i_98552++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98548 = 0; i_98548 < (int64_t) 16; i_98548++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_91158 = ((double *) mem_100229)[i_98552 * (int64_t) 16 + i_98548];
                
                // futhark/microgpt.fut:299:52-80
                
                double zs_res_91159 = 1.0 / zs_rhs_91158;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91160;
                double r_91162 = 0.0;
                
                for (int64_t i_91161 = 0; i_91161 < (int64_t) 16; i_91161++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91163 = ((double *) mem_100202)[i_98552 * (int64_t) 256 + i_98548 * (int64_t) 16 + i_91161];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91164 = ((double *) mem_100072)[i_98552 * (int64_t) 256 + i_98548 * (int64_t) 16 + i_91161];
                    
                    // futhark/microgpt.fut:299:145-206
                    
                    double zt_res_91165 = zt_lhs_91163 * zt_rhs_91164;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91166 = r_91162 + zt_res_91165;
                    double r_tmp_101277 = zp_res_91166;
                    
                    r_91162 = r_tmp_101277;
                }
                defunc_0_lifted_lambda_res_91160 = r_91162;
                // futhark/microgpt.fut:299:92-208
                
                double zt_res_91167 = zs_res_91159 * defunc_0_lifted_lambda_res_91160;
                
                // futhark/microgpt.fut:299:56-208
                
                double zt_res_91168 = zs_res_91159 * zt_res_91167;
                
                // futhark/microgpt.fut:299:44-208
                
                double neg_res_91169 = -zt_res_91168;
                
                ((double *) mem_100250)[i_98548] = neg_res_91169;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100245, i_98552 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100250, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98564 = 0; i_98564 < (int64_t) 4; i_98564++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98560 = 0; i_98560 < (int64_t) 16; i_98560++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_91184 = ((double *) mem_100229)[i_98564 * (int64_t) 16 + i_98560];
                
                // futhark/microgpt.fut:300:56-84
                
                double zs_res_91185 = 1.0 / zs_rhs_91184;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_91186 = ((double *) mem_100245)[i_98564 * (int64_t) 16 + i_98560];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98556 = 0; i_98556 < (int64_t) 16; i_98556++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_91193 = ((double *) mem_100072)[i_98564 * (int64_t) 256 + i_98560 * (int64_t) 16 + i_98556];
                    
                    // futhark/microgpt.fut:300:60-118
                    
                    double zt_res_91194 = zs_res_91185 * zt_rhs_91193;
                    
                    // futhark/microgpt.fut:300:86-144
                    
                    double zp_res_91195 = zp_rhs_91186 + zt_res_91194;
                    
                    ((double *) mem_100272)[i_98556] = zp_res_91195;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100267, i_98560 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100272, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100261, i_98564 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100267, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98576 = 0; i_98576 < (int64_t) 4; i_98576++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98572 = 0; i_98572 < (int64_t) 16; i_98572++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98568 = 0; i_98568 < (int64_t) 16; i_98568++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double exp_arg0_91217 = ((double *) mem_100149)[i_98576 * (int64_t) 256 + i_98572 * (int64_t) 16 + i_98568];
                    
                    // futhark/microgpt.fut:301:55-89
                    
                    double exp_res_91218 = futrts_exp64(exp_arg0_91217);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_91219 = ((double *) mem_100261)[i_98576 * (int64_t) 256 + i_98572 * (int64_t) 16 + i_98568];
                    
                    // futhark/microgpt.fut:301:55-123
                    
                    double zt_res_91220 = exp_res_91218 * zt_rhs_91219;
                    
                    ((double *) mem_100299)[i_98568] = zt_res_91220;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100294, i_98572 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100299, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100288, i_98576 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100294, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98584 = 0; i_98584 < (int64_t) 4; i_98584++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98580 = 0; i_98580 < (int64_t) 16; i_98580++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91236;
                double r_91238 = 0.0;
                
                for (int64_t i_91237 = 0; i_91237 < (int64_t) 16; i_91237++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_91239 = ((double *) mem_100288)[i_98584 * (int64_t) 256 + i_98580 * (int64_t) 16 + i_91237];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91240 = r_91238 + lifted_lambda_res_91239;
                    double r_tmp_101286 = zp_res_91240;
                    
                    r_91238 = r_tmp_101286;
                }
                defunc_0_lifted_lambda_res_91236 = r_91238;
                // futhark/microgpt.fut:302:44-101
                
                double neg_res_91241 = -defunc_0_lifted_lambda_res_91236;
                
                ((double *) mem_100320)[i_98580] = neg_res_91241;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100315, i_98584 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98599 = 0; i_98599 < (int64_t) 4; i_98599++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98595 = 0; i_98595 < (int64_t) 16; i_98595++) {
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_97927;
                int64_t defunc_0_reduce_res_97928;
                double redout_98586;
                int64_t redout_98587;
                
                redout_98586 = -INFINITY;
                redout_98587 = (int64_t) 16;
                for (int64_t i_98588 = 0; i_98588 < (int64_t) 16; i_98588++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_97049 = ((double *) mem_100017)[i_98599 * (int64_t) 256 + i_98595 * (int64_t) 16 + i_98588];
                    
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_91272 = lifted_lambda_res_97049 < redout_98586;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_91273;
                    
                    if (zg_res_91272) {
                        lifted_lambda_res_91273 = redout_98586;
                    } else {
                        lifted_lambda_res_91273 = lifted_lambda_res_97049;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_91274;
                    
                    if (zg_res_91272) {
                        lifted_lambda_res_91274 = redout_98587;
                    } else {
                        lifted_lambda_res_91274 = i_98588;
                    }
                    
                    double redout_tmp_101289 = lifted_lambda_res_91273;
                    int64_t redout_tmp_101290 = lifted_lambda_res_91274;
                    
                    redout_98586 = redout_tmp_101289;
                    redout_98587 = redout_tmp_101290;
                }
                defunc_0_reduce_res_97927 = redout_98586;
                defunc_0_reduce_res_97928 = redout_98587;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98591 = 0; i_98591 < (int64_t) 16; i_98591++) {
                    // futhark/microgpt.fut:303:54-163
                    
                    bool cond_91277 = i_98591 == defunc_0_reduce_res_97928;
                    
                    // futhark/microgpt.fut:303:54-163
                    
                    double lifted_lambda_res_91278;
                    
                    if (cond_91277) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double lifted_lambda_res_t_res_97926 = ((double *) mem_100315)[i_98599 * (int64_t) 16 + i_98595];
                        
                        lifted_lambda_res_91278 = lifted_lambda_res_t_res_97926;
                    } else {
                        lifted_lambda_res_91278 = 0.0;
                    }
                    ((double *) mem_100342)[i_98591] = lifted_lambda_res_91278;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100337, i_98595 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100342, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100331, i_98599 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100337, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98611 = 0; i_98611 < (int64_t) 4; i_98611++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98607 = 0; i_98607 < (int64_t) 16; i_98607++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98603 = 0; i_98603 < (int64_t) 16; i_98603++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_91309 = ((double *) mem_100288)[i_98611 * (int64_t) 256 + i_98607 * (int64_t) 16 + i_98603];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_rhs_91310 = ((double *) mem_100331)[i_98611 * (int64_t) 256 + i_98607 * (int64_t) 16 + i_98603];
                    
                    // futhark/microgpt.fut:304:54-115
                    
                    double zp_res_91311 = zp_lhs_91309 + zp_rhs_91310;
                    
                    ((double *) mem_100369)[i_98603] = zp_res_91311;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100364, i_98607 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100369, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100358, i_98611 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100364, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98623 = 0; i_98623 < (int64_t) 4; i_98623++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98619 = 0; i_98619 < (int64_t) 16; i_98619++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98615 = 0; i_98615 < (int64_t) 16; i_98615++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_91333 = ((double *) mem_100358)[i_98623 * (int64_t) 256 + i_98619 * (int64_t) 16 + i_98615];
                    
                    // futhark/microgpt.fut:305:54-96
                    
                    double zs_res_91334 = zs_lhs_91333 / 2.0;
                    
                    ((double *) mem_100396)[i_98615] = zs_res_91334;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100391, i_98619 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100396, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100385, i_98623 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100391, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98643 = 0; i_98643 < (int64_t) 4; i_98643++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98636 = 0; i_98636 < (int64_t) 16; i_98636++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_98629 = 0; i_98629 < (int64_t) 4; i_98629++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_97159;
                    double r_97161 = 0.0;
                    
                    for (int64_t i_97160 = 0; i_97160 < (int64_t) 16; i_97160++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_97162 = ((double *) mem_99390)[i_98643 * (int64_t) 64 + i_97160 * (int64_t) 4 + i_98629];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_97163 = ((double *) mem_100385)[i_98643 * (int64_t) 256 + i_97160 * (int64_t) 16 + i_98636];
                        
                        // futhark/microgpt.fut:307:75-135
                        
                        double zt_res_97164 = zt_lhs_97162 * zt_rhs_97163;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_97165 = r_97161 + zt_res_97164;
                        double r_tmp_101304 = zp_res_97165;
                        
                        r_97161 = r_tmp_101304;
                    }
                    defunc_0_lifted_lambda_res_97159 = r_97161;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_97172;
                    double r_97174 = 0.0;
                    
                    for (int64_t i_97173 = 0; i_97173 < (int64_t) 16; i_97173++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_97175 = ((double *) mem_100385)[i_98643 * (int64_t) 256 + i_98636 * (int64_t) 16 + i_97173];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_97176 = ((double *) mem_99389)[i_98643 * (int64_t) 64 + i_97173 * (int64_t) 4 + i_98629];
                        
                        // futhark/microgpt.fut:308:75-135
                        
                        double zt_res_97177 = zt_lhs_97175 * zt_rhs_97176;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_97178 = r_97174 + zt_res_97177;
                        double r_tmp_101305 = zp_res_97178;
                        
                        r_97174 = r_tmp_101305;
                    }
                    defunc_0_lifted_lambda_res_97172 = r_97174;
                    ((double *) mem_100434)[i_98629] = defunc_0_lifted_lambda_res_97172;
                    ((double *) mem_100435)[i_98629] = defunc_0_lifted_lambda_res_97159;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100424, i_98636 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100434, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_100425, i_98636 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100435, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100412, i_98643 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_100424, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_100413, i_98643 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_100425, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98662 = 0; i_98662 < (int64_t) 16; i_98662++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98652 = 0; i_98652 < (int64_t) 16; i_98652++) {
                // futhark/microgpt.fut:309:57-60
                
                int64_t tmp_97241 = sdiv64(i_98652, (int64_t) 4);
                
                // futhark/microgpt.fut:309:44-62
                
                bool x_97242 = sle64((int64_t) 0, tmp_97241);
                
                // futhark/microgpt.fut:309:44-62
                
                bool y_97243 = slt64(tmp_97241, (int64_t) 4);
                
                // futhark/microgpt.fut:309:44-62
                
                bool bounds_check_97244 = x_97242 && y_97243;
                
                // futhark/microgpt.fut:309:44-62
                
                bool index_certs_97245;
                
                if (!bounds_check_97244) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_97241, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:309:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:309:13-85\n   #6  futhark/microgpt.fut:453:5-76\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:309:79-82
                
                int64_t tmp_97246 = smod64(i_98652, (int64_t) 4);
                
                // futhark/microgpt.fut:309:44-84
                
                bool x_97247 = sle64((int64_t) 0, tmp_97246);
                
                // futhark/microgpt.fut:309:44-84
                
                bool y_97248 = slt64(tmp_97246, (int64_t) 4);
                
                // futhark/microgpt.fut:309:44-84
                
                bool bounds_check_97249 = x_97247 && y_97248;
                
                // futhark/microgpt.fut:309:44-84
                
                bool index_certs_97250;
                
                if (!bounds_check_97249) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_97246, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:309:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:309:13-85\n   #6  futhark/microgpt.fut:453:5-76\n   #7  futhark/microgpt.fut:458:26-464:31\n   #8  futhark/microgpt.fut:480:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_97251 = ((double *) mem_100148)[tmp_97241 * (int64_t) 64 + i_98662 * (int64_t) 4 + tmp_97246];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_97264 = ((double *) mem_100413)[tmp_97241 * (int64_t) 64 + i_98662 * (int64_t) 4 + tmp_97246];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_97280 = ((double *) mem_100412)[tmp_97241 * (int64_t) 64 + i_98662 * (int64_t) 4 + tmp_97246];
                
                ((double *) mem_100481)[i_98652] = lifted_lambda_res_97280;
                ((double *) mem_100482)[i_98652] = lifted_lambda_res_97264;
                ((double *) mem_100483)[i_98652] = lifted_lambda_res_97251;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100466, i_98662 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100481, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100467, i_98662 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100482, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100468, i_98662 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100483, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98687 = 0; i_98687 < (int64_t) 16; i_98687++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98674 = 0; i_98674 < (int64_t) 16; i_98674++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97443;
                double r_97445 = 0.0;
                
                for (int64_t i_97444 = 0; i_97444 < (int64_t) 16; i_97444++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97446 = ((double *) mem_param_99114.mem)[i_97444 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97447 = ((double *) mem_100468)[i_98687 * (int64_t) 16 + i_97444];
                    
                    // futhark/microgpt.fut:312:69-114
                    
                    double zt_res_97448 = zt_lhs_97446 * zt_rhs_97447;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97449 = r_97445 + zt_res_97448;
                    double r_tmp_101320 = zp_res_97449;
                    
                    r_97445 = r_tmp_101320;
                }
                defunc_0_lifted_lambda_res_97443 = r_97445;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97450;
                double r_97452 = 0.0;
                
                for (int64_t i_97451 = 0; i_97451 < (int64_t) 16; i_97451++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97453 = ((double *) mem_param_99090.mem)[i_97451 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97454 = ((double *) mem_100467)[i_98687 * (int64_t) 16 + i_97451];
                    
                    // futhark/microgpt.fut:312:145-190
                    
                    double zt_res_97455 = zt_lhs_97453 * zt_rhs_97454;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97456 = r_97452 + zt_res_97455;
                    double r_tmp_101321 = zp_res_97456;
                    
                    r_97452 = r_tmp_101321;
                }
                defunc_0_lifted_lambda_res_97450 = r_97452;
                // futhark/microgpt.fut:312:47-192
                
                double zp_res_97457 = defunc_0_lifted_lambda_res_97443 + defunc_0_lifted_lambda_res_97450;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97458;
                double r_97460 = 0.0;
                
                for (int64_t i_97459 = 0; i_97459 < (int64_t) 16; i_97459++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97461 = ((double *) mem_param_99102.mem)[i_97459 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97462 = ((double *) mem_100466)[i_98687 * (int64_t) 16 + i_97459];
                    
                    // futhark/microgpt.fut:312:222-267
                    
                    double zt_res_97463 = zt_lhs_97461 * zt_rhs_97462;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97464 = r_97460 + zt_res_97463;
                    double r_tmp_101322 = zp_res_97464;
                    
                    r_97460 = r_tmp_101322;
                }
                defunc_0_lifted_lambda_res_97458 = r_97460;
                // futhark/microgpt.fut:312:118-269
                
                double zp_res_97465 = zp_res_97457 + defunc_0_lifted_lambda_res_97458;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97472;
                double r_97474 = 0.0;
                
                for (int64_t i_97473 = 0; i_97473 < (int64_t) 16; i_97473++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97475 = ((double *) mem_100466)[i_97473 * (int64_t) 16 + i_98687];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97476 = ((double *) mem_99289)[i_97473 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:330:68-111
                    
                    double zt_res_97477 = zt_lhs_97475 * zt_rhs_97476;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97478 = r_97474 + zt_res_97477;
                    double r_tmp_101323 = zp_res_97478;
                    
                    r_97474 = r_tmp_101323;
                }
                defunc_0_lifted_lambda_res_97472 = r_97474;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97488;
                double r_97490 = 0.0;
                
                for (int64_t i_97489 = 0; i_97489 < (int64_t) 16; i_97489++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97491 = ((double *) mem_100467)[i_97489 * (int64_t) 16 + i_98687];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97492 = ((double *) mem_99289)[i_97489 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:331:68-111
                    
                    double zt_res_97493 = zt_lhs_97491 * zt_rhs_97492;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97494 = r_97490 + zt_res_97493;
                    double r_tmp_101324 = zp_res_97494;
                    
                    r_97490 = r_tmp_101324;
                }
                defunc_0_lifted_lambda_res_97488 = r_97490;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97506;
                double r_97508 = 0.0;
                
                for (int64_t i_97507 = 0; i_97507 < (int64_t) 16; i_97507++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97509 = ((double *) mem_100468)[i_97507 * (int64_t) 16 + i_98687];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97510 = ((double *) mem_99289)[i_97507 * (int64_t) 16 + i_98674];
                    
                    // futhark/microgpt.fut:332:68-111
                    
                    double zt_res_97511 = zt_lhs_97509 * zt_rhs_97510;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97512 = r_97508 + zt_res_97511;
                    double r_tmp_101325 = zp_res_97512;
                    
                    r_97508 = r_tmp_101325;
                }
                defunc_0_lifted_lambda_res_97506 = r_97508;
                ((double *) mem_100534)[i_98674] = defunc_0_lifted_lambda_res_97506;
                ((double *) mem_100535)[i_98674] = defunc_0_lifted_lambda_res_97488;
                ((double *) mem_100536)[i_98674] = defunc_0_lifted_lambda_res_97472;
                ((double *) mem_100537)[i_98674] = zp_res_97465;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100514, i_98687 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100534, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100515, i_98687 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100535, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100516, i_98687 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100536, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100517, i_98687 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98694 = 0; i_98694 < (int64_t) 16; i_98694++) {
            // futhark/microgpt.fut:316:51-63
            
            double zs_rhs_91567 = ((double *) mem_99546)[i_98694];
            
            // futhark/microgpt.fut:316:43-63
            
            double zs_res_91568 = 1.0 / zs_rhs_91567;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91569;
            double r_91571 = 0.0;
            
            for (int64_t i_91570 = 0; i_91570 < (int64_t) 16; i_91570++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91572 = ((double *) mem_99256)[i_98694 * (int64_t) 16 + i_91570];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91573 = ((double *) mem_100517)[i_98694 * (int64_t) 16 + i_91570];
                
                // futhark/microgpt.fut:316:120-163
                
                double zt_res_91574 = zt_lhs_91572 * zt_rhs_91573;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91575 = r_91571 + zt_res_91574;
                double r_tmp_101327 = zp_res_91575;
                
                r_91571 = r_tmp_101327;
            }
            defunc_0_lifted_lambda_res_91569 = r_91571;
            // futhark/microgpt.fut:316:75-165
            
            double zt_res_91576 = zs_res_91568 * defunc_0_lifted_lambda_res_91569;
            
            // futhark/microgpt.fut:316:47-165
            
            double zt_res_91577 = zs_res_91568 * zt_res_91576;
            
            // futhark/microgpt.fut:316:35-165
            
            double neg_res_91578 = -zt_res_91577;
            
            ((double *) mem_100578)[i_98694] = neg_res_91578;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98698 = 0; i_98698 < (int64_t) 16; i_98698++) {
            // futhark/microgpt.fut:317:72-84
            
            double zp_lhs_91586 = ((double *) mem_99327)[i_98698];
            
            // futhark/microgpt.fut:317:72-112
            
            double zp_res_91587 = 1.0e-5 + zp_lhs_91586;
            
            // futhark/microgpt.fut:317:64-112
            
            double sqrt_res_91588 = futrts_sqrt64(zp_res_91587);
            
            // futhark/microgpt.fut:317:50-114
            
            double zt_res_91589 = 2.0 * sqrt_res_91588;
            
            // futhark/microgpt.fut:317:36-114
            
            double zs_res_91590 = 1.0 / zt_res_91589;
            
            // futhark/microgpt.fut:317:122-134
            
            double zt_rhs_91591 = ((double *) mem_100578)[i_98698];
            
            // futhark/microgpt.fut:317:40-134
            
            double zt_res_91592 = zs_res_91590 * zt_rhs_91591;
            
            ((double *) mem_100585)[i_98698] = zt_res_91592;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98702 = 0; i_98702 < (int64_t) 16; i_98702++) {
            // futhark/microgpt.fut:318:45-57
            
            double zs_lhs_91600 = ((double *) mem_100585)[i_98702];
            
            // futhark/microgpt.fut:318:45-72
            
            double zs_res_91601 = zs_lhs_91600 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_101330 = 0; nest_i_101330 < (int64_t) 16; nest_i_101330++) {
                ((double *) mem_100592)[i_98702 * (int64_t) 16 + nest_i_101330] = zs_res_91601;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98710 = 0; i_98710 < (int64_t) 16; i_98710++) {
            // futhark/microgpt.fut:319:82-94
            
            double zs_rhs_91610 = ((double *) mem_99546)[i_98710];
            
            // futhark/microgpt.fut:319:74-94
            
            double zs_res_91611 = 1.0 / zs_rhs_91610;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98706 = 0; i_98706 < (int64_t) 16; i_98706++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_91618 = ((double *) mem_99969)[i_98710 * (int64_t) 16 + i_98706];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91619 = ((double *) mem_100517)[i_98710 * (int64_t) 16 + i_98706];
                
                // futhark/microgpt.fut:319:78-120
                
                double zt_res_91620 = zs_res_91611 * zt_rhs_91619;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_91621 = ((double *) mem_99256)[i_98710 * (int64_t) 16 + i_98706];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91622 = ((double *) mem_100592)[i_98710 * (int64_t) 16 + i_98706];
                
                // futhark/microgpt.fut:319:127-170
                
                double zt_res_91623 = zt_lhs_91621 * zt_rhs_91622;
                
                // futhark/microgpt.fut:319:96-170
                
                double zp_res_91624 = zt_res_91620 + zt_res_91623;
                
                // futhark/microgpt.fut:319:122-221
                
                double zp_res_91625 = zt_res_91623 + zp_res_91624;
                
                // futhark/microgpt.fut:319:45-221
                
                double zp_res_91626 = zp_lhs_91618 + zp_res_91625;
                
                ((double *) mem_100607)[i_98706] = zp_res_91626;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100602, i_98710 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100607, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98714 = 0; i_98714 < (int64_t) 16; i_98714++) {
            // futhark/microgpt.fut:323:51-63
            
            double zs_rhs_91674 = ((double *) mem_99326)[i_98714];
            
            // futhark/microgpt.fut:323:43-63
            
            double zs_res_91675 = 1.0 / zs_rhs_91674;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91676;
            double r_91678 = 0.0;
            
            for (int64_t i_91677 = 0; i_91677 < (int64_t) 16; i_91677++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91679 = ((double *) mem_99224)[i_98714 * (int64_t) 16 + i_91677];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91680 = ((double *) mem_100602)[i_98714 * (int64_t) 16 + i_91677];
                
                // futhark/microgpt.fut:323:120-163
                
                double zt_res_91681 = zt_lhs_91679 * zt_rhs_91680;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91682 = r_91678 + zt_res_91681;
                double r_tmp_101334 = zp_res_91682;
                
                r_91678 = r_tmp_101334;
            }
            defunc_0_lifted_lambda_res_91676 = r_91678;
            // futhark/microgpt.fut:323:75-165
            
            double zt_res_91683 = zs_res_91675 * defunc_0_lifted_lambda_res_91676;
            
            // futhark/microgpt.fut:323:47-165
            
            double zt_res_91684 = zs_res_91675 * zt_res_91683;
            
            // futhark/microgpt.fut:323:35-165
            
            double neg_res_91685 = -zt_res_91684;
            
            ((double *) mem_100618)[i_98714] = neg_res_91685;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98718 = 0; i_98718 < (int64_t) 16; i_98718++) {
            // futhark/microgpt.fut:324:72-84
            
            double zp_lhs_91693 = ((double *) mem_99287)[i_98718];
            
            // futhark/microgpt.fut:324:72-112
            
            double zp_res_91694 = 1.0e-5 + zp_lhs_91693;
            
            // futhark/microgpt.fut:324:64-112
            
            double sqrt_res_91695 = futrts_sqrt64(zp_res_91694);
            
            // futhark/microgpt.fut:324:50-114
            
            double zt_res_91696 = 2.0 * sqrt_res_91695;
            
            // futhark/microgpt.fut:324:36-114
            
            double zs_res_91697 = 1.0 / zt_res_91696;
            
            // futhark/microgpt.fut:324:122-134
            
            double zt_rhs_91698 = ((double *) mem_100618)[i_98718];
            
            // futhark/microgpt.fut:324:40-134
            
            double zt_res_91699 = zs_res_91697 * zt_rhs_91698;
            
            ((double *) mem_100625)[i_98718] = zt_res_91699;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98722 = 0; i_98722 < (int64_t) 16; i_98722++) {
            // futhark/microgpt.fut:325:45-57
            
            double zs_lhs_91707 = ((double *) mem_100625)[i_98722];
            
            // futhark/microgpt.fut:325:45-72
            
            double zs_res_91708 = zs_lhs_91707 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_101337 = 0; nest_i_101337 < (int64_t) 16; nest_i_101337++) {
                ((double *) mem_100632)[i_98722 * (int64_t) 16 + nest_i_101337] = zs_res_91708;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98730 = 0; i_98730 < (int64_t) 16; i_98730++) {
            // futhark/microgpt.fut:326:56-68
            
            double zs_rhs_91717 = ((double *) mem_99326)[i_98730];
            
            // futhark/microgpt.fut:326:48-68
            
            double zs_res_91718 = 1.0 / zs_rhs_91717;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98726 = 0; i_98726 < (int64_t) 16; i_98726++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91725 = ((double *) mem_100602)[i_98730 * (int64_t) 16 + i_98726];
                
                // futhark/microgpt.fut:326:52-94
                
                double zt_res_91726 = zs_res_91718 * zt_rhs_91725;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_91727 = ((double *) mem_99224)[i_98730 * (int64_t) 16 + i_98726];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_91728 = ((double *) mem_100632)[i_98730 * (int64_t) 16 + i_98726];
                
                // futhark/microgpt.fut:326:101-144
                
                double zt_res_91729 = zt_lhs_91727 * zt_rhs_91728;
                
                // futhark/microgpt.fut:326:70-144
                
                double zp_res_91730 = zt_res_91726 + zt_res_91729;
                
                // futhark/microgpt.fut:326:96-195
                
                double zp_res_91731 = zt_res_91729 + zp_res_91730;
                
                ((double *) mem_100647)[i_98726] = zp_res_91731;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100642, i_98730 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100647, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98743 = 0; i_98743 < (int64_t) 16; i_98743++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98736 = 0; i_98736 < (int64_t) 16; i_98736++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_97538 = ((double *) mem_100642)[i_98743 * (int64_t) 16 + i_98736];
                
                ((double *) mem_100668)[i_98736] = lifted_lambda_res_97538;
                ((double *) mem_100669)[i_98736] = lifted_lambda_res_97538;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100658, i_98743 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100659, i_98743 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98752 = 0; i_98752 < (int64_t) 64; i_98752++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98748 = 0; i_98748 < (int64_t) 16; i_98748++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91845;
                double r_91847 = 0.0;
                
                for (int64_t i_91846 = 0; i_91846 < (int64_t) 16; i_91846++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91848 = ((double *) mem_99913)[i_91846 * (int64_t) 64 + i_98752];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91849 = ((double *) mem_99602)[i_91846 * (int64_t) 16 + i_98748];
                    
                    // futhark/microgpt.fut:334:67-111
                    
                    double zt_res_91850 = zt_lhs_91848 * zt_rhs_91849;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91851 = r_91847 + zt_res_91850;
                    double r_tmp_101346 = zp_res_91851;
                    
                    r_91847 = r_tmp_101346;
                }
                defunc_0_lifted_lambda_res_91845 = r_91847;
                ((double *) mem_100695)[i_98748] = defunc_0_lifted_lambda_res_91845;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100690, i_98752 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_98765 = 0; i_98765 < (int64_t) 27; i_98765++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_98758 = 0; i_98758 < (int64_t) 16; i_98758++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97566;
                double r_97568 = 0.0;
                
                for (int64_t i_97567 = 0; i_97567 < (int64_t) 16; i_97567++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_97569 = ((double *) mem_99849)[i_97567 * (int64_t) 27 + i_98765];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_97570 = ((double *) mem_99695)[i_97567 * (int64_t) 16 + i_98758];
                    
                    // futhark/microgpt.fut:336:68-111
                    
                    double zt_res_97571 = zt_lhs_97569 * zt_rhs_97570;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97572 = r_97568 + zt_res_97571;
                    double r_tmp_101351 = zp_res_97572;
                    
                    r_97568 = r_tmp_101351;
                }
                defunc_0_lifted_lambda_res_97566 = r_97568;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_97575;
                double r_97577 = 0.0;
                
                for (int64_t i_97576 = 0; i_97576 < (int64_t) 16; i_97576++) {
                    int64_t zeze_lhs_97578 = ((int64_t *) seqs_mem_99082.mem)[step_89818 * (int64_t) 16 + i_97576];
                    
                    // futhark/microgpt.fut:454:58-109
                    
                    bool cond_97579 = zeze_lhs_97578 == i_98765;
                    
                    // futhark/microgpt.fut:454:58-109
                    
                    double lifted_lambda_res_97580;
                    
                    if (cond_97579) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_97955 = ((double *) mem_100658)[i_97576 * (int64_t) 16 + i_98758];
                        
                        lifted_lambda_res_97580 = lifted_lambda_res_t_res_97955;
                    } else {
                        lifted_lambda_res_97580 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_97586 = r_97577 + lifted_lambda_res_97580;
                    double r_tmp_101352 = zp_res_97586;
                    
                    r_97577 = r_tmp_101352;
                }
                defunc_0_lifted_lambda_res_97575 = r_97577;
                ((double *) mem_100716)[i_98758] = defunc_0_lifted_lambda_res_97575;
                ((double *) mem_100717)[i_98758] = defunc_0_lifted_lambda_res_97566;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100706, i_98765 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100716, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_100707, i_98765 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_100717, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_91929 = sitofp_i64_f64(step_89818);
        
        // futhark/microgpt.fut:410:46-65
        
        double zm_rhs_91930 = i64_res_91929 / 500.0;
        
        // futhark/microgpt.fut:410:24-65
        
        double zt_rhs_91931 = 1.0 - zm_rhs_91930;
        
        // futhark/microgpt.fut:410:19-65
        
        double lt_r_91932 = 1.0e-2 * zt_rhs_91931;
        
        // futhark/microgpt.fut:412:5-52
        if (memblock_alloc(ctx, &mem_100738, (int64_t) 3456, "mem_100738")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-52
        // futhark/microgpt.fut:412:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100738.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99106.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:412:5-52
        if (memblock_alloc(ctx, &mem_100740, (int64_t) 3456, "mem_100740")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-52
        // futhark/microgpt.fut:412:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100740.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99142.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:412:5-52
        if (memblock_alloc(ctx, &mem_100742, (int64_t) 3456, "mem_100742")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-52
        // futhark/microgpt.fut:412:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100742.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99178.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:412:5-52
        if (memblock_alloc(ctx, &mem_100744, (int64_t) 3456, "mem_100744")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:412:5-52
        // futhark/microgpt.fut:412:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100744.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100706, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:412:5-52
        if (futrts_adam_opt_w_10560(ctx, &ext_mem_100748, &ext_mem_100747, &ext_mem_100746, mem_100738, mem_100740, mem_100742, mem_100744, (int64_t) 27, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100738, "mem_100738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100740, "mem_100740") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100742, "mem_100742") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100744, "mem_100744") != 0)
            return 1;
        // futhark/microgpt.fut:414:5-52
        if (memblock_alloc(ctx, &mem_100749, (int64_t) 2048, "mem_100749")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-52
        // futhark/microgpt.fut:414:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100749.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99098.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-52
        if (memblock_alloc(ctx, &mem_100751, (int64_t) 2048, "mem_100751")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-52
        // futhark/microgpt.fut:414:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100751.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99134.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-52
        if (memblock_alloc(ctx, &mem_100753, (int64_t) 2048, "mem_100753")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-52
        // futhark/microgpt.fut:414:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100753.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99170.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-52
        if (memblock_alloc(ctx, &mem_100755, (int64_t) 2048, "mem_100755")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:414:5-52
        // futhark/microgpt.fut:414:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100755.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100659, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:414:5-52
        if (futrts_adam_opt_w_10561(ctx, &ext_mem_100759, &ext_mem_100758, &ext_mem_100757, mem_100749, mem_100751, mem_100753, mem_100755, (int64_t) 16, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100749, "mem_100749") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100751, "mem_100751") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100753, "mem_100753") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100755, "mem_100755") != 0)
            return 1;
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_100760, (int64_t) 2048, "mem_100760")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100760.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99102.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_100762, (int64_t) 2048, "mem_100762")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100762.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99138.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_100764, (int64_t) 2048, "mem_100764")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100764.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99174.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (memblock_alloc(ctx, &mem_100766, (int64_t) 2048, "mem_100766")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:416:5-56
        // futhark/microgpt.fut:416:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100766.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100516, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:416:5-56
        if (futrts_adam_opt_w_10561(ctx, &ext_mem_100770, &ext_mem_100769, &ext_mem_100768, mem_100760, mem_100762, mem_100764, mem_100766, (int64_t) 16, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100760, "mem_100760") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100762, "mem_100762") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100764, "mem_100764") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100766, "mem_100766") != 0)
            return 1;
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_100771, (int64_t) 2048, "mem_100771")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100771.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99090.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_100773, (int64_t) 2048, "mem_100773")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100773.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99126.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_100775, (int64_t) 2048, "mem_100775")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100775.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99162.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (memblock_alloc(ctx, &mem_100777, (int64_t) 2048, "mem_100777")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:418:5-56
        // futhark/microgpt.fut:418:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100777.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100515, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:418:5-56
        if (futrts_adam_opt_w_10561(ctx, &ext_mem_100781, &ext_mem_100780, &ext_mem_100779, mem_100771, mem_100773, mem_100775, mem_100777, (int64_t) 16, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100771, "mem_100771") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100773, "mem_100773") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100775, "mem_100775") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100777, "mem_100777") != 0)
            return 1;
        // futhark/microgpt.fut:420:5-56
        if (memblock_alloc(ctx, &mem_100782, (int64_t) 2048, "mem_100782")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-56
        // futhark/microgpt.fut:420:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100782.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99114.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:420:5-56
        if (memblock_alloc(ctx, &mem_100784, (int64_t) 2048, "mem_100784")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-56
        // futhark/microgpt.fut:420:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100784.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99150.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:420:5-56
        if (memblock_alloc(ctx, &mem_100786, (int64_t) 2048, "mem_100786")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-56
        // futhark/microgpt.fut:420:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100786.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99186.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:420:5-56
        if (memblock_alloc(ctx, &mem_100788, (int64_t) 2048, "mem_100788")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:420:5-56
        // futhark/microgpt.fut:420:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100788.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100514, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:420:5-56
        if (futrts_adam_opt_w_10561(ctx, &ext_mem_100792, &ext_mem_100791, &ext_mem_100790, mem_100782, mem_100784, mem_100786, mem_100788, (int64_t) 16, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100782, "mem_100782") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100784, "mem_100784") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100786, "mem_100786") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100788, "mem_100788") != 0)
            return 1;
        // futhark/microgpt.fut:422:5-56
        if (memblock_alloc(ctx, &mem_100793, (int64_t) 2048, "mem_100793")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-56
        // futhark/microgpt.fut:422:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100793.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99094.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:422:5-56
        if (memblock_alloc(ctx, &mem_100795, (int64_t) 2048, "mem_100795")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-56
        // futhark/microgpt.fut:422:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100795.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99130.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:422:5-56
        if (memblock_alloc(ctx, &mem_100797, (int64_t) 2048, "mem_100797")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-56
        // futhark/microgpt.fut:422:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100797.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99166.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:422:5-56
        if (memblock_alloc(ctx, &mem_100799, (int64_t) 2048, "mem_100799")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:422:5-56
        // futhark/microgpt.fut:422:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100799.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_99985, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:422:5-56
        if (futrts_adam_opt_w_10561(ctx, &ext_mem_100803, &ext_mem_100802, &ext_mem_100801, mem_100793, mem_100795, mem_100797, mem_100799, (int64_t) 16, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100793, "mem_100793") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100795, "mem_100795") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100797, "mem_100797") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100799, "mem_100799") != 0)
            return 1;
        // futhark/microgpt.fut:424:5-52
        if (memblock_alloc(ctx, &mem_100804, (int64_t) 8192, "mem_100804")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-52
        // futhark/microgpt.fut:424:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100804.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99110.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:424:5-52
        if (memblock_alloc(ctx, &mem_100806, (int64_t) 8192, "mem_100806")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-52
        // futhark/microgpt.fut:424:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100806.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99146.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:424:5-52
        if (memblock_alloc(ctx, &mem_100808, (int64_t) 8192, "mem_100808")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-52
        // futhark/microgpt.fut:424:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100808.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99182.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:424:5-52
        if (memblock_alloc(ctx, &mem_100810, (int64_t) 8192, "mem_100810")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:424:5-52
        // futhark/microgpt.fut:424:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100810.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100690, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:424:5-52
        if (futrts_adam_opt_w_10560(ctx, &ext_mem_100814, &ext_mem_100813, &ext_mem_100812, mem_100804, mem_100806, mem_100808, mem_100810, (int64_t) 64, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100804, "mem_100804") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100806, "mem_100806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100808, "mem_100808") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100810, "mem_100810") != 0)
            return 1;
        // futhark/microgpt.fut:426:5-60
        if (memblock_alloc(ctx, &mem_100815, (int64_t) 8192, "mem_100815")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:426:5-60
        // futhark/microgpt.fut:426:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100815.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_99086.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:426:5-60
        if (memblock_alloc(ctx, &mem_100817, (int64_t) 8192, "mem_100817")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:426:5-60
        // futhark/microgpt.fut:426:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100817.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_99122.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:426:5-60
        if (memblock_alloc(ctx, &mem_100819, (int64_t) 8192, "mem_100819")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:426:5-60
        // futhark/microgpt.fut:426:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100819.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_99158.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:426:5-60
        if (memblock_alloc(ctx, &mem_100821, (int64_t) 8192, "mem_100821")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:426:5-60
        // futhark/microgpt.fut:426:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100821.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_99881, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:426:5-60
        if (futrts_adam_opt_w_10560(ctx, &ext_mem_100825, &ext_mem_100824, &ext_mem_100823, mem_100815, mem_100817, mem_100819, mem_100821, (int64_t) 16, (int64_t) 64, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100815, "mem_100815") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100817, "mem_100817") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100819, "mem_100819") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100821, "mem_100821") != 0)
            return 1;
        // futhark/microgpt.fut:428:5-56
        if (memblock_alloc(ctx, &mem_100826, (int64_t) 3456, "mem_100826")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-56
        // futhark/microgpt.fut:428:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100826.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99118.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-56
        if (memblock_alloc(ctx, &mem_100828, (int64_t) 3456, "mem_100828")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-56
        // futhark/microgpt.fut:428:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100828.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99154.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-56
        if (memblock_alloc(ctx, &mem_100830, (int64_t) 3456, "mem_100830")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-56
        // futhark/microgpt.fut:428:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100830.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_99190.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-56
        if (memblock_alloc(ctx, &mem_100832, (int64_t) 3456, "mem_100832")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-56
        // futhark/microgpt.fut:428:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_100832.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_100707, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-56
        if (futrts_adam_opt_w_10560(ctx, &ext_mem_100836, &ext_mem_100835, &ext_mem_100834, mem_100826, mem_100828, mem_100830, mem_100832, (int64_t) 27, (int64_t) 16, step_89818, lt_r_91932) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_100826, "mem_100826") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100828, "mem_100828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100830, "mem_100830") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100832, "mem_100832") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101044, &ext_mem_100825, "ext_mem_100825") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101045, &ext_mem_100781, "ext_mem_100781") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101046, &ext_mem_100803, "ext_mem_100803") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101047, &ext_mem_100759, "ext_mem_100759") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101048, &ext_mem_100770, "ext_mem_100770") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101049, &ext_mem_100748, "ext_mem_100748") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101050, &ext_mem_100814, "ext_mem_100814") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101051, &ext_mem_100792, "ext_mem_100792") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101052, &ext_mem_100836, "ext_mem_100836") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101053, &ext_mem_100824, "ext_mem_100824") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101054, &ext_mem_100780, "ext_mem_100780") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101055, &ext_mem_100802, "ext_mem_100802") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101056, &ext_mem_100758, "ext_mem_100758") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101057, &ext_mem_100769, "ext_mem_100769") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101058, &ext_mem_100747, "ext_mem_100747") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101059, &ext_mem_100813, "ext_mem_100813") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101060, &ext_mem_100791, "ext_mem_100791") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101061, &ext_mem_100835, "ext_mem_100835") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101062, &ext_mem_100823, "ext_mem_100823") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101063, &ext_mem_100779, "ext_mem_100779") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101064, &ext_mem_100801, "ext_mem_100801") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101065, &ext_mem_100757, "ext_mem_100757") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101066, &ext_mem_100768, "ext_mem_100768") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101067, &ext_mem_100746, "ext_mem_100746") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101068, &ext_mem_100812, "ext_mem_100812") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101069, &ext_mem_100790, "ext_mem_100790") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_101070, &ext_mem_100834, "ext_mem_100834") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99086, &mem_param_tmp_101044, "mem_param_tmp_101044") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99090, &mem_param_tmp_101045, "mem_param_tmp_101045") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99094, &mem_param_tmp_101046, "mem_param_tmp_101046") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99098, &mem_param_tmp_101047, "mem_param_tmp_101047") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99102, &mem_param_tmp_101048, "mem_param_tmp_101048") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99106, &mem_param_tmp_101049, "mem_param_tmp_101049") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99110, &mem_param_tmp_101050, "mem_param_tmp_101050") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99114, &mem_param_tmp_101051, "mem_param_tmp_101051") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99118, &mem_param_tmp_101052, "mem_param_tmp_101052") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99122, &mem_param_tmp_101053, "mem_param_tmp_101053") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99126, &mem_param_tmp_101054, "mem_param_tmp_101054") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99130, &mem_param_tmp_101055, "mem_param_tmp_101055") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99134, &mem_param_tmp_101056, "mem_param_tmp_101056") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99138, &mem_param_tmp_101057, "mem_param_tmp_101057") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99142, &mem_param_tmp_101058, "mem_param_tmp_101058") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99146, &mem_param_tmp_101059, "mem_param_tmp_101059") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99150, &mem_param_tmp_101060, "mem_param_tmp_101060") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99154, &mem_param_tmp_101061, "mem_param_tmp_101061") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99158, &mem_param_tmp_101062, "mem_param_tmp_101062") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99162, &mem_param_tmp_101063, "mem_param_tmp_101063") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99166, &mem_param_tmp_101064, "mem_param_tmp_101064") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99170, &mem_param_tmp_101065, "mem_param_tmp_101065") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99174, &mem_param_tmp_101066, "mem_param_tmp_101066") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99178, &mem_param_tmp_101067, "mem_param_tmp_101067") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99182, &mem_param_tmp_101068, "mem_param_tmp_101068") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99186, &mem_param_tmp_101069, "mem_param_tmp_101069") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_99190, &mem_param_tmp_101070, "mem_param_tmp_101070") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_100944, &mem_param_99086, "mem_param_99086") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100943, &mem_param_99090, "mem_param_99090") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100942, &mem_param_99094, "mem_param_99094") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100941, &mem_param_99098, "mem_param_99098") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100940, &mem_param_99102, "mem_param_99102") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100939, &mem_param_99106, "mem_param_99106") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100938, &mem_param_99110, "mem_param_99110") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100937, &mem_param_99114, "mem_param_99114") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100936, &mem_param_99118, "mem_param_99118") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100935, &mem_param_99122, "mem_param_99122") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100934, &mem_param_99126, "mem_param_99126") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100933, &mem_param_99130, "mem_param_99130") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100932, &mem_param_99134, "mem_param_99134") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100931, &mem_param_99138, "mem_param_99138") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100930, &mem_param_99142, "mem_param_99142") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100929, &mem_param_99146, "mem_param_99146") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100928, &mem_param_99150, "mem_param_99150") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100927, &mem_param_99154, "mem_param_99154") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100926, &mem_param_99158, "mem_param_99158") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100925, &mem_param_99162, "mem_param_99162") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100924, &mem_param_99166, "mem_param_99166") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100923, &mem_param_99170, "mem_param_99170") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100922, &mem_param_99174, "mem_param_99174") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100921, &mem_param_99178, "mem_param_99178") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100920, &mem_param_99182, "mem_param_99182") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100919, &mem_param_99186, "mem_param_99186") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_100918, &mem_param_99190, "mem_param_99190") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101017, &ext_mem_100939, "ext_mem_100939") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101018, &ext_mem_100941, "ext_mem_100941") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101019, &ext_mem_100940, "ext_mem_100940") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101020, &ext_mem_100943, "ext_mem_100943") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101021, &ext_mem_100937, "ext_mem_100937") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101022, &ext_mem_100942, "ext_mem_100942") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101023, &ext_mem_100938, "ext_mem_100938") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101024, &ext_mem_100944, "ext_mem_100944") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101025, &ext_mem_100936, "ext_mem_100936") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101026, &ext_mem_100930, "ext_mem_100930") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101027, &ext_mem_100932, "ext_mem_100932") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101028, &ext_mem_100931, "ext_mem_100931") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101029, &ext_mem_100934, "ext_mem_100934") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101030, &ext_mem_100928, "ext_mem_100928") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101031, &ext_mem_100933, "ext_mem_100933") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101032, &ext_mem_100929, "ext_mem_100929") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101033, &ext_mem_100935, "ext_mem_100935") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101034, &ext_mem_100927, "ext_mem_100927") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101035, &ext_mem_100921, "ext_mem_100921") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101036, &ext_mem_100923, "ext_mem_100923") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101037, &ext_mem_100922, "ext_mem_100922") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101038, &ext_mem_100925, "ext_mem_100925") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101039, &ext_mem_100919, "ext_mem_100919") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101040, &ext_mem_100924, "ext_mem_100924") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101041, &ext_mem_100920, "ext_mem_100920") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101042, &ext_mem_100926, "ext_mem_100926") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101043, &ext_mem_100918, "ext_mem_100918") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101444, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101445, &mem_out_101018, "mem_out_101018") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101446, &mem_out_101019, "mem_out_101019") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101447, &mem_out_101020, "mem_out_101020") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101448, &mem_out_101021, "mem_out_101021") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101449, &mem_out_101022, "mem_out_101022") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101450, &mem_out_101023, "mem_out_101023") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101451, &mem_out_101024, "mem_out_101024") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101452, &mem_out_101025, "mem_out_101025") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101453, &mem_out_101026, "mem_out_101026") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101454, &mem_out_101027, "mem_out_101027") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101455, &mem_out_101028, "mem_out_101028") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101456, &mem_out_101029, "mem_out_101029") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101457, &mem_out_101030, "mem_out_101030") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101458, &mem_out_101031, "mem_out_101031") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101459, &mem_out_101032, "mem_out_101032") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101460, &mem_out_101033, "mem_out_101033") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101461, &mem_out_101034, "mem_out_101034") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101462, &mem_out_101035, "mem_out_101035") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101463, &mem_out_101036, "mem_out_101036") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101464, &mem_out_101037, "mem_out_101037") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101465, &mem_out_101038, "mem_out_101038") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101466, &mem_out_101039, "mem_out_101039") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101467, &mem_out_101040, "mem_out_101040") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101468, &mem_out_101041, "mem_out_101041") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101469, &mem_out_101042, "mem_out_101042") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101470, &mem_out_101043, "mem_out_101043") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_99191);
        free(mem_99192);
        free(mem_99201);
        free(mem_99208);
        free(mem_99223);
        free(mem_99224);
        free(mem_99233);
        free(mem_99240);
        free(mem_99255);
        free(mem_99256);
        free(mem_99265);
        free(mem_99266);
        free(mem_99287);
        free(mem_99288);
        free(mem_99289);
        free(mem_99301);
        free(mem_99302);
        free(mem_99326);
        free(mem_99327);
        free(mem_99328);
        free(mem_99329);
        free(mem_99330);
        free(mem_99349);
        free(mem_99350);
        free(mem_99351);
        free(mem_99388);
        free(mem_99389);
        free(mem_99390);
        free(mem_99406);
        free(mem_99407);
        free(mem_99408);
        free(mem_99421);
        free(mem_99422);
        free(mem_99423);
        free(mem_99469);
        free(mem_99470);
        free(mem_99481);
        free(mem_99482);
        free(mem_99491);
        free(mem_99492);
        free(mem_99513);
        free(mem_99518);
        free(mem_99525);
        free(mem_99546);
        free(mem_99547);
        free(mem_99555);
        free(mem_99569);
        free(mem_99574);
        free(mem_99585);
        free(mem_99590);
        free(mem_99601);
        free(mem_99602);
        free(mem_99611);
        free(mem_99612);
        free(mem_99633);
        free(mem_99634);
        free(mem_99642);
        free(mem_99656);
        free(mem_99657);
        free(mem_99665);
        free(mem_99679);
        free(mem_99684);
        free(mem_99695);
        free(mem_99700);
        free(mem_99711);
        free(mem_99716);
        free(mem_99727);
        free(mem_99734);
        free(mem_99739);
        free(mem_99750);
        free(mem_99755);
        free(mem_99766);
        free(mem_99773);
        free(mem_99777);
        free(mem_99787);
        free(mem_99792);
        free(mem_99799);
        free(mem_99810);
        free(mem_99815);
        free(mem_99826);
        free(mem_99833);
        free(mem_99838);
        free(mem_99849);
        free(mem_99854);
        free(mem_99865);
        free(mem_99870);
        free(mem_99881);
        free(mem_99882);
        free(mem_99891);
        free(mem_99892);
        free(mem_99913);
        free(mem_99918);
        free(mem_99929);
        free(mem_99934);
        free(mem_99945);
        free(mem_99952);
        free(mem_99959);
        free(mem_99969);
        free(mem_99974);
        free(mem_99985);
        free(mem_99986);
        free(mem_99995);
        free(mem_99996);
        free(mem_100017);
        free(mem_100018);
        free(mem_100029);
        free(mem_100030);
        free(mem_100039);
        free(mem_100046);
        free(mem_100071);
        free(mem_100072);
        free(mem_100073);
        free(mem_100088);
        free(mem_100089);
        free(mem_100090);
        free(mem_100102);
        free(mem_100109);
        free(mem_100116);
        free(mem_100148);
        free(mem_100149);
        free(mem_100160);
        free(mem_100161);
        free(mem_100170);
        free(mem_100177);
        free(mem_100202);
        free(mem_100208);
        free(mem_100213);
        free(mem_100229);
        free(mem_100234);
        free(mem_100245);
        free(mem_100250);
        free(mem_100261);
        free(mem_100267);
        free(mem_100272);
        free(mem_100288);
        free(mem_100294);
        free(mem_100299);
        free(mem_100315);
        free(mem_100320);
        free(mem_100331);
        free(mem_100337);
        free(mem_100342);
        free(mem_100358);
        free(mem_100364);
        free(mem_100369);
        free(mem_100385);
        free(mem_100391);
        free(mem_100396);
        free(mem_100412);
        free(mem_100413);
        free(mem_100424);
        free(mem_100425);
        free(mem_100434);
        free(mem_100435);
        free(mem_100466);
        free(mem_100467);
        free(mem_100468);
        free(mem_100481);
        free(mem_100482);
        free(mem_100483);
        free(mem_100514);
        free(mem_100515);
        free(mem_100516);
        free(mem_100517);
        free(mem_100534);
        free(mem_100535);
        free(mem_100536);
        free(mem_100537);
        free(mem_100578);
        free(mem_100585);
        free(mem_100592);
        free(mem_100602);
        free(mem_100607);
        free(mem_100618);
        free(mem_100625);
        free(mem_100632);
        free(mem_100642);
        free(mem_100647);
        free(mem_100658);
        free(mem_100659);
        free(mem_100668);
        free(mem_100669);
        free(mem_100690);
        free(mem_100695);
        free(mem_100706);
        free(mem_100707);
        free(mem_100716);
        free(mem_100717);
        if (memblock_unref(ctx, &mem_param_tmp_101070, "mem_param_tmp_101070") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101069, "mem_param_tmp_101069") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101068, "mem_param_tmp_101068") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101067, "mem_param_tmp_101067") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101066, "mem_param_tmp_101066") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101065, "mem_param_tmp_101065") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101064, "mem_param_tmp_101064") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101063, "mem_param_tmp_101063") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101062, "mem_param_tmp_101062") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101061, "mem_param_tmp_101061") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101060, "mem_param_tmp_101060") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101059, "mem_param_tmp_101059") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101058, "mem_param_tmp_101058") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101057, "mem_param_tmp_101057") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101056, "mem_param_tmp_101056") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101055, "mem_param_tmp_101055") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101054, "mem_param_tmp_101054") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101053, "mem_param_tmp_101053") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101052, "mem_param_tmp_101052") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101051, "mem_param_tmp_101051") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101050, "mem_param_tmp_101050") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101049, "mem_param_tmp_101049") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101048, "mem_param_tmp_101048") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101047, "mem_param_tmp_101047") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101046, "mem_param_tmp_101046") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101045, "mem_param_tmp_101045") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_101044, "mem_param_tmp_101044") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100834, "ext_mem_100834") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100835, "ext_mem_100835") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100836, "ext_mem_100836") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100832, "mem_100832") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100830, "mem_100830") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100828, "mem_100828") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100826, "mem_100826") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100823, "ext_mem_100823") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100824, "ext_mem_100824") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100825, "ext_mem_100825") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100821, "mem_100821") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100819, "mem_100819") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100817, "mem_100817") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100815, "mem_100815") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100812, "ext_mem_100812") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100813, "ext_mem_100813") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100814, "ext_mem_100814") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100810, "mem_100810") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100808, "mem_100808") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100806, "mem_100806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100804, "mem_100804") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100801, "ext_mem_100801") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100802, "ext_mem_100802") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100803, "ext_mem_100803") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100799, "mem_100799") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100797, "mem_100797") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100795, "mem_100795") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100793, "mem_100793") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100790, "ext_mem_100790") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100791, "ext_mem_100791") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100792, "ext_mem_100792") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100788, "mem_100788") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100786, "mem_100786") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100784, "mem_100784") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100782, "mem_100782") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100779, "ext_mem_100779") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100780, "ext_mem_100780") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100781, "ext_mem_100781") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100777, "mem_100777") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100775, "mem_100775") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100773, "mem_100773") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100771, "mem_100771") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100768, "ext_mem_100768") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100769, "ext_mem_100769") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100770, "ext_mem_100770") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100766, "mem_100766") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100764, "mem_100764") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100762, "mem_100762") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100760, "mem_100760") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100757, "ext_mem_100757") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100758, "ext_mem_100758") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100759, "ext_mem_100759") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100755, "mem_100755") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100753, "mem_100753") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100751, "mem_100751") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100749, "mem_100749") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100746, "ext_mem_100746") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100747, "ext_mem_100747") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100748, "ext_mem_100748") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100744, "mem_100744") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100742, "mem_100742") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100740, "mem_100740") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_100738, "mem_100738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99190, "mem_param_99190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99186, "mem_param_99186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99182, "mem_param_99182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99178, "mem_param_99178") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99174, "mem_param_99174") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99170, "mem_param_99170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99166, "mem_param_99166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99162, "mem_param_99162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99158, "mem_param_99158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99154, "mem_param_99154") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99150, "mem_param_99150") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99146, "mem_param_99146") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99142, "mem_param_99142") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99138, "mem_param_99138") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99134, "mem_param_99134") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99130, "mem_param_99130") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99126, "mem_param_99126") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99122, "mem_param_99122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99118, "mem_param_99118") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99114, "mem_param_99114") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99110, "mem_param_99110") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99106, "mem_param_99106") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99102, "mem_param_99102") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99098, "mem_param_99098") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99094, "mem_param_99094") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99090, "mem_param_99090") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_99086, "mem_param_99086") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100918, "ext_mem_100918") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100919, "ext_mem_100919") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100920, "ext_mem_100920") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100921, "ext_mem_100921") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100922, "ext_mem_100922") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100923, "ext_mem_100923") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100924, "ext_mem_100924") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100925, "ext_mem_100925") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100926, "ext_mem_100926") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100927, "ext_mem_100927") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100928, "ext_mem_100928") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100929, "ext_mem_100929") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100930, "ext_mem_100930") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100931, "ext_mem_100931") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100932, "ext_mem_100932") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100933, "ext_mem_100933") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100934, "ext_mem_100934") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100935, "ext_mem_100935") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100936, "ext_mem_100936") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100937, "ext_mem_100937") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100938, "ext_mem_100938") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100939, "ext_mem_100939") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100940, "ext_mem_100940") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100941, "ext_mem_100941") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100942, "ext_mem_100942") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100943, "ext_mem_100943") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_100944, "ext_mem_100944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101043, "mem_out_101043") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101042, "mem_out_101042") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101041, "mem_out_101041") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101040, "mem_out_101040") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101039, "mem_out_101039") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101038, "mem_out_101038") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101037, "mem_out_101037") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101036, "mem_out_101036") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101035, "mem_out_101035") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101034, "mem_out_101034") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101033, "mem_out_101033") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101032, "mem_out_101032") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101031, "mem_out_101031") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101030, "mem_out_101030") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101029, "mem_out_101029") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101028, "mem_out_101028") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101027, "mem_out_101027") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101026, "mem_out_101026") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101025, "mem_out_101025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101024, "mem_out_101024") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101023, "mem_out_101023") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101022, "mem_out_101022") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101021, "mem_out_101021") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101020, "mem_out_101020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101019, "mem_out_101019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101018, "mem_out_101018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_101659, struct memblock *mem_out_p_101660, struct memblock *mem_out_p_101661, struct memblock *mem_out_p_101662, struct memblock *mem_out_p_101663, struct memblock *mem_out_p_101664, struct memblock *mem_out_p_101665, struct memblock *mem_out_p_101666, struct memblock *mem_out_p_101667)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mem_99044 = ctx->constants->mem_99044;
    struct memblock mem_99045 = ctx->constants->mem_99045;
    struct memblock mem_99046 = ctx->constants->mem_99046;
    struct memblock mem_99047 = ctx->constants->mem_99047;
    struct memblock mem_99048 = ctx->constants->mem_99048;
    struct memblock mem_99049 = ctx->constants->mem_99049;
    struct memblock mem_99050 = ctx->constants->mem_99050;
    struct memblock mem_99051 = ctx->constants->mem_99051;
    struct memblock mem_99052 = ctx->constants->mem_99052;
    
    if (memblock_set(ctx, &mem_out_101017, &mem_99051, "mem_99051") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101018, &mem_99047, "mem_99047") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101019, &mem_99049, "mem_99049") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101020, &mem_99045, "mem_99045") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101021, &mem_99046, "mem_99046") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101022, &mem_99044, "mem_99044") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101023, &mem_99050, "mem_99050") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101024, &mem_99048, "mem_99048") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_101025, &mem_99052, "mem_99052") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101659, &mem_out_101017, "mem_out_101017") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101660, &mem_out_101018, "mem_out_101018") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101661, &mem_out_101019, "mem_out_101019") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101662, &mem_out_101020, "mem_out_101020") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101663, &mem_out_101021, "mem_out_101021") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101664, &mem_out_101022, "mem_out_101022") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101665, &mem_out_101023, "mem_out_101023") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101666, &mem_out_101024, "mem_out_101024") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_101667, &mem_out_101025, "mem_out_101025") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_101025, "mem_out_101025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101024, "mem_out_101024") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101023, "mem_out_101023") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101022, "mem_out_101022") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101021, "mem_out_101021") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101020, "mem_out_101020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101019, "mem_out_101019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101018, "mem_out_101018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_101017, "mem_out_101017") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock mask_mem_99063;
    
    mask_mem_99063.references = NULL;
    
    struct memblock tokens_mem_99062;
    
    tokens_mem_99062.references = NULL;
    
    struct memblock wvoc_mem_99061;
    
    wvoc_mem_99061.references = NULL;
    
    struct memblock wval_mem_99060;
    
    wval_mem_99060.references = NULL;
    
    struct memblock wup_mem_99059;
    
    wup_mem_99059.references = NULL;
    
    struct memblock wte_mem_99058;
    
    wte_mem_99058.references = NULL;
    
    struct memblock wqry_mem_99057;
    
    wqry_mem_99057.references = NULL;
    
    struct memblock wpe_mem_99056;
    
    wpe_mem_99056.references = NULL;
    
    struct memblock wout_mem_99055;
    
    wout_mem_99055.references = NULL;
    
    struct memblock wkey_mem_99054;
    
    wkey_mem_99054.references = NULL;
    
    struct memblock wdown_mem_99053;
    
    wdown_mem_99053.references = NULL;
    wdown_mem_99053 = in0->v0->mem;
    wkey_mem_99054 = in0->v1->mem;
    wout_mem_99055 = in0->v2->mem;
    wpe_mem_99056 = in0->v3->mem;
    wqry_mem_99057 = in0->v4->mem;
    wte_mem_99058 = in0->v5->mem;
    wup_mem_99059 = in0->v6->mem;
    wval_mem_99060 = in0->v7->mem;
    wvoc_mem_99061 = in0->v8->mem;
    tokens_mem_99062 = in1->mem;
    mask_mem_99063 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_101017, wdown_mem_99053, wkey_mem_99054, wout_mem_99055, wpe_mem_99056, wqry_mem_99057, wte_mem_99058, wup_mem_99059, wval_mem_99060, wvoc_mem_99061, tokens_mem_99062, mask_mem_99063);
        if (ret == 0) {
            struct memblock mem_99044 = ctx->constants->mem_99044;
            struct memblock mem_99045 = ctx->constants->mem_99045;
            struct memblock mem_99046 = ctx->constants->mem_99046;
            struct memblock mem_99047 = ctx->constants->mem_99047;
            struct memblock mem_99048 = ctx->constants->mem_99048;
            struct memblock mem_99049 = ctx->constants->mem_99049;
            struct memblock mem_99050 = ctx->constants->mem_99050;
            struct memblock mem_99051 = ctx->constants->mem_99051;
            struct memblock mem_99052 = ctx->constants->mem_99052;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_101017;
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
    
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock wvoc_mem_99061;
    
    wvoc_mem_99061.references = NULL;
    
    struct memblock wdown_mem_99060;
    
    wdown_mem_99060.references = NULL;
    
    struct memblock wup_mem_99059;
    
    wup_mem_99059.references = NULL;
    
    struct memblock wout_mem_99058;
    
    wout_mem_99058.references = NULL;
    
    struct memblock wval_mem_99057;
    
    wval_mem_99057.references = NULL;
    
    struct memblock wkey_mem_99056;
    
    wkey_mem_99056.references = NULL;
    
    struct memblock wqry_mem_99055;
    
    wqry_mem_99055.references = NULL;
    
    struct memblock wpe_mem_99054;
    
    wpe_mem_99054.references = NULL;
    
    struct memblock wte_mem_99053;
    
    wte_mem_99053.references = NULL;
    wte_mem_99053 = in0->mem;
    wpe_mem_99054 = in1->mem;
    wqry_mem_99055 = in2->mem;
    wkey_mem_99056 = in3->mem;
    wval_mem_99057 = in4->mem;
    wout_mem_99058 = in5->mem;
    wup_mem_99059 = in6->mem;
    wdown_mem_99060 = in7->mem;
    wvoc_mem_99061 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_101017, &mem_out_101018, &mem_out_101019, &mem_out_101020, &mem_out_101021, &mem_out_101022, &mem_out_101023, &mem_out_101024, &mem_out_101025, wte_mem_99053, wpe_mem_99054, wqry_mem_99055, wkey_mem_99056, wval_mem_99057, wout_mem_99058, wup_mem_99059, wdown_mem_99060, wvoc_mem_99061);
        if (ret == 0) {
            struct memblock mem_99044 = ctx->constants->mem_99044;
            struct memblock mem_99045 = ctx->constants->mem_99045;
            struct memblock mem_99046 = ctx->constants->mem_99046;
            struct memblock mem_99047 = ctx->constants->mem_99047;
            struct memblock mem_99048 = ctx->constants->mem_99048;
            struct memblock mem_99049 = ctx->constants->mem_99049;
            struct memblock mem_99050 = ctx->constants->mem_99050;
            struct memblock mem_99051 = ctx->constants->mem_99051;
            struct memblock mem_99052 = ctx->constants->mem_99052;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_101017;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_101018;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_101019;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_101020;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_101021;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_101022;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_101023;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_101024;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_101025;
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
    
    struct memblock mem_out_101043;
    
    mem_out_101043.references = NULL;
    
    struct memblock mem_out_101042;
    
    mem_out_101042.references = NULL;
    
    struct memblock mem_out_101041;
    
    mem_out_101041.references = NULL;
    
    struct memblock mem_out_101040;
    
    mem_out_101040.references = NULL;
    
    struct memblock mem_out_101039;
    
    mem_out_101039.references = NULL;
    
    struct memblock mem_out_101038;
    
    mem_out_101038.references = NULL;
    
    struct memblock mem_out_101037;
    
    mem_out_101037.references = NULL;
    
    struct memblock mem_out_101036;
    
    mem_out_101036.references = NULL;
    
    struct memblock mem_out_101035;
    
    mem_out_101035.references = NULL;
    
    struct memblock mem_out_101034;
    
    mem_out_101034.references = NULL;
    
    struct memblock mem_out_101033;
    
    mem_out_101033.references = NULL;
    
    struct memblock mem_out_101032;
    
    mem_out_101032.references = NULL;
    
    struct memblock mem_out_101031;
    
    mem_out_101031.references = NULL;
    
    struct memblock mem_out_101030;
    
    mem_out_101030.references = NULL;
    
    struct memblock mem_out_101029;
    
    mem_out_101029.references = NULL;
    
    struct memblock mem_out_101028;
    
    mem_out_101028.references = NULL;
    
    struct memblock mem_out_101027;
    
    mem_out_101027.references = NULL;
    
    struct memblock mem_out_101026;
    
    mem_out_101026.references = NULL;
    
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    
    struct memblock seqs_mem_99082;
    
    seqs_mem_99082.references = NULL;
    
    struct memblock dls_mem_99081;
    
    dls_mem_99081.references = NULL;
    
    struct memblock masks_mem_99080;
    
    masks_mem_99080.references = NULL;
    
    struct memblock wvoc_mem_99079;
    
    wvoc_mem_99079.references = NULL;
    
    struct memblock wval_mem_99078;
    
    wval_mem_99078.references = NULL;
    
    struct memblock wup_mem_99077;
    
    wup_mem_99077.references = NULL;
    
    struct memblock wte_mem_99076;
    
    wte_mem_99076.references = NULL;
    
    struct memblock wqry_mem_99075;
    
    wqry_mem_99075.references = NULL;
    
    struct memblock wpe_mem_99074;
    
    wpe_mem_99074.references = NULL;
    
    struct memblock wout_mem_99073;
    
    wout_mem_99073.references = NULL;
    
    struct memblock wkey_mem_99072;
    
    wkey_mem_99072.references = NULL;
    
    struct memblock wdown_mem_99071;
    
    wdown_mem_99071.references = NULL;
    
    struct memblock wvoc_mem_99070;
    
    wvoc_mem_99070.references = NULL;
    
    struct memblock wval_mem_99069;
    
    wval_mem_99069.references = NULL;
    
    struct memblock wup_mem_99068;
    
    wup_mem_99068.references = NULL;
    
    struct memblock wte_mem_99067;
    
    wte_mem_99067.references = NULL;
    
    struct memblock wqry_mem_99066;
    
    wqry_mem_99066.references = NULL;
    
    struct memblock wpe_mem_99065;
    
    wpe_mem_99065.references = NULL;
    
    struct memblock wout_mem_99064;
    
    wout_mem_99064.references = NULL;
    
    struct memblock wkey_mem_99063;
    
    wkey_mem_99063.references = NULL;
    
    struct memblock wdown_mem_99062;
    
    wdown_mem_99062.references = NULL;
    
    struct memblock wvoc_mem_99061;
    
    wvoc_mem_99061.references = NULL;
    
    struct memblock wval_mem_99060;
    
    wval_mem_99060.references = NULL;
    
    struct memblock wup_mem_99059;
    
    wup_mem_99059.references = NULL;
    
    struct memblock wte_mem_99058;
    
    wte_mem_99058.references = NULL;
    
    struct memblock wqry_mem_99057;
    
    wqry_mem_99057.references = NULL;
    
    struct memblock wpe_mem_99056;
    
    wpe_mem_99056.references = NULL;
    
    struct memblock wout_mem_99055;
    
    wout_mem_99055.references = NULL;
    
    struct memblock wkey_mem_99054;
    
    wkey_mem_99054.references = NULL;
    
    struct memblock wdown_mem_99053;
    
    wdown_mem_99053.references = NULL;
    wdown_mem_99053 = in0->v0->mem;
    wkey_mem_99054 = in0->v1->mem;
    wout_mem_99055 = in0->v2->mem;
    wpe_mem_99056 = in0->v3->mem;
    wqry_mem_99057 = in0->v4->mem;
    wte_mem_99058 = in0->v5->mem;
    wup_mem_99059 = in0->v6->mem;
    wval_mem_99060 = in0->v7->mem;
    wvoc_mem_99061 = in0->v8->mem;
    wdown_mem_99062 = in1->v0->mem;
    wkey_mem_99063 = in1->v1->mem;
    wout_mem_99064 = in1->v2->mem;
    wpe_mem_99065 = in1->v3->mem;
    wqry_mem_99066 = in1->v4->mem;
    wte_mem_99067 = in1->v5->mem;
    wup_mem_99068 = in1->v6->mem;
    wval_mem_99069 = in1->v7->mem;
    wvoc_mem_99070 = in1->v8->mem;
    wdown_mem_99071 = in2->v0->mem;
    wkey_mem_99072 = in2->v1->mem;
    wout_mem_99073 = in2->v2->mem;
    wpe_mem_99074 = in2->v3->mem;
    wqry_mem_99075 = in2->v4->mem;
    wte_mem_99076 = in2->v5->mem;
    wup_mem_99077 = in2->v6->mem;
    wval_mem_99078 = in2->v7->mem;
    wvoc_mem_99079 = in2->v8->mem;
    masks_mem_99080 = in3->mem;
    dls_mem_99081 = in4->mem;
    seqs_mem_99082 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_101017, &mem_out_101018, &mem_out_101019, &mem_out_101020, &mem_out_101021, &mem_out_101022, &mem_out_101023, &mem_out_101024, &mem_out_101025, &mem_out_101026, &mem_out_101027, &mem_out_101028, &mem_out_101029, &mem_out_101030, &mem_out_101031, &mem_out_101032, &mem_out_101033, &mem_out_101034, &mem_out_101035, &mem_out_101036, &mem_out_101037, &mem_out_101038, &mem_out_101039, &mem_out_101040, &mem_out_101041, &mem_out_101042, &mem_out_101043, wdown_mem_99053, wkey_mem_99054, wout_mem_99055, wpe_mem_99056, wqry_mem_99057, wte_mem_99058, wup_mem_99059, wval_mem_99060, wvoc_mem_99061, wdown_mem_99062, wkey_mem_99063, wout_mem_99064, wpe_mem_99065, wqry_mem_99066, wte_mem_99067, wup_mem_99068, wval_mem_99069, wvoc_mem_99070, wdown_mem_99071, wkey_mem_99072, wout_mem_99073, wpe_mem_99074, wqry_mem_99075, wte_mem_99076, wup_mem_99077, wval_mem_99078, wvoc_mem_99079, masks_mem_99080, dls_mem_99081, seqs_mem_99082);
        if (ret == 0) {
            struct memblock mem_99044 = ctx->constants->mem_99044;
            struct memblock mem_99045 = ctx->constants->mem_99045;
            struct memblock mem_99046 = ctx->constants->mem_99046;
            struct memblock mem_99047 = ctx->constants->mem_99047;
            struct memblock mem_99048 = ctx->constants->mem_99048;
            struct memblock mem_99049 = ctx->constants->mem_99049;
            struct memblock mem_99050 = ctx->constants->mem_99050;
            struct memblock mem_99051 = ctx->constants->mem_99051;
            struct memblock mem_99052 = ctx->constants->mem_99052;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_101017;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_101018;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_101019;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_101020;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_101021;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_101022;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_101023;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_101024;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_101025;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_101026;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_101027;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_101028;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_101029;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_101030;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_101031;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_101032;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_101033;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_101034;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_101035;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_101036;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_101037;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_101038;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_101039;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_101040;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_101041;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_101042;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_101043;
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
    
    struct memblock mem_out_101025;
    
    mem_out_101025.references = NULL;
    
    struct memblock mem_out_101024;
    
    mem_out_101024.references = NULL;
    
    struct memblock mem_out_101023;
    
    mem_out_101023.references = NULL;
    
    struct memblock mem_out_101022;
    
    mem_out_101022.references = NULL;
    
    struct memblock mem_out_101021;
    
    mem_out_101021.references = NULL;
    
    struct memblock mem_out_101020;
    
    mem_out_101020.references = NULL;
    
    struct memblock mem_out_101019;
    
    mem_out_101019.references = NULL;
    
    struct memblock mem_out_101018;
    
    mem_out_101018.references = NULL;
    
    struct memblock mem_out_101017;
    
    mem_out_101017.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_101017, &mem_out_101018, &mem_out_101019, &mem_out_101020, &mem_out_101021, &mem_out_101022, &mem_out_101023, &mem_out_101024, &mem_out_101025);
        if (ret == 0) {
            struct memblock mem_99044 = ctx->constants->mem_99044;
            struct memblock mem_99045 = ctx->constants->mem_99045;
            struct memblock mem_99046 = ctx->constants->mem_99046;
            struct memblock mem_99047 = ctx->constants->mem_99047;
            struct memblock mem_99048 = ctx->constants->mem_99048;
            struct memblock mem_99049 = ctx->constants->mem_99049;
            struct memblock mem_99050 = ctx->constants->mem_99050;
            struct memblock mem_99051 = ctx->constants->mem_99051;
            struct memblock mem_99052 = ctx->constants->mem_99052;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_101017;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_101018;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_101019;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_101020;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_101021;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_101022;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_101023;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_101024;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_101025;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
