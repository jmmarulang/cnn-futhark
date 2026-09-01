
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
    struct memblock mem_83312;
    struct memblock mem_83313;
    struct memblock mem_83314;
    struct memblock mem_83315;
    struct memblock mem_83316;
    struct memblock mem_83317;
    struct memblock mem_83318;
    struct memblock mem_83319;
    struct memblock mem_83320;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10347(struct futhark_context *ctx, struct memblock *mem_out_p_85439, struct memblock *mem_out_p_85440, struct memblock *mem_out_p_85441, struct memblock w_mem_83321, struct memblock mw_mem_83322, struct memblock vw_mem_83323, struct memblock dw_mem_83324, int64_t n_60374, int64_t m_60375, int64_t step_60380, double lt_r_60381);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10348(struct futhark_context *ctx, struct memblock *mem_out_p_85444, struct memblock *mem_out_p_85445, struct memblock *mem_out_p_85446, struct memblock w_mem_83321, struct memblock mw_mem_83322, struct memblock vw_mem_83323, struct memblock dw_mem_83324, int64_t n_61407, int64_t m_61408, int64_t step_61413, double lt_r_61414);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85449, struct memblock wdown_mem_83321, struct memblock wkey_mem_83322, struct memblock wout_mem_83323, struct memblock wpe_mem_83324, struct memblock wqry_mem_83325, struct memblock wte_mem_83326, struct memblock wup_mem_83327, struct memblock wval_mem_83328, struct memblock wvoc_mem_83329, struct memblock tokens_mem_83330, struct memblock mask_mem_83331);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85504, struct memblock *mem_out_p_85505, struct memblock *mem_out_p_85506, struct memblock *mem_out_p_85507, struct memblock *mem_out_p_85508, struct memblock *mem_out_p_85509, struct memblock *mem_out_p_85510, struct memblock *mem_out_p_85511, struct memblock *mem_out_p_85512, struct memblock wte_mem_83321, struct memblock wpe_mem_83322, struct memblock wqry_mem_83323, struct memblock wkey_mem_83324, struct memblock wval_mem_83325, struct memblock wout_mem_83326, struct memblock wup_mem_83327, struct memblock wdown_mem_83328, struct memblock wvoc_mem_83329);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85513, struct memblock *mem_out_p_85514, struct memblock *mem_out_p_85515, struct memblock *mem_out_p_85516, struct memblock *mem_out_p_85517, struct memblock *mem_out_p_85518, struct memblock *mem_out_p_85519, struct memblock *mem_out_p_85520, struct memblock *mem_out_p_85521, struct memblock *mem_out_p_85522, struct memblock *mem_out_p_85523, struct memblock *mem_out_p_85524, struct memblock *mem_out_p_85525, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock wdown_mem_83321, struct memblock wkey_mem_83322, struct memblock wout_mem_83323, struct memblock wpe_mem_83324, struct memblock wqry_mem_83325, struct memblock wte_mem_83326, struct memblock wup_mem_83327, struct memblock wval_mem_83328, struct memblock wvoc_mem_83329, struct memblock wdown_mem_83330, struct memblock wkey_mem_83331, struct memblock wout_mem_83332, struct memblock wpe_mem_83333, struct memblock wqry_mem_83334, struct memblock wte_mem_83335, struct memblock wup_mem_83336, struct memblock wval_mem_83337, struct memblock wvoc_mem_83338, struct memblock wdown_mem_83339, struct memblock wkey_mem_83340, struct memblock wout_mem_83341, struct memblock wpe_mem_83342, struct memblock wqry_mem_83343, struct memblock wte_mem_83344, struct memblock wup_mem_83345, struct memblock wval_mem_83346, struct memblock wvoc_mem_83347, struct memblock masks_mem_83348, struct memblock dls_mem_83349, struct memblock seqs_mem_83350);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85709, struct memblock *mem_out_p_85710, struct memblock *mem_out_p_85711, struct memblock *mem_out_p_85712, struct memblock *mem_out_p_85713, struct memblock *mem_out_p_85714, struct memblock *mem_out_p_85715, struct memblock *mem_out_p_85716, struct memblock *mem_out_p_85717);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_83312 (ctx->constants->mem_83312)
    #define mem_83313 (ctx->constants->mem_83313)
    #define mem_83314 (ctx->constants->mem_83314)
    #define mem_83315 (ctx->constants->mem_83315)
    #define mem_83316 (ctx->constants->mem_83316)
    #define mem_83317 (ctx->constants->mem_83317)
    #define mem_83318 (ctx->constants->mem_83318)
    #define mem_83319 (ctx->constants->mem_83319)
    #define mem_83320 (ctx->constants->mem_83320)
    mem_83312.references = NULL;
    mem_83313.references = NULL;
    mem_83314.references = NULL;
    mem_83315.references = NULL;
    mem_83316.references = NULL;
    mem_83317.references = NULL;
    mem_83318.references = NULL;
    mem_83319.references = NULL;
    mem_83320.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83312, (int64_t) 3456, "mem_83312")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85421 = 0; nest_i_85421 < (int64_t) 27; nest_i_85421++) {
        for (int64_t nest_i_85422 = 0; nest_i_85422 < (int64_t) 16; nest_i_85422++) {
            ((double *) mem_83312.mem)[nest_i_85421 * (int64_t) 16 + nest_i_85422] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83313, (int64_t) 2048, "mem_83313")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85423 = 0; nest_i_85423 < (int64_t) 16; nest_i_85423++) {
        for (int64_t nest_i_85424 = 0; nest_i_85424 < (int64_t) 16; nest_i_85424++) {
            ((double *) mem_83313.mem)[nest_i_85423 * (int64_t) 16 + nest_i_85424] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83314, (int64_t) 2048, "mem_83314")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85425 = 0; nest_i_85425 < (int64_t) 16; nest_i_85425++) {
        for (int64_t nest_i_85426 = 0; nest_i_85426 < (int64_t) 16; nest_i_85426++) {
            ((double *) mem_83314.mem)[nest_i_85425 * (int64_t) 16 + nest_i_85426] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83315, (int64_t) 2048, "mem_83315")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85427 = 0; nest_i_85427 < (int64_t) 16; nest_i_85427++) {
        for (int64_t nest_i_85428 = 0; nest_i_85428 < (int64_t) 16; nest_i_85428++) {
            ((double *) mem_83315.mem)[nest_i_85427 * (int64_t) 16 + nest_i_85428] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83316, (int64_t) 2048, "mem_83316")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85429 = 0; nest_i_85429 < (int64_t) 16; nest_i_85429++) {
        for (int64_t nest_i_85430 = 0; nest_i_85430 < (int64_t) 16; nest_i_85430++) {
            ((double *) mem_83316.mem)[nest_i_85429 * (int64_t) 16 + nest_i_85430] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83317, (int64_t) 2048, "mem_83317")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85431 = 0; nest_i_85431 < (int64_t) 16; nest_i_85431++) {
        for (int64_t nest_i_85432 = 0; nest_i_85432 < (int64_t) 16; nest_i_85432++) {
            ((double *) mem_83317.mem)[nest_i_85431 * (int64_t) 16 + nest_i_85432] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83318, (int64_t) 8192, "mem_83318")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85433 = 0; nest_i_85433 < (int64_t) 64; nest_i_85433++) {
        for (int64_t nest_i_85434 = 0; nest_i_85434 < (int64_t) 16; nest_i_85434++) {
            ((double *) mem_83318.mem)[nest_i_85433 * (int64_t) 16 + nest_i_85434] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83319, (int64_t) 8192, "mem_83319")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85435 = 0; nest_i_85435 < (int64_t) 16; nest_i_85435++) {
        for (int64_t nest_i_85436 = 0; nest_i_85436 < (int64_t) 64; nest_i_85436++) {
            ((double *) mem_83319.mem)[nest_i_85435 * (int64_t) 64 + nest_i_85436] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83320, (int64_t) 3456, "mem_83320")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_85437 = 0; nest_i_85437 < (int64_t) 27; nest_i_85437++) {
        for (int64_t nest_i_85438 = 0; nest_i_85438 < (int64_t) 16; nest_i_85438++) {
            ((double *) mem_83320.mem)[nest_i_85437 * (int64_t) 16 + nest_i_85438] = 0.0;
        }
    }
    #undef mem_83312
    #undef mem_83313
    #undef mem_83314
    #undef mem_83315
    #undef mem_83316
    #undef mem_83317
    #undef mem_83318
    #undef mem_83319
    #undef mem_83320
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_83312, "ctx->constants->mem_83312") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83313, "ctx->constants->mem_83313") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83314, "ctx->constants->mem_83314") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83315, "ctx->constants->mem_83315") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83316, "ctx->constants->mem_83316") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83317, "ctx->constants->mem_83317") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83318, "ctx->constants->mem_83318") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83319, "ctx->constants->mem_83319") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_83320, "ctx->constants->mem_83320") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10347(struct futhark_context *ctx, struct memblock *mem_out_p_85439, struct memblock *mem_out_p_85440, struct memblock *mem_out_p_85441, struct memblock w_mem_83321, struct memblock mw_mem_83322, struct memblock vw_mem_83323, struct memblock dw_mem_83324, int64_t n_60374, int64_t m_60375, int64_t step_60380, double lt_r_60381)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83365_cached_sizze_85442 = 0;
    unsigned char *mem_83365 = NULL;
    int64_t mem_83368_cached_sizze_85443 = 0;
    unsigned char *mem_83368 = NULL;
    struct memblock mem_83403;
    
    mem_83403.references = NULL;
    
    struct memblock mem_83330;
    
    mem_83330.references = NULL;
    
    struct memblock mem_83327;
    
    mem_83327.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83325 = (int64_t) 8 * n_60374;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83326 = m_60375 * binop_x_83325;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83327, bytes_83326, "mem_83327")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83330, bytes_83326, "mem_83330")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82473 = 0; i_82473 < n_60374; i_82473++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82466 = 0; i_82466 < m_60375; i_82466++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78586 = ((double *) mw_mem_83322.mem)[i_82473 * m_60375 + i_82466];
            
            // futhark/microgpt.fut:356:10-20
            
            double zp_lhs_78587 = 0.85 * zt_rhs_78586;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78588 = ((double *) dw_mem_83324.mem)[i_82473 * m_60375 + i_82466];
            
            // futhark/microgpt.fut:356:35-45
            
            double zp_rhs_78589 = 0.15000000000000002 * zt_rhs_78588;
            
            // futhark/microgpt.fut:356:21-45
            
            double lifted_lambda_res_78590 = zp_lhs_78587 + zp_rhs_78589;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78597 = ((double *) vw_mem_83323.mem)[i_82473 * m_60375 + i_82466];
            
            // futhark/microgpt.fut:358:10-20
            
            double zp_lhs_78598 = 0.99 * zt_rhs_78597;
            
            // futhark/microgpt.fut:358:35-45
            
            double zt_lhs_78600 = 1.0000000000000009e-2 * zt_rhs_78588;
            
            // futhark/microgpt.fut:358:46-56
            
            double zp_rhs_78601 = zt_rhs_78588 * zt_lhs_78600;
            
            // futhark/microgpt.fut:358:21-56
            
            double lifted_lambda_res_78602 = zp_lhs_78598 + zp_rhs_78601;
            
            ((double *) mem_83327.mem)[i_82473 * m_60375 + i_82466] = lifted_lambda_res_78602;
            ((double *) mem_83330.mem)[i_82473 * m_60375 + i_82466] = lifted_lambda_res_78590;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65352 = sitofp_i64_f64(step_60380);
    
    // futhark/microgpt.fut:360:54-57
    
    double ztzt_rhs_65353 = 1.0 + i64_res_65352;
    
    // futhark/microgpt.fut:360:30-57
    
    double zm_rhs_65354 = fpow64(0.85, ztzt_rhs_65353);
    
    // futhark/microgpt.fut:360:23-57
    
    double zs_rhs_65355 = 1.0 - zm_rhs_65354;
    
    // futhark/microgpt.fut:362:31-58
    
    double zm_rhs_65393 = fpow64(0.99, ztzt_rhs_65353);
    
    // futhark/microgpt.fut:362:23-58
    
    double zs_rhs_65394 = 1.0 - zm_rhs_65393;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83365_cached_sizze_85442 < bytes_83326) {
        err = lexical_realloc(ctx, &mem_83365, &mem_83365_cached_sizze_85442, bytes_83326);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83368_cached_sizze_85443 < bytes_83326) {
        err = lexical_realloc(ctx, &mem_83368, &mem_83368_cached_sizze_85443, bytes_83326);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82487 = 0; i_82487 < n_60374; i_82487++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82480 = 0; i_82480 < m_60375; i_82480++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78622 = ((double *) mem_83330.mem)[i_82487 * m_60375 + i_82480];
            
            // futhark/microgpt.fut:360:18-57
            
            double lifted_lambda_res_78623 = zs_lhs_78622 / zs_rhs_65355;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78630 = ((double *) mem_83327.mem)[i_82487 * m_60375 + i_82480];
            
            // futhark/microgpt.fut:362:18-58
            
            double lifted_lambda_res_78631 = zs_lhs_78630 / zs_rhs_65394;
            
            ((double *) mem_83365)[i_82487 * m_60375 + i_82480] = lifted_lambda_res_78631;
            ((double *) mem_83368)[i_82487 * m_60375 + i_82480] = lifted_lambda_res_78623;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83403, bytes_83326, "mem_83403")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82496 = 0; i_82496 < n_60374; i_82496++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82492 = 0; i_82492 < m_60375; i_82492++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64516 = ((double *) w_mem_83321.mem)[i_82496 * m_60375 + i_82492];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64517 = ((double *) mem_83368)[i_82496 * m_60375 + i_82492];
            
            // futhark/microgpt.fut:364:21-34
            
            double zs_lhs_64518 = lt_r_60381 * zt_rhs_64517;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64519 = ((double *) mem_83365)[i_82496 * m_60375 + i_82492];
            
            // futhark/microgpt.fut:364:51-57
            
            double zp_lhs_64520 = fpow64(ztzt_lhs_64519, 0.5);
            
            // futhark/microgpt.fut:364:59-71
            
            double zs_rhs_64521 = 1.0e-8 + zp_lhs_64520;
            
            // futhark/microgpt.fut:364:35-71
            
            double zm_rhs_64522 = zs_lhs_64518 / zs_rhs_64521;
            
            // futhark/microgpt.fut:364:13-71
            
            double lifted_lambda_res_64523 = zm_lhs_64516 - zm_rhs_64522;
            
            ((double *) mem_83403.mem)[i_82496 * m_60375 + i_82492] = lifted_lambda_res_64523;
        }
    }
    if (memblock_set(ctx, &mem_out_85120, &mem_83403, "mem_83403") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85121, &mem_83330, "mem_83330") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85122, &mem_83327, "mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85439, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85440, &mem_out_85121, "mem_out_85121") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85441, &mem_out_85122, "mem_out_85122") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83365);
        free(mem_83368);
        if (memblock_unref(ctx, &mem_83403, "mem_83403") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83330, "mem_83330") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83327, "mem_83327") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85122, "mem_out_85122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85121, "mem_out_85121") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10348(struct futhark_context *ctx, struct memblock *mem_out_p_85444, struct memblock *mem_out_p_85445, struct memblock *mem_out_p_85446, struct memblock w_mem_83321, struct memblock mw_mem_83322, struct memblock vw_mem_83323, struct memblock dw_mem_83324, int64_t n_61407, int64_t m_61408, int64_t step_61413, double lt_r_61414)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83365_cached_sizze_85447 = 0;
    unsigned char *mem_83365 = NULL;
    int64_t mem_83368_cached_sizze_85448 = 0;
    unsigned char *mem_83368 = NULL;
    struct memblock mem_83403;
    
    mem_83403.references = NULL;
    
    struct memblock mem_83330;
    
    mem_83330.references = NULL;
    
    struct memblock mem_83327;
    
    mem_83327.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_83325 = (int64_t) 8 * n_61407;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_83326 = m_61408 * binop_x_83325;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83327, bytes_83326, "mem_83327")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83330, bytes_83326, "mem_83330")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82473 = 0; i_82473 < n_61407; i_82473++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82466 = 0; i_82466 < m_61408; i_82466++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78586 = ((double *) mw_mem_83322.mem)[i_82473 * m_61408 + i_82466];
            
            // futhark/microgpt.fut:356:10-20
            
            double zp_lhs_78587 = 0.85 * zt_rhs_78586;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78588 = ((double *) dw_mem_83324.mem)[i_82473 * m_61408 + i_82466];
            
            // futhark/microgpt.fut:356:35-45
            
            double zp_rhs_78589 = 0.15000000000000002 * zt_rhs_78588;
            
            // futhark/microgpt.fut:356:21-45
            
            double lifted_lambda_res_78590 = zp_lhs_78587 + zp_rhs_78589;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_78597 = ((double *) vw_mem_83323.mem)[i_82473 * m_61408 + i_82466];
            
            // futhark/microgpt.fut:358:10-20
            
            double zp_lhs_78598 = 0.99 * zt_rhs_78597;
            
            // futhark/microgpt.fut:358:35-45
            
            double zt_lhs_78600 = 1.0000000000000009e-2 * zt_rhs_78588;
            
            // futhark/microgpt.fut:358:46-56
            
            double zp_rhs_78601 = zt_rhs_78588 * zt_lhs_78600;
            
            // futhark/microgpt.fut:358:21-56
            
            double lifted_lambda_res_78602 = zp_lhs_78598 + zp_rhs_78601;
            
            ((double *) mem_83327.mem)[i_82473 * m_61408 + i_82466] = lifted_lambda_res_78602;
            ((double *) mem_83330.mem)[i_82473 * m_61408 + i_82466] = lifted_lambda_res_78590;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_65352 = sitofp_i64_f64(step_61413);
    
    // futhark/microgpt.fut:360:54-57
    
    double ztzt_rhs_65353 = 1.0 + i64_res_65352;
    
    // futhark/microgpt.fut:360:30-57
    
    double zm_rhs_65354 = fpow64(0.85, ztzt_rhs_65353);
    
    // futhark/microgpt.fut:360:23-57
    
    double zs_rhs_65355 = 1.0 - zm_rhs_65354;
    
    // futhark/microgpt.fut:362:31-58
    
    double zm_rhs_65393 = fpow64(0.99, ztzt_rhs_65353);
    
    // futhark/microgpt.fut:362:23-58
    
    double zs_rhs_65394 = 1.0 - zm_rhs_65393;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83365_cached_sizze_85447 < bytes_83326) {
        err = lexical_realloc(ctx, &mem_83365, &mem_83365_cached_sizze_85447, bytes_83326);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83368_cached_sizze_85448 < bytes_83326) {
        err = lexical_realloc(ctx, &mem_83368, &mem_83368_cached_sizze_85448, bytes_83326);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82487 = 0; i_82487 < n_61407; i_82487++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82480 = 0; i_82480 < m_61408; i_82480++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78622 = ((double *) mem_83330.mem)[i_82487 * m_61408 + i_82480];
            
            // futhark/microgpt.fut:360:18-57
            
            double lifted_lambda_res_78623 = zs_lhs_78622 / zs_rhs_65355;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_78630 = ((double *) mem_83327.mem)[i_82487 * m_61408 + i_82480];
            
            // futhark/microgpt.fut:362:18-58
            
            double lifted_lambda_res_78631 = zs_lhs_78630 / zs_rhs_65394;
            
            ((double *) mem_83365)[i_82487 * m_61408 + i_82480] = lifted_lambda_res_78631;
            ((double *) mem_83368)[i_82487 * m_61408 + i_82480] = lifted_lambda_res_78623;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83403, bytes_83326, "mem_83403")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82496 = 0; i_82496 < n_61407; i_82496++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82492 = 0; i_82492 < m_61408; i_82492++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_64516 = ((double *) w_mem_83321.mem)[i_82496 * m_61408 + i_82492];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_64517 = ((double *) mem_83368)[i_82496 * m_61408 + i_82492];
            
            // futhark/microgpt.fut:364:21-34
            
            double zs_lhs_64518 = lt_r_61414 * zt_rhs_64517;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_64519 = ((double *) mem_83365)[i_82496 * m_61408 + i_82492];
            
            // futhark/microgpt.fut:364:51-57
            
            double zp_lhs_64520 = fpow64(ztzt_lhs_64519, 0.5);
            
            // futhark/microgpt.fut:364:59-71
            
            double zs_rhs_64521 = 1.0e-8 + zp_lhs_64520;
            
            // futhark/microgpt.fut:364:35-71
            
            double zm_rhs_64522 = zs_lhs_64518 / zs_rhs_64521;
            
            // futhark/microgpt.fut:364:13-71
            
            double lifted_lambda_res_64523 = zm_lhs_64516 - zm_rhs_64522;
            
            ((double *) mem_83403.mem)[i_82496 * m_61408 + i_82492] = lifted_lambda_res_64523;
        }
    }
    if (memblock_set(ctx, &mem_out_85120, &mem_83403, "mem_83403") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85121, &mem_83330, "mem_83330") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85122, &mem_83327, "mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85444, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85445, &mem_out_85121, "mem_out_85121") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85446, &mem_out_85122, "mem_out_85122") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83365);
        free(mem_83368);
        if (memblock_unref(ctx, &mem_83403, "mem_83403") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83330, "mem_83330") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_83327, "mem_83327") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85122, "mem_out_85122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85121, "mem_out_85121") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_85449, struct memblock wdown_mem_83321, struct memblock wkey_mem_83322, struct memblock wout_mem_83323, struct memblock wpe_mem_83324, struct memblock wqry_mem_83325, struct memblock wte_mem_83326, struct memblock wup_mem_83327, struct memblock wval_mem_83328, struct memblock wvoc_mem_83329, struct memblock tokens_mem_83330, struct memblock mask_mem_83331)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83332_cached_sizze_85450 = 0;
    unsigned char *mem_83332 = NULL;
    int64_t mem_83337_cached_sizze_85451 = 0;
    unsigned char *mem_83337 = NULL;
    int64_t mem_83348_cached_sizze_85452 = 0;
    unsigned char *mem_83348 = NULL;
    int64_t mem_83353_cached_sizze_85453 = 0;
    unsigned char *mem_83353 = NULL;
    int64_t mem_83364_cached_sizze_85454 = 0;
    unsigned char *mem_83364 = NULL;
    int64_t mem_83369_cached_sizze_85455 = 0;
    unsigned char *mem_83369 = NULL;
    int64_t mem_83376_cached_sizze_85456 = 0;
    unsigned char *mem_83376 = NULL;
    int64_t mem_83387_cached_sizze_85457 = 0;
    unsigned char *mem_83387 = NULL;
    int64_t mem_83392_cached_sizze_85458 = 0;
    unsigned char *mem_83392 = NULL;
    int64_t mem_83399_cached_sizze_85459 = 0;
    unsigned char *mem_83399 = NULL;
    int64_t mem_83410_cached_sizze_85460 = 0;
    unsigned char *mem_83410 = NULL;
    int64_t mem_83411_cached_sizze_85461 = 0;
    unsigned char *mem_83411 = NULL;
    int64_t mem_83412_cached_sizze_85462 = 0;
    unsigned char *mem_83412 = NULL;
    int64_t mem_83425_cached_sizze_85463 = 0;
    unsigned char *mem_83425 = NULL;
    int64_t mem_83426_cached_sizze_85464 = 0;
    unsigned char *mem_83426 = NULL;
    int64_t mem_83427_cached_sizze_85465 = 0;
    unsigned char *mem_83427 = NULL;
    int64_t mem_83458_cached_sizze_85466 = 0;
    unsigned char *mem_83458 = NULL;
    int64_t mem_83459_cached_sizze_85467 = 0;
    unsigned char *mem_83459 = NULL;
    int64_t mem_83460_cached_sizze_85468 = 0;
    unsigned char *mem_83460 = NULL;
    int64_t mem_83476_cached_sizze_85469 = 0;
    unsigned char *mem_83476 = NULL;
    int64_t mem_83477_cached_sizze_85470 = 0;
    unsigned char *mem_83477 = NULL;
    int64_t mem_83478_cached_sizze_85471 = 0;
    unsigned char *mem_83478 = NULL;
    int64_t mem_83491_cached_sizze_85472 = 0;
    unsigned char *mem_83491 = NULL;
    int64_t mem_83492_cached_sizze_85473 = 0;
    unsigned char *mem_83492 = NULL;
    int64_t mem_83493_cached_sizze_85474 = 0;
    unsigned char *mem_83493 = NULL;
    int64_t mem_83539_cached_sizze_85475 = 0;
    unsigned char *mem_83539 = NULL;
    int64_t mem_83545_cached_sizze_85476 = 0;
    unsigned char *mem_83545 = NULL;
    int64_t mem_83550_cached_sizze_85477 = 0;
    unsigned char *mem_83550 = NULL;
    int64_t mem_83561_cached_sizze_85478 = 0;
    unsigned char *mem_83561 = NULL;
    int64_t mem_83566_cached_sizze_85479 = 0;
    unsigned char *mem_83566 = NULL;
    int64_t mem_83577_cached_sizze_85480 = 0;
    unsigned char *mem_83577 = NULL;
    int64_t mem_83582_cached_sizze_85481 = 0;
    unsigned char *mem_83582 = NULL;
    int64_t mem_83589_cached_sizze_85482 = 0;
    unsigned char *mem_83589 = NULL;
    int64_t mem_83596_cached_sizze_85483 = 0;
    unsigned char *mem_83596 = NULL;
    int64_t mem_83607_cached_sizze_85484 = 0;
    unsigned char *mem_83607 = NULL;
    int64_t mem_83612_cached_sizze_85485 = 0;
    unsigned char *mem_83612 = NULL;
    int64_t mem_83628_cached_sizze_85486 = 0;
    unsigned char *mem_83628 = NULL;
    int64_t mem_83633_cached_sizze_85487 = 0;
    unsigned char *mem_83633 = NULL;
    int64_t mem_83644_cached_sizze_85488 = 0;
    unsigned char *mem_83644 = NULL;
    int64_t mem_83649_cached_sizze_85489 = 0;
    unsigned char *mem_83649 = NULL;
    int64_t mem_83660_cached_sizze_85490 = 0;
    unsigned char *mem_83660 = NULL;
    int64_t mem_83665_cached_sizze_85491 = 0;
    unsigned char *mem_83665 = NULL;
    int64_t mem_83676_cached_sizze_85492 = 0;
    unsigned char *mem_83676 = NULL;
    int64_t mem_83681_cached_sizze_85493 = 0;
    unsigned char *mem_83681 = NULL;
    int64_t mem_83688_cached_sizze_85494 = 0;
    unsigned char *mem_83688 = NULL;
    int64_t mem_83699_cached_sizze_85495 = 0;
    unsigned char *mem_83699 = NULL;
    int64_t mem_83704_cached_sizze_85496 = 0;
    unsigned char *mem_83704 = NULL;
    int64_t mem_83715_cached_sizze_85497 = 0;
    unsigned char *mem_83715 = NULL;
    int64_t mem_83720_cached_sizze_85498 = 0;
    unsigned char *mem_83720 = NULL;
    int64_t mem_83731_cached_sizze_85499 = 0;
    unsigned char *mem_83731 = NULL;
    int64_t mem_83736_cached_sizze_85500 = 0;
    unsigned char *mem_83736 = NULL;
    int64_t mem_83747_cached_sizze_85501 = 0;
    unsigned char *mem_83747 = NULL;
    int64_t mem_83752_cached_sizze_85502 = 0;
    unsigned char *mem_83752 = NULL;
    int64_t mem_83768_cached_sizze_85503 = 0;
    unsigned char *mem_83768 = NULL;
    struct memblock mem_83763;
    
    mem_83763.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83332_cached_sizze_85450 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83332, &mem_83332_cached_sizze_85450, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83337_cached_sizze_85451 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83337, &mem_83337_cached_sizze_85451, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82468 = 0; i_82468 < (int64_t) 16; i_82468++) {
        // futhark/microgpt.fut:346:41-50
        
        int64_t tmp_73019 = ((int64_t *) tokens_mem_83330.mem)[i_82468];
        
        // futhark/microgpt.fut:346:37-51
        
        bool x_73020 = sle64((int64_t) 0, tmp_73019);
        
        // futhark/microgpt.fut:346:37-51
        
        bool y_73021 = slt64(tmp_73019, (int64_t) 27);
        
        // futhark/microgpt.fut:346:37-51
        
        bool bounds_check_73022 = x_73020 && y_73021;
        
        // futhark/microgpt.fut:346:37-51
        
        bool index_certs_73023;
        
        if (!bounds_check_73022) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73019, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:346:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:346:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82464 = 0; i_82464 < (int64_t) 16; i_82464++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73030 = ((double *) wte_mem_83326.mem)[tmp_73019 * (int64_t) 16 + i_82464];
            
            ((double *) mem_83337)[i_82464] = lifted_lambda_res_73030;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83332, i_82468 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83337, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83348_cached_sizze_85452 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83348, &mem_83348_cached_sizze_85452, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83353_cached_sizze_85453 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83353, &mem_83353_cached_sizze_85453, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82476 = 0; i_82476 < (int64_t) 16; i_82476++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82472 = 0; i_82472 < (int64_t) 16; i_82472++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73062 = ((double *) wpe_mem_83324.mem)[i_82476 * (int64_t) 16 + i_82472];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73063 = ((double *) mem_83332)[i_82476 * (int64_t) 16 + i_82472];
            
            // futhark/microgpt.fut:149:38-70
            
            double zp_res_73064 = zp_lhs_73062 + zp_rhs_73063;
            
            ((double *) mem_83353)[i_82472] = zp_res_73064;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83348, i_82476 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83364_cached_sizze_85454 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83364, &mem_83364_cached_sizze_85454, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83369_cached_sizze_85455 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83369, &mem_83369_cached_sizze_85455, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83376_cached_sizze_85456 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83376, &mem_83376_cached_sizze_85456, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82488 = 0; i_82488 < (int64_t) 16; i_82488++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82480 = 0; i_82480 < (int64_t) 16; i_82480++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73079 = ((double *) mem_83348)[i_82488 * (int64_t) 16 + i_82480];
            
            // futhark/microgpt.fut:150:64-93
            
            double zt_res_73080 = zt_lhs_73079 * zt_lhs_73079;
            
            ((double *) mem_83369)[i_82480] = zt_res_73080;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73082;
        double r_73084 = 0.0;
        
        for (int64_t i_73083 = 0; i_73083 < (int64_t) 16; i_73083++) {
            // futhark/microgpt.fut:151:35-43
            
            double lifted_lambda_res_73085 = ((double *) mem_83369)[i_73083];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73086 = r_73084 + lifted_lambda_res_73085;
            double r_tmp_85127 = zp_res_73086;
            
            r_73084 = r_tmp_85127;
        }
        defunc_0_lifted_lambda_res_73082 = r_73084;
        // futhark/microgpt.fut:151:17-60
        
        double zs_res_73087 = defunc_0_lifted_lambda_res_73082 / 16.0;
        
        // futhark/microgpt.fut:152:24-55
        
        double zp_res_73088 = 1.0e-5 + zs_res_73087;
        
        // futhark/microgpt.fut:152:16-55
        
        double sqrt_res_73089 = futrts_sqrt64(zp_res_73088);
        
        // futhark/microgpt.fut:153:42-53
        
        double zs_res_73090 = 1.0 / sqrt_res_73089;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82484 = 0; i_82484 < (int64_t) 16; i_82484++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73097 = ((double *) mem_83348)[i_82488 * (int64_t) 16 + i_82484];
            
            // futhark/microgpt.fut:153:24-53
            
            double zt_res_73098 = zs_res_73090 * zt_lhs_73097;
            
            ((double *) mem_83376)[i_82484] = zt_res_73098;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83364, i_82488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83376, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83387_cached_sizze_85457 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83387, &mem_83387_cached_sizze_85457, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83392_cached_sizze_85458 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83392, &mem_83392_cached_sizze_85458, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83399_cached_sizze_85459 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83399, &mem_83399_cached_sizze_85459, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82500 = 0; i_82500 < (int64_t) 16; i_82500++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82492 = 0; i_82492 < (int64_t) 16; i_82492++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73113 = ((double *) mem_83364)[i_82500 * (int64_t) 16 + i_82492];
            
            // futhark/microgpt.fut:154:64-93
            
            double zt_res_73114 = zt_lhs_73113 * zt_lhs_73113;
            
            ((double *) mem_83392)[i_82492] = zt_res_73114;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73116;
        double r_73118 = 0.0;
        
        for (int64_t i_73117 = 0; i_73117 < (int64_t) 16; i_73117++) {
            // futhark/microgpt.fut:155:35-43
            
            double lifted_lambda_res_73119 = ((double *) mem_83392)[i_73117];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73120 = r_73118 + lifted_lambda_res_73119;
            double r_tmp_85131 = zp_res_73120;
            
            r_73118 = r_tmp_85131;
        }
        defunc_0_lifted_lambda_res_73116 = r_73118;
        // futhark/microgpt.fut:155:17-60
        
        double zs_res_73121 = defunc_0_lifted_lambda_res_73116 / 16.0;
        
        // futhark/microgpt.fut:156:24-55
        
        double zp_res_73122 = 1.0e-5 + zs_res_73121;
        
        // futhark/microgpt.fut:156:16-55
        
        double sqrt_res_73123 = futrts_sqrt64(zp_res_73122);
        
        // futhark/microgpt.fut:157:42-53
        
        double zs_res_73124 = 1.0 / sqrt_res_73123;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82496 = 0; i_82496 < (int64_t) 16; i_82496++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73131 = ((double *) mem_83364)[i_82500 * (int64_t) 16 + i_82496];
            
            // futhark/microgpt.fut:157:24-53
            
            double zt_res_73132 = zs_res_73124 * zt_lhs_73131;
            
            ((double *) mem_83399)[i_82496] = zt_res_73132;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83387, i_82500 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83399, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83410_cached_sizze_85460 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83410, &mem_83410_cached_sizze_85460, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83411_cached_sizze_85461 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83411, &mem_83411_cached_sizze_85461, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83412_cached_sizze_85462 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83412, &mem_83412_cached_sizze_85462, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83425_cached_sizze_85463 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83425, &mem_83425_cached_sizze_85463, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83426_cached_sizze_85464 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83426, &mem_83426_cached_sizze_85464, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83427_cached_sizze_85465 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83427, &mem_83427_cached_sizze_85465, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82518 = 0; i_82518 < (int64_t) 16; i_82518++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82508 = 0; i_82508 < (int64_t) 16; i_82508++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78805;
            double r_78807 = 0.0;
            
            for (int64_t i_78806 = 0; i_78806 < (int64_t) 16; i_78806++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78808 = ((double *) wqry_mem_83325.mem)[i_82508 * (int64_t) 16 + i_78806];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78809 = ((double *) mem_83387)[i_82518 * (int64_t) 16 + i_78806];
                
                // futhark/microgpt.fut:158:72-103
                
                double zt_res_78810 = zt_lhs_78808 * zt_rhs_78809;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78811 = r_78807 + zt_res_78810;
                double r_tmp_85139 = zp_res_78811;
                
                r_78807 = r_tmp_85139;
            }
            defunc_0_lifted_lambda_res_78805 = r_78807;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78818;
            double r_78820 = 0.0;
            
            for (int64_t i_78819 = 0; i_78819 < (int64_t) 16; i_78819++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78821 = ((double *) wkey_mem_83322.mem)[i_82508 * (int64_t) 16 + i_78819];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78822 = ((double *) mem_83387)[i_82518 * (int64_t) 16 + i_78819];
                
                // futhark/microgpt.fut:159:72-103
                
                double zt_res_78823 = zt_lhs_78821 * zt_rhs_78822;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78824 = r_78820 + zt_res_78823;
                double r_tmp_85140 = zp_res_78824;
                
                r_78820 = r_tmp_85140;
            }
            defunc_0_lifted_lambda_res_78818 = r_78820;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78834;
            double r_78836 = 0.0;
            
            for (int64_t i_78835 = 0; i_78835 < (int64_t) 16; i_78835++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78837 = ((double *) wval_mem_83328.mem)[i_82508 * (int64_t) 16 + i_78835];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78838 = ((double *) mem_83387)[i_82518 * (int64_t) 16 + i_78835];
                
                // futhark/microgpt.fut:160:72-103
                
                double zt_res_78839 = zt_lhs_78837 * zt_rhs_78838;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78840 = r_78836 + zt_res_78839;
                double r_tmp_85141 = zp_res_78840;
                
                r_78836 = r_tmp_85141;
            }
            defunc_0_lifted_lambda_res_78834 = r_78836;
            ((double *) mem_83425)[i_82508] = defunc_0_lifted_lambda_res_78834;
            ((double *) mem_83426)[i_82508] = defunc_0_lifted_lambda_res_78818;
            ((double *) mem_83427)[i_82508] = defunc_0_lifted_lambda_res_78805;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83410, i_82518 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83425, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83411, i_82518 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83426, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83412, i_82518 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83427, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83458_cached_sizze_85466 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83458, &mem_83458_cached_sizze_85466, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83459_cached_sizze_85467 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83459, &mem_83459_cached_sizze_85467, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83460_cached_sizze_85468 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83460, &mem_83460_cached_sizze_85468, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83476_cached_sizze_85469 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83476, &mem_83476_cached_sizze_85469, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83477_cached_sizze_85470 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83477, &mem_83477_cached_sizze_85470, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83478_cached_sizze_85471 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83478, &mem_83478_cached_sizze_85471, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83491_cached_sizze_85472 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83491, &mem_83491_cached_sizze_85472, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83492_cached_sizze_85473 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83492, &mem_83492_cached_sizze_85473, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83493_cached_sizze_85474 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83493, &mem_83493_cached_sizze_85474, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82548 = 0; i_82548 < (int64_t) 4; i_82548++) {
        // futhark/microgpt.fut:161:83-86
        
        int64_t zp_lhs_78680 = mul64((int64_t) 4, i_82548);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82538 = 0; i_82538 < (int64_t) 16; i_82538++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82528 = 0; i_82528 < (int64_t) 4; i_82528++) {
                // futhark/microgpt.fut:161:88-93
                
                int64_t tmp_78998 = add64(zp_lhs_78680, i_82528);
                
                // futhark/microgpt.fut:161:69-95
                
                bool x_78999 = sle64((int64_t) 0, tmp_78998);
                
                // futhark/microgpt.fut:161:69-95
                
                bool y_79000 = slt64(tmp_78998, (int64_t) 16);
                
                // futhark/microgpt.fut:161:69-95
                
                bool bounds_check_79001 = x_78999 && y_79000;
                
                // futhark/microgpt.fut:161:69-95
                
                bool index_certs_79002;
                
                if (!bounds_check_79001) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_78998, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:161:69-95\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:161:52-96\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:161:33-98\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:161:15-100\n   #10 futhark/microgpt.fut:347:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79003 = ((double *) mem_83412)[i_82538 * (int64_t) 16 + tmp_78998];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79011 = ((double *) mem_83411)[i_82538 * (int64_t) 16 + tmp_78998];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79022 = ((double *) mem_83410)[i_82538 * (int64_t) 16 + tmp_78998];
                
                ((double *) mem_83491)[i_82528] = lifted_lambda_res_79022;
                ((double *) mem_83492)[i_82528] = lifted_lambda_res_79011;
                ((double *) mem_83493)[i_82528] = lifted_lambda_res_79003;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83476, i_82538 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83491, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83477, i_82538 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83492, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83478, i_82538 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83493, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83458, i_82548 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83476, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83459, i_82548 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83477, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83460, i_82548 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83478, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83539_cached_sizze_85475 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83539, &mem_83539_cached_sizze_85475, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83545_cached_sizze_85476 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83545, &mem_83545_cached_sizze_85476, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83550_cached_sizze_85477 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83550, &mem_83550_cached_sizze_85477, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83561_cached_sizze_85478 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83561, &mem_83561_cached_sizze_85478, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83566_cached_sizze_85479 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83566, &mem_83566_cached_sizze_85479, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83577_cached_sizze_85480 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83577, &mem_83577_cached_sizze_85480, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83582_cached_sizze_85481 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83582, &mem_83582_cached_sizze_85481, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83589_cached_sizze_85482 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83589, &mem_83589_cached_sizze_85482, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83596_cached_sizze_85483 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83596, &mem_83596_cached_sizze_85483, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83607_cached_sizze_85484 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83607, &mem_83607_cached_sizze_85484, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83612_cached_sizze_85485 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83612, &mem_83612_cached_sizze_85485, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82596 = 0; i_82596 < (int64_t) 4; i_82596++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82558 = 0; i_82558 < (int64_t) 16; i_82558++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82554 = 0; i_82554 < (int64_t) 16; i_82554++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73277;
                double r_73279 = 0.0;
                
                for (int64_t i_73278 = 0; i_73278 < (int64_t) 4; i_73278++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73280 = ((double *) mem_83460)[i_82596 * (int64_t) 64 + i_82558 * (int64_t) 4 + i_73278];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73281 = ((double *) mem_83459)[i_82596 * (int64_t) 64 + i_82554 * (int64_t) 4 + i_73278];
                    
                    // futhark/microgpt.fut:164:100-139
                    
                    double zt_res_73282 = zt_lhs_73280 * zt_rhs_73281;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73283 = r_73279 + zt_res_73282;
                    double r_tmp_85154 = zp_res_73283;
                    
                    r_73279 = r_tmp_85154;
                }
                defunc_0_lifted_lambda_res_73277 = r_73279;
                ((double *) mem_83550)[i_82554] = defunc_0_lifted_lambda_res_73277;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83545, i_82558 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83550, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82566 = 0; i_82566 < (int64_t) 16; i_82566++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82562 = 0; i_82562 < (int64_t) 16; i_82562++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_73298 = ((double *) mem_83545)[i_82566 * (int64_t) 16 + i_82562];
                
                // futhark/microgpt.fut:165:43-70
                
                double zs_res_73299 = zs_lhs_73298 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_73300 = ((double *) mask_mem_83331.mem)[i_82566 * (int64_t) 16 + i_82562];
                
                // futhark/microgpt.fut:165:57-90
                
                double zp_res_73301 = zs_res_73299 + zp_rhs_73300;
                
                ((double *) mem_83566)[i_82562] = zp_res_73301;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83561, i_82566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83566, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82584 = 0; i_82584 < (int64_t) 16; i_82584++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_79097;
            double redout_82568 = -INFINITY;
            
            for (int64_t i_82569 = 0; i_82569 < (int64_t) 16; i_82569++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79049 = ((double *) mem_83561)[i_82584 * (int64_t) 16 + i_82569];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_73322 = fmax64(lifted_lambda_res_79049, redout_82568);
                double redout_tmp_85158 = max_res_73322;
                
                redout_82568 = redout_tmp_85158;
            }
            defunc_0_reduce_res_79097 = redout_82568;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_73323 = -defunc_0_reduce_res_79097;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82572 = 0; i_82572 < (int64_t) 16; i_82572++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_73330 = ((double *) mem_83561)[i_82584 * (int64_t) 16 + i_82572];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_73331 = neg_res_73323 + lifted_lambda_res_73330;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_73332 = futrts_exp64(zp_res_73331);
                
                ((double *) mem_83582)[i_82572] = exp_res_73332;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73334;
            double r_73336 = 0.0;
            
            for (int64_t i_73335 = 0; i_73335 < (int64_t) 16; i_73335++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_73337 = ((double *) mem_83582)[i_73335];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73338 = r_73336 + lifted_lambda_res_73337;
                double r_tmp_85160 = zp_res_73338;
                
                r_73336 = r_tmp_85160;
            }
            defunc_0_lifted_lambda_res_73334 = r_73336;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82576 = 0; i_82576 < (int64_t) 16; i_82576++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_73345 = ((double *) mem_83582)[i_82576];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_73346 = zs_lhs_73345 / defunc_0_lifted_lambda_res_73334;
                
                ((double *) mem_83589)[i_82576] = zs_res_73346;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82580 = 0; i_82580 < (int64_t) 16; i_82580++) {
                // futhark/microgpt.fut:167:23-31
                
                double lifted_lambda_res_73354 = ((double *) mem_83589)[i_82580];
                
                ((double *) mem_83596)[i_82580] = lifted_lambda_res_73354;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83577, i_82584 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83596, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82592 = 0; i_82592 < (int64_t) 16; i_82592++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82588 = 0; i_82588 < (int64_t) 4; i_82588++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_73369;
                double r_73371 = 0.0;
                
                for (int64_t i_73370 = 0; i_73370 < (int64_t) 16; i_73370++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_73372 = ((double *) mem_83577)[i_82592 * (int64_t) 16 + i_73370];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_73373 = ((double *) mem_83458)[i_82596 * (int64_t) 64 + i_73370 * (int64_t) 4 + i_82588];
                    
                    // futhark/microgpt.fut:168:61-96
                    
                    double zt_res_73374 = zt_lhs_73372 * zt_rhs_73373;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_73375 = r_73371 + zt_res_73374;
                    double r_tmp_85165 = zp_res_73375;
                    
                    r_73371 = r_tmp_85165;
                }
                defunc_0_lifted_lambda_res_73369 = r_73371;
                ((double *) mem_83612)[i_82588] = defunc_0_lifted_lambda_res_73369;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83607, i_82592 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83612, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_83539, i_82596 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83607, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83628_cached_sizze_85486 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83628, &mem_83628_cached_sizze_85486, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83633_cached_sizze_85487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83633, &mem_83633_cached_sizze_85487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82604 = 0; i_82604 < (int64_t) 16; i_82604++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82600 = 0; i_82600 < (int64_t) 16; i_82600++) {
            // futhark/microgpt.fut:169:61-64
            
            int64_t tmp_73387 = sdiv64(i_82600, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool x_73388 = sle64((int64_t) 0, tmp_73387);
            
            // futhark/microgpt.fut:169:53-66
            
            bool y_73389 = slt64(tmp_73387, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-66
            
            bool bounds_check_73390 = x_73388 && y_73389;
            
            // futhark/microgpt.fut:169:53-66
            
            bool index_certs_73391;
            
            if (!bounds_check_73390) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73387, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-66\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:347:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:169:77-80
            
            int64_t tmp_73392 = smod64(i_82600, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool x_73393 = sle64((int64_t) 0, tmp_73392);
            
            // futhark/microgpt.fut:169:53-82
            
            bool y_73394 = slt64(tmp_73392, (int64_t) 4);
            
            // futhark/microgpt.fut:169:53-82
            
            bool bounds_check_73395 = x_73393 && y_73394;
            
            // futhark/microgpt.fut:169:53-82
            
            bool index_certs_73396;
            
            if (!bounds_check_73395) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_73392, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:169:53-82\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:169:35-83\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:169:16-85\n   #7  futhark/microgpt.fut:347:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_73397 = ((double *) mem_83539)[tmp_73387 * (int64_t) 64 + i_82604 * (int64_t) 4 + tmp_73392];
            
            ((double *) mem_83633)[i_82600] = lifted_lambda_res_73397;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83628, i_82604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83633, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83644_cached_sizze_85488 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83644, &mem_83644_cached_sizze_85488, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83649_cached_sizze_85489 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83649, &mem_83649_cached_sizze_85489, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82612 = 0; i_82612 < (int64_t) 16; i_82612++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82608 = 0; i_82608 < (int64_t) 16; i_82608++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73412;
            double r_73414 = 0.0;
            
            for (int64_t i_73413 = 0; i_73413 < (int64_t) 16; i_73413++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73415 = ((double *) wout_mem_83323.mem)[i_82608 * (int64_t) 16 + i_73413];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73416 = ((double *) mem_83628)[i_82612 * (int64_t) 16 + i_73413];
                
                // futhark/microgpt.fut:170:73-105
                
                double zt_res_73417 = zt_lhs_73415 * zt_rhs_73416;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73418 = r_73414 + zt_res_73417;
                double r_tmp_85170 = zp_res_73418;
                
                r_73414 = r_tmp_85170;
            }
            defunc_0_lifted_lambda_res_73412 = r_73414;
            ((double *) mem_83649)[i_82608] = defunc_0_lifted_lambda_res_73412;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83644, i_82612 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83649, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83660_cached_sizze_85490 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83660, &mem_83660_cached_sizze_85490, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83665_cached_sizze_85491 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83665, &mem_83665_cached_sizze_85491, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82620 = 0; i_82620 < (int64_t) 16; i_82620++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82616 = 0; i_82616 < (int64_t) 16; i_82616++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73433 = ((double *) mem_83644)[i_82620 * (int64_t) 16 + i_82616];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73434 = ((double *) mem_83364)[i_82620 * (int64_t) 16 + i_82616];
            
            // futhark/microgpt.fut:171:42-72
            
            double zp_res_73435 = zp_lhs_73433 + zp_rhs_73434;
            
            ((double *) mem_83665)[i_82616] = zp_res_73435;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83660, i_82620 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83676_cached_sizze_85492 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83676, &mem_83676_cached_sizze_85492, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83681_cached_sizze_85493 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83681, &mem_83681_cached_sizze_85493, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83688_cached_sizze_85494 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83688, &mem_83688_cached_sizze_85494, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82632 = 0; i_82632 < (int64_t) 16; i_82632++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82624 = 0; i_82624 < (int64_t) 16; i_82624++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73450 = ((double *) mem_83660)[i_82632 * (int64_t) 16 + i_82624];
            
            // futhark/microgpt.fut:172:65-96
            
            double zt_res_73451 = zt_lhs_73450 * zt_lhs_73450;
            
            ((double *) mem_83681)[i_82624] = zt_res_73451;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_73453;
        double r_73455 = 0.0;
        
        for (int64_t i_73454 = 0; i_73454 < (int64_t) 16; i_73454++) {
            // futhark/microgpt.fut:173:35-43
            
            double lifted_lambda_res_73456 = ((double *) mem_83681)[i_73454];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_73457 = r_73455 + lifted_lambda_res_73456;
            double r_tmp_85175 = zp_res_73457;
            
            r_73455 = r_tmp_85175;
        }
        defunc_0_lifted_lambda_res_73453 = r_73455;
        // futhark/microgpt.fut:173:17-60
        
        double zs_res_73458 = defunc_0_lifted_lambda_res_73453 / 16.0;
        
        // futhark/microgpt.fut:174:24-55
        
        double zp_res_73459 = 1.0e-5 + zs_res_73458;
        
        // futhark/microgpt.fut:174:16-55
        
        double sqrt_res_73460 = futrts_sqrt64(zp_res_73459);
        
        // futhark/microgpt.fut:175:43-54
        
        double zs_res_73461 = 1.0 / sqrt_res_73460;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82628 = 0; i_82628 < (int64_t) 16; i_82628++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_73468 = ((double *) mem_83660)[i_82632 * (int64_t) 16 + i_82628];
            
            // futhark/microgpt.fut:175:24-54
            
            double zt_res_73469 = zs_res_73461 * zt_lhs_73468;
            
            ((double *) mem_83688)[i_82628] = zt_res_73469;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83676, i_82632 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83688, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83699_cached_sizze_85495 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83699, &mem_83699_cached_sizze_85495, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83704_cached_sizze_85496 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83704, &mem_83704_cached_sizze_85496, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82640 = 0; i_82640 < (int64_t) 16; i_82640++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82636 = 0; i_82636 < (int64_t) 64; i_82636++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73485;
            double r_73487 = 0.0;
            
            for (int64_t i_73486 = 0; i_73486 < (int64_t) 16; i_73486++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73488 = ((double *) wup_mem_83327.mem)[i_82636 * (int64_t) 16 + i_73486];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73489 = ((double *) mem_83676)[i_82640 * (int64_t) 16 + i_73486];
                
                // futhark/microgpt.fut:176:73-104
                
                double zt_res_73490 = zt_lhs_73488 * zt_rhs_73489;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73491 = r_73487 + zt_res_73490;
                double r_tmp_85179 = zp_res_73491;
                
                r_73487 = r_tmp_85179;
            }
            defunc_0_lifted_lambda_res_73485 = r_73487;
            ((double *) mem_83704)[i_82636] = defunc_0_lifted_lambda_res_73485;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83699, i_82640 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83704, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83715_cached_sizze_85497 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83715, &mem_83715_cached_sizze_85497, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83720_cached_sizze_85498 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83720, &mem_83720_cached_sizze_85498, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82648 = 0; i_82648 < (int64_t) 16; i_82648++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82644 = 0; i_82644 < (int64_t) 64; i_82644++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_73506 = ((double *) mem_83699)[i_82648 * (int64_t) 64 + i_82644];
            
            // futhark/microgpt.fut:177:42-66
            
            double max_res_73507 = fmax64(0.0, max_arg0_73506);
            
            ((double *) mem_83720)[i_82644] = max_res_73507;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83715, i_82648 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83720, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83731_cached_sizze_85499 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83731, &mem_83731_cached_sizze_85499, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83736_cached_sizze_85500 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83736, &mem_83736_cached_sizze_85500, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82656 = 0; i_82656 < (int64_t) 16; i_82656++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82652 = 0; i_82652 < (int64_t) 16; i_82652++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73522;
            double r_73524 = 0.0;
            
            for (int64_t i_73523 = 0; i_73523 < (int64_t) 64; i_73523++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73525 = ((double *) wdown_mem_83321.mem)[i_82652 * (int64_t) 64 + i_73523];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73526 = ((double *) mem_83715)[i_82656 * (int64_t) 64 + i_73523];
                
                // futhark/microgpt.fut:178:73-106
                
                double zt_res_73527 = zt_lhs_73525 * zt_rhs_73526;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73528 = r_73524 + zt_res_73527;
                double r_tmp_85184 = zp_res_73528;
                
                r_73524 = r_tmp_85184;
            }
            defunc_0_lifted_lambda_res_73522 = r_73524;
            ((double *) mem_83736)[i_82652] = defunc_0_lifted_lambda_res_73522;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83731, i_82656 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83736, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83747_cached_sizze_85501 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83747, &mem_83747_cached_sizze_85501, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83752_cached_sizze_85502 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83752, &mem_83752_cached_sizze_85502, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82664 = 0; i_82664 < (int64_t) 16; i_82664++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82660 = 0; i_82660 < (int64_t) 16; i_82660++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_73543 = ((double *) mem_83731)[i_82664 * (int64_t) 16 + i_82660];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_73544 = ((double *) mem_83660)[i_82664 * (int64_t) 16 + i_82660];
            
            // futhark/microgpt.fut:179:42-73
            
            double zp_res_73545 = zp_lhs_73543 + zp_rhs_73544;
            
            ((double *) mem_83752)[i_82660] = zp_res_73545;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83747, i_82664 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83752, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_83763, (int64_t) 3456, "mem_83763")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83768_cached_sizze_85503 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83768, &mem_83768_cached_sizze_85503, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_82672 = 0; i_82672 < (int64_t) 16; i_82672++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82668 = 0; i_82668 < (int64_t) 27; i_82668++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_73561;
            double r_73563 = 0.0;
            
            for (int64_t i_73562 = 0; i_73562 < (int64_t) 16; i_73562++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_73564 = ((double *) wvoc_mem_83329.mem)[i_82668 * (int64_t) 16 + i_73562];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_73565 = ((double *) mem_83747)[i_82672 * (int64_t) 16 + i_73562];
                
                // futhark/microgpt.fut:180:62-94
                
                double zt_res_73566 = zt_lhs_73564 * zt_rhs_73565;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_73567 = r_73563 + zt_res_73566;
                double r_tmp_85189 = zp_res_73567;
                
                r_73563 = r_tmp_85189;
            }
            defunc_0_lifted_lambda_res_73561 = r_73563;
            ((double *) mem_83768)[i_82668] = defunc_0_lifted_lambda_res_73561;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_83763.mem, i_82672 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83768, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_85120, &mem_83763, "mem_83763") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85449, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_83332);
        free(mem_83337);
        free(mem_83348);
        free(mem_83353);
        free(mem_83364);
        free(mem_83369);
        free(mem_83376);
        free(mem_83387);
        free(mem_83392);
        free(mem_83399);
        free(mem_83410);
        free(mem_83411);
        free(mem_83412);
        free(mem_83425);
        free(mem_83426);
        free(mem_83427);
        free(mem_83458);
        free(mem_83459);
        free(mem_83460);
        free(mem_83476);
        free(mem_83477);
        free(mem_83478);
        free(mem_83491);
        free(mem_83492);
        free(mem_83493);
        free(mem_83539);
        free(mem_83545);
        free(mem_83550);
        free(mem_83561);
        free(mem_83566);
        free(mem_83577);
        free(mem_83582);
        free(mem_83589);
        free(mem_83596);
        free(mem_83607);
        free(mem_83612);
        free(mem_83628);
        free(mem_83633);
        free(mem_83644);
        free(mem_83649);
        free(mem_83660);
        free(mem_83665);
        free(mem_83676);
        free(mem_83681);
        free(mem_83688);
        free(mem_83699);
        free(mem_83704);
        free(mem_83715);
        free(mem_83720);
        free(mem_83731);
        free(mem_83736);
        free(mem_83747);
        free(mem_83752);
        free(mem_83768);
        if (memblock_unref(ctx, &mem_83763, "mem_83763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_85504, struct memblock *mem_out_p_85505, struct memblock *mem_out_p_85506, struct memblock *mem_out_p_85507, struct memblock *mem_out_p_85508, struct memblock *mem_out_p_85509, struct memblock *mem_out_p_85510, struct memblock *mem_out_p_85511, struct memblock *mem_out_p_85512, struct memblock wte_mem_83321, struct memblock wpe_mem_83322, struct memblock wqry_mem_83323, struct memblock wkey_mem_83324, struct memblock wval_mem_83325, struct memblock wout_mem_83326, struct memblock wup_mem_83327, struct memblock wdown_mem_83328, struct memblock wvoc_mem_83329)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    if (memblock_set(ctx, &mem_out_85120, &wdown_mem_83328, "wdown_mem_83328") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85121, &wkey_mem_83324, "wkey_mem_83324") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85122, &wout_mem_83326, "wout_mem_83326") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85123, &wpe_mem_83322, "wpe_mem_83322") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85124, &wqry_mem_83323, "wqry_mem_83323") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85125, &wte_mem_83321, "wte_mem_83321") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85126, &wup_mem_83327, "wup_mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &wval_mem_83325, "wval_mem_83325") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &wvoc_mem_83329, "wvoc_mem_83329") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85504, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85505, &mem_out_85121, "mem_out_85121") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85506, &mem_out_85122, "mem_out_85122") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85507, &mem_out_85123, "mem_out_85123") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85508, &mem_out_85124, "mem_out_85124") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85509, &mem_out_85125, "mem_out_85125") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85510, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85511, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85512, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85125, "mem_out_85125") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85124, "mem_out_85124") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85123, "mem_out_85123") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85122, "mem_out_85122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85121, "mem_out_85121") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_85513, struct memblock *mem_out_p_85514, struct memblock *mem_out_p_85515, struct memblock *mem_out_p_85516, struct memblock *mem_out_p_85517, struct memblock *mem_out_p_85518, struct memblock *mem_out_p_85519, struct memblock *mem_out_p_85520, struct memblock *mem_out_p_85521, struct memblock *mem_out_p_85522, struct memblock *mem_out_p_85523, struct memblock *mem_out_p_85524, struct memblock *mem_out_p_85525, struct memblock *mem_out_p_85526, struct memblock *mem_out_p_85527, struct memblock *mem_out_p_85528, struct memblock *mem_out_p_85529, struct memblock *mem_out_p_85530, struct memblock *mem_out_p_85531, struct memblock *mem_out_p_85532, struct memblock *mem_out_p_85533, struct memblock *mem_out_p_85534, struct memblock *mem_out_p_85535, struct memblock *mem_out_p_85536, struct memblock *mem_out_p_85537, struct memblock *mem_out_p_85538, struct memblock *mem_out_p_85539, struct memblock wdown_mem_83321, struct memblock wkey_mem_83322, struct memblock wout_mem_83323, struct memblock wpe_mem_83324, struct memblock wqry_mem_83325, struct memblock wte_mem_83326, struct memblock wup_mem_83327, struct memblock wval_mem_83328, struct memblock wvoc_mem_83329, struct memblock wdown_mem_83330, struct memblock wkey_mem_83331, struct memblock wout_mem_83332, struct memblock wpe_mem_83333, struct memblock wqry_mem_83334, struct memblock wte_mem_83335, struct memblock wup_mem_83336, struct memblock wval_mem_83337, struct memblock wvoc_mem_83338, struct memblock wdown_mem_83339, struct memblock wkey_mem_83340, struct memblock wout_mem_83341, struct memblock wpe_mem_83342, struct memblock wqry_mem_83343, struct memblock wte_mem_83344, struct memblock wup_mem_83345, struct memblock wval_mem_83346, struct memblock wvoc_mem_83347, struct memblock masks_mem_83348, struct memblock dls_mem_83349, struct memblock seqs_mem_83350)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_83459_cached_sizze_85540 = 0;
    unsigned char *mem_83459 = NULL;
    int64_t mem_83460_cached_sizze_85541 = 0;
    unsigned char *mem_83460 = NULL;
    int64_t mem_83469_cached_sizze_85542 = 0;
    unsigned char *mem_83469 = NULL;
    int64_t mem_83476_cached_sizze_85543 = 0;
    unsigned char *mem_83476 = NULL;
    int64_t mem_83491_cached_sizze_85544 = 0;
    unsigned char *mem_83491 = NULL;
    int64_t mem_83492_cached_sizze_85545 = 0;
    unsigned char *mem_83492 = NULL;
    int64_t mem_83501_cached_sizze_85546 = 0;
    unsigned char *mem_83501 = NULL;
    int64_t mem_83508_cached_sizze_85547 = 0;
    unsigned char *mem_83508 = NULL;
    int64_t mem_83523_cached_sizze_85548 = 0;
    unsigned char *mem_83523 = NULL;
    int64_t mem_83524_cached_sizze_85549 = 0;
    unsigned char *mem_83524 = NULL;
    int64_t mem_83533_cached_sizze_85550 = 0;
    unsigned char *mem_83533 = NULL;
    int64_t mem_83534_cached_sizze_85551 = 0;
    unsigned char *mem_83534 = NULL;
    int64_t mem_83555_cached_sizze_85552 = 0;
    unsigned char *mem_83555 = NULL;
    int64_t mem_83556_cached_sizze_85553 = 0;
    unsigned char *mem_83556 = NULL;
    int64_t mem_83557_cached_sizze_85554 = 0;
    unsigned char *mem_83557 = NULL;
    int64_t mem_83569_cached_sizze_85555 = 0;
    unsigned char *mem_83569 = NULL;
    int64_t mem_83570_cached_sizze_85556 = 0;
    unsigned char *mem_83570 = NULL;
    int64_t mem_83594_cached_sizze_85557 = 0;
    unsigned char *mem_83594 = NULL;
    int64_t mem_83595_cached_sizze_85558 = 0;
    unsigned char *mem_83595 = NULL;
    int64_t mem_83596_cached_sizze_85559 = 0;
    unsigned char *mem_83596 = NULL;
    int64_t mem_83597_cached_sizze_85560 = 0;
    unsigned char *mem_83597 = NULL;
    int64_t mem_83598_cached_sizze_85561 = 0;
    unsigned char *mem_83598 = NULL;
    int64_t mem_83617_cached_sizze_85562 = 0;
    unsigned char *mem_83617 = NULL;
    int64_t mem_83618_cached_sizze_85563 = 0;
    unsigned char *mem_83618 = NULL;
    int64_t mem_83619_cached_sizze_85564 = 0;
    unsigned char *mem_83619 = NULL;
    int64_t mem_83656_cached_sizze_85565 = 0;
    unsigned char *mem_83656 = NULL;
    int64_t mem_83657_cached_sizze_85566 = 0;
    unsigned char *mem_83657 = NULL;
    int64_t mem_83658_cached_sizze_85567 = 0;
    unsigned char *mem_83658 = NULL;
    int64_t mem_83674_cached_sizze_85568 = 0;
    unsigned char *mem_83674 = NULL;
    int64_t mem_83675_cached_sizze_85569 = 0;
    unsigned char *mem_83675 = NULL;
    int64_t mem_83676_cached_sizze_85570 = 0;
    unsigned char *mem_83676 = NULL;
    int64_t mem_83689_cached_sizze_85571 = 0;
    unsigned char *mem_83689 = NULL;
    int64_t mem_83690_cached_sizze_85572 = 0;
    unsigned char *mem_83690 = NULL;
    int64_t mem_83691_cached_sizze_85573 = 0;
    unsigned char *mem_83691 = NULL;
    int64_t mem_83737_cached_sizze_85574 = 0;
    unsigned char *mem_83737 = NULL;
    int64_t mem_83738_cached_sizze_85575 = 0;
    unsigned char *mem_83738 = NULL;
    int64_t mem_83749_cached_sizze_85576 = 0;
    unsigned char *mem_83749 = NULL;
    int64_t mem_83750_cached_sizze_85577 = 0;
    unsigned char *mem_83750 = NULL;
    int64_t mem_83759_cached_sizze_85578 = 0;
    unsigned char *mem_83759 = NULL;
    int64_t mem_83760_cached_sizze_85579 = 0;
    unsigned char *mem_83760 = NULL;
    int64_t mem_83781_cached_sizze_85580 = 0;
    unsigned char *mem_83781 = NULL;
    int64_t mem_83786_cached_sizze_85581 = 0;
    unsigned char *mem_83786 = NULL;
    int64_t mem_83797_cached_sizze_85582 = 0;
    unsigned char *mem_83797 = NULL;
    int64_t mem_83802_cached_sizze_85583 = 0;
    unsigned char *mem_83802 = NULL;
    int64_t mem_83809_cached_sizze_85584 = 0;
    unsigned char *mem_83809 = NULL;
    int64_t mem_83816_cached_sizze_85585 = 0;
    unsigned char *mem_83816 = NULL;
    int64_t mem_83827_cached_sizze_85586 = 0;
    unsigned char *mem_83827 = NULL;
    int64_t mem_83832_cached_sizze_85587 = 0;
    unsigned char *mem_83832 = NULL;
    int64_t mem_83853_cached_sizze_85588 = 0;
    unsigned char *mem_83853 = NULL;
    int64_t mem_83854_cached_sizze_85589 = 0;
    unsigned char *mem_83854 = NULL;
    int64_t mem_83862_cached_sizze_85590 = 0;
    unsigned char *mem_83862 = NULL;
    int64_t mem_83876_cached_sizze_85591 = 0;
    unsigned char *mem_83876 = NULL;
    int64_t mem_83881_cached_sizze_85592 = 0;
    unsigned char *mem_83881 = NULL;
    int64_t mem_83892_cached_sizze_85593 = 0;
    unsigned char *mem_83892 = NULL;
    int64_t mem_83897_cached_sizze_85594 = 0;
    unsigned char *mem_83897 = NULL;
    int64_t mem_83908_cached_sizze_85595 = 0;
    unsigned char *mem_83908 = NULL;
    int64_t mem_83909_cached_sizze_85596 = 0;
    unsigned char *mem_83909 = NULL;
    int64_t mem_83918_cached_sizze_85597 = 0;
    unsigned char *mem_83918 = NULL;
    int64_t mem_83919_cached_sizze_85598 = 0;
    unsigned char *mem_83919 = NULL;
    int64_t mem_83940_cached_sizze_85599 = 0;
    unsigned char *mem_83940 = NULL;
    int64_t mem_83941_cached_sizze_85600 = 0;
    unsigned char *mem_83941 = NULL;
    int64_t mem_83949_cached_sizze_85601 = 0;
    unsigned char *mem_83949 = NULL;
    int64_t mem_83963_cached_sizze_85602 = 0;
    unsigned char *mem_83963 = NULL;
    int64_t mem_83964_cached_sizze_85603 = 0;
    unsigned char *mem_83964 = NULL;
    int64_t mem_83972_cached_sizze_85604 = 0;
    unsigned char *mem_83972 = NULL;
    int64_t mem_83986_cached_sizze_85605 = 0;
    unsigned char *mem_83986 = NULL;
    int64_t mem_83991_cached_sizze_85606 = 0;
    unsigned char *mem_83991 = NULL;
    int64_t mem_84002_cached_sizze_85607 = 0;
    unsigned char *mem_84002 = NULL;
    int64_t mem_84007_cached_sizze_85608 = 0;
    unsigned char *mem_84007 = NULL;
    int64_t mem_84018_cached_sizze_85609 = 0;
    unsigned char *mem_84018 = NULL;
    int64_t mem_84023_cached_sizze_85610 = 0;
    unsigned char *mem_84023 = NULL;
    int64_t mem_84034_cached_sizze_85611 = 0;
    unsigned char *mem_84034 = NULL;
    int64_t mem_84035_cached_sizze_85612 = 0;
    unsigned char *mem_84035 = NULL;
    int64_t mem_84044_cached_sizze_85613 = 0;
    unsigned char *mem_84044 = NULL;
    int64_t mem_84045_cached_sizze_85614 = 0;
    unsigned char *mem_84045 = NULL;
    int64_t mem_84058_cached_sizze_85615 = 0;
    unsigned char *mem_84058 = NULL;
    int64_t mem_84059_cached_sizze_85616 = 0;
    unsigned char *mem_84059 = NULL;
    int64_t mem_84072_cached_sizze_85617 = 0;
    unsigned char *mem_84072 = NULL;
    int64_t mem_84073_cached_sizze_85618 = 0;
    unsigned char *mem_84073 = NULL;
    int64_t mem_84094_cached_sizze_85619 = 0;
    unsigned char *mem_84094 = NULL;
    int64_t mem_84101_cached_sizze_85620 = 0;
    unsigned char *mem_84101 = NULL;
    int64_t mem_84106_cached_sizze_85621 = 0;
    unsigned char *mem_84106 = NULL;
    int64_t mem_84117_cached_sizze_85622 = 0;
    unsigned char *mem_84117 = NULL;
    int64_t mem_84122_cached_sizze_85623 = 0;
    unsigned char *mem_84122 = NULL;
    int64_t mem_84133_cached_sizze_85624 = 0;
    unsigned char *mem_84133 = NULL;
    int64_t mem_84134_cached_sizze_85625 = 0;
    unsigned char *mem_84134 = NULL;
    int64_t mem_84143_cached_sizze_85626 = 0;
    unsigned char *mem_84143 = NULL;
    int64_t mem_84144_cached_sizze_85627 = 0;
    unsigned char *mem_84144 = NULL;
    int64_t mem_84165_cached_sizze_85628 = 0;
    unsigned char *mem_84165 = NULL;
    int64_t mem_84170_cached_sizze_85629 = 0;
    unsigned char *mem_84170 = NULL;
    int64_t mem_84181_cached_sizze_85630 = 0;
    unsigned char *mem_84181 = NULL;
    int64_t mem_84186_cached_sizze_85631 = 0;
    unsigned char *mem_84186 = NULL;
    int64_t mem_84197_cached_sizze_85632 = 0;
    unsigned char *mem_84197 = NULL;
    int64_t mem_84204_cached_sizze_85633 = 0;
    unsigned char *mem_84204 = NULL;
    int64_t mem_84211_cached_sizze_85634 = 0;
    unsigned char *mem_84211 = NULL;
    int64_t mem_84221_cached_sizze_85635 = 0;
    unsigned char *mem_84221 = NULL;
    int64_t mem_84226_cached_sizze_85636 = 0;
    unsigned char *mem_84226 = NULL;
    int64_t mem_84237_cached_sizze_85637 = 0;
    unsigned char *mem_84237 = NULL;
    int64_t mem_84238_cached_sizze_85638 = 0;
    unsigned char *mem_84238 = NULL;
    int64_t mem_84247_cached_sizze_85639 = 0;
    unsigned char *mem_84247 = NULL;
    int64_t mem_84248_cached_sizze_85640 = 0;
    unsigned char *mem_84248 = NULL;
    int64_t mem_84269_cached_sizze_85641 = 0;
    unsigned char *mem_84269 = NULL;
    int64_t mem_84270_cached_sizze_85642 = 0;
    unsigned char *mem_84270 = NULL;
    int64_t mem_84281_cached_sizze_85643 = 0;
    unsigned char *mem_84281 = NULL;
    int64_t mem_84282_cached_sizze_85644 = 0;
    unsigned char *mem_84282 = NULL;
    int64_t mem_84291_cached_sizze_85645 = 0;
    unsigned char *mem_84291 = NULL;
    int64_t mem_84298_cached_sizze_85646 = 0;
    unsigned char *mem_84298 = NULL;
    int64_t mem_84323_cached_sizze_85647 = 0;
    unsigned char *mem_84323 = NULL;
    int64_t mem_84324_cached_sizze_85648 = 0;
    unsigned char *mem_84324 = NULL;
    int64_t mem_84335_cached_sizze_85649 = 0;
    unsigned char *mem_84335 = NULL;
    int64_t mem_84336_cached_sizze_85650 = 0;
    unsigned char *mem_84336 = NULL;
    int64_t mem_84345_cached_sizze_85651 = 0;
    unsigned char *mem_84345 = NULL;
    int64_t mem_84352_cached_sizze_85652 = 0;
    unsigned char *mem_84352 = NULL;
    int64_t mem_84359_cached_sizze_85653 = 0;
    unsigned char *mem_84359 = NULL;
    int64_t mem_84366_cached_sizze_85654 = 0;
    unsigned char *mem_84366 = NULL;
    int64_t mem_84391_cached_sizze_85655 = 0;
    unsigned char *mem_84391 = NULL;
    int64_t mem_84392_cached_sizze_85656 = 0;
    unsigned char *mem_84392 = NULL;
    int64_t mem_84403_cached_sizze_85657 = 0;
    unsigned char *mem_84403 = NULL;
    int64_t mem_84404_cached_sizze_85658 = 0;
    unsigned char *mem_84404 = NULL;
    int64_t mem_84413_cached_sizze_85659 = 0;
    unsigned char *mem_84413 = NULL;
    int64_t mem_84420_cached_sizze_85660 = 0;
    unsigned char *mem_84420 = NULL;
    int64_t mem_84445_cached_sizze_85661 = 0;
    unsigned char *mem_84445 = NULL;
    int64_t mem_84450_cached_sizze_85662 = 0;
    unsigned char *mem_84450 = NULL;
    int64_t mem_84461_cached_sizze_85663 = 0;
    unsigned char *mem_84461 = NULL;
    int64_t mem_84467_cached_sizze_85664 = 0;
    unsigned char *mem_84467 = NULL;
    int64_t mem_84472_cached_sizze_85665 = 0;
    unsigned char *mem_84472 = NULL;
    int64_t mem_84488_cached_sizze_85666 = 0;
    unsigned char *mem_84488 = NULL;
    int64_t mem_84494_cached_sizze_85667 = 0;
    unsigned char *mem_84494 = NULL;
    int64_t mem_84499_cached_sizze_85668 = 0;
    unsigned char *mem_84499 = NULL;
    int64_t mem_84515_cached_sizze_85669 = 0;
    unsigned char *mem_84515 = NULL;
    int64_t mem_84516_cached_sizze_85670 = 0;
    unsigned char *mem_84516 = NULL;
    int64_t mem_84527_cached_sizze_85671 = 0;
    unsigned char *mem_84527 = NULL;
    int64_t mem_84528_cached_sizze_85672 = 0;
    unsigned char *mem_84528 = NULL;
    int64_t mem_84537_cached_sizze_85673 = 0;
    unsigned char *mem_84537 = NULL;
    int64_t mem_84538_cached_sizze_85674 = 0;
    unsigned char *mem_84538 = NULL;
    int64_t mem_84569_cached_sizze_85675 = 0;
    unsigned char *mem_84569 = NULL;
    int64_t mem_84570_cached_sizze_85676 = 0;
    unsigned char *mem_84570 = NULL;
    int64_t mem_84571_cached_sizze_85677 = 0;
    unsigned char *mem_84571 = NULL;
    int64_t mem_84584_cached_sizze_85678 = 0;
    unsigned char *mem_84584 = NULL;
    int64_t mem_84585_cached_sizze_85679 = 0;
    unsigned char *mem_84585 = NULL;
    int64_t mem_84586_cached_sizze_85680 = 0;
    unsigned char *mem_84586 = NULL;
    int64_t mem_84617_cached_sizze_85681 = 0;
    unsigned char *mem_84617 = NULL;
    int64_t mem_84618_cached_sizze_85682 = 0;
    unsigned char *mem_84618 = NULL;
    int64_t mem_84619_cached_sizze_85683 = 0;
    unsigned char *mem_84619 = NULL;
    int64_t mem_84620_cached_sizze_85684 = 0;
    unsigned char *mem_84620 = NULL;
    int64_t mem_84637_cached_sizze_85685 = 0;
    unsigned char *mem_84637 = NULL;
    int64_t mem_84638_cached_sizze_85686 = 0;
    unsigned char *mem_84638 = NULL;
    int64_t mem_84639_cached_sizze_85687 = 0;
    unsigned char *mem_84639 = NULL;
    int64_t mem_84640_cached_sizze_85688 = 0;
    unsigned char *mem_84640 = NULL;
    int64_t mem_84681_cached_sizze_85689 = 0;
    unsigned char *mem_84681 = NULL;
    int64_t mem_84688_cached_sizze_85690 = 0;
    unsigned char *mem_84688 = NULL;
    int64_t mem_84695_cached_sizze_85691 = 0;
    unsigned char *mem_84695 = NULL;
    int64_t mem_84705_cached_sizze_85692 = 0;
    unsigned char *mem_84705 = NULL;
    int64_t mem_84710_cached_sizze_85693 = 0;
    unsigned char *mem_84710 = NULL;
    int64_t mem_84721_cached_sizze_85694 = 0;
    unsigned char *mem_84721 = NULL;
    int64_t mem_84728_cached_sizze_85695 = 0;
    unsigned char *mem_84728 = NULL;
    int64_t mem_84735_cached_sizze_85696 = 0;
    unsigned char *mem_84735 = NULL;
    int64_t mem_84745_cached_sizze_85697 = 0;
    unsigned char *mem_84745 = NULL;
    int64_t mem_84750_cached_sizze_85698 = 0;
    unsigned char *mem_84750 = NULL;
    int64_t mem_84761_cached_sizze_85699 = 0;
    unsigned char *mem_84761 = NULL;
    int64_t mem_84762_cached_sizze_85700 = 0;
    unsigned char *mem_84762 = NULL;
    int64_t mem_84771_cached_sizze_85701 = 0;
    unsigned char *mem_84771 = NULL;
    int64_t mem_84772_cached_sizze_85702 = 0;
    unsigned char *mem_84772 = NULL;
    int64_t mem_84793_cached_sizze_85703 = 0;
    unsigned char *mem_84793 = NULL;
    int64_t mem_84798_cached_sizze_85704 = 0;
    unsigned char *mem_84798 = NULL;
    int64_t mem_84809_cached_sizze_85705 = 0;
    unsigned char *mem_84809 = NULL;
    int64_t mem_84810_cached_sizze_85706 = 0;
    unsigned char *mem_84810 = NULL;
    int64_t mem_84819_cached_sizze_85707 = 0;
    unsigned char *mem_84819 = NULL;
    int64_t mem_84820_cached_sizze_85708 = 0;
    unsigned char *mem_84820 = NULL;
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
    
    struct memblock mem_param_tmp_85152;
    
    mem_param_tmp_85152.references = NULL;
    
    struct memblock mem_param_tmp_85151;
    
    mem_param_tmp_85151.references = NULL;
    
    struct memblock mem_param_tmp_85150;
    
    mem_param_tmp_85150.references = NULL;
    
    struct memblock mem_param_tmp_85149;
    
    mem_param_tmp_85149.references = NULL;
    
    struct memblock mem_param_tmp_85148;
    
    mem_param_tmp_85148.references = NULL;
    
    struct memblock mem_param_tmp_85147;
    
    mem_param_tmp_85147.references = NULL;
    
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
    
    struct memblock ext_mem_84860;
    
    ext_mem_84860.references = NULL;
    
    struct memblock ext_mem_84861;
    
    ext_mem_84861.references = NULL;
    
    struct memblock ext_mem_84862;
    
    ext_mem_84862.references = NULL;
    
    struct memblock mem_84858;
    
    mem_84858.references = NULL;
    
    struct memblock mem_84856;
    
    mem_84856.references = NULL;
    
    struct memblock mem_84854;
    
    mem_84854.references = NULL;
    
    struct memblock mem_84852;
    
    mem_84852.references = NULL;
    
    struct memblock ext_mem_84849;
    
    ext_mem_84849.references = NULL;
    
    struct memblock ext_mem_84850;
    
    ext_mem_84850.references = NULL;
    
    struct memblock ext_mem_84851;
    
    ext_mem_84851.references = NULL;
    
    struct memblock mem_84847;
    
    mem_84847.references = NULL;
    
    struct memblock mem_84845;
    
    mem_84845.references = NULL;
    
    struct memblock mem_84843;
    
    mem_84843.references = NULL;
    
    struct memblock mem_84841;
    
    mem_84841.references = NULL;
    
    struct memblock mem_param_83458;
    
    mem_param_83458.references = NULL;
    
    struct memblock mem_param_83454;
    
    mem_param_83454.references = NULL;
    
    struct memblock mem_param_83450;
    
    mem_param_83450.references = NULL;
    
    struct memblock mem_param_83446;
    
    mem_param_83446.references = NULL;
    
    struct memblock mem_param_83442;
    
    mem_param_83442.references = NULL;
    
    struct memblock mem_param_83438;
    
    mem_param_83438.references = NULL;
    
    struct memblock mem_param_83434;
    
    mem_param_83434.references = NULL;
    
    struct memblock mem_param_83430;
    
    mem_param_83430.references = NULL;
    
    struct memblock mem_param_83426;
    
    mem_param_83426.references = NULL;
    
    struct memblock mem_param_83422;
    
    mem_param_83422.references = NULL;
    
    struct memblock mem_param_83418;
    
    mem_param_83418.references = NULL;
    
    struct memblock mem_param_83414;
    
    mem_param_83414.references = NULL;
    
    struct memblock mem_param_83410;
    
    mem_param_83410.references = NULL;
    
    struct memblock mem_param_83406;
    
    mem_param_83406.references = NULL;
    
    struct memblock mem_param_83402;
    
    mem_param_83402.references = NULL;
    
    struct memblock mem_param_83398;
    
    mem_param_83398.references = NULL;
    
    struct memblock mem_param_83394;
    
    mem_param_83394.references = NULL;
    
    struct memblock mem_param_83390;
    
    mem_param_83390.references = NULL;
    
    struct memblock mem_param_83386;
    
    mem_param_83386.references = NULL;
    
    struct memblock mem_param_83382;
    
    mem_param_83382.references = NULL;
    
    struct memblock mem_param_83378;
    
    mem_param_83378.references = NULL;
    
    struct memblock mem_param_83374;
    
    mem_param_83374.references = NULL;
    
    struct memblock mem_param_83370;
    
    mem_param_83370.references = NULL;
    
    struct memblock mem_param_83366;
    
    mem_param_83366.references = NULL;
    
    struct memblock mem_param_83362;
    
    mem_param_83362.references = NULL;
    
    struct memblock mem_param_83358;
    
    mem_param_83358.references = NULL;
    
    struct memblock mem_param_83354;
    
    mem_param_83354.references = NULL;
    
    struct memblock ext_mem_85021;
    
    ext_mem_85021.references = NULL;
    
    struct memblock ext_mem_85022;
    
    ext_mem_85022.references = NULL;
    
    struct memblock ext_mem_85023;
    
    ext_mem_85023.references = NULL;
    
    struct memblock ext_mem_85024;
    
    ext_mem_85024.references = NULL;
    
    struct memblock ext_mem_85025;
    
    ext_mem_85025.references = NULL;
    
    struct memblock ext_mem_85026;
    
    ext_mem_85026.references = NULL;
    
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
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_83459_cached_sizze_85540 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83459, &mem_83459_cached_sizze_85540, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83460_cached_sizze_85541 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83460, &mem_83460_cached_sizze_85541, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83469_cached_sizze_85542 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83469, &mem_83469_cached_sizze_85542, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83476_cached_sizze_85543 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83476, &mem_83476_cached_sizze_85543, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83491_cached_sizze_85544 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_83491, &mem_83491_cached_sizze_85544, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83492_cached_sizze_85545 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83492, &mem_83492_cached_sizze_85545, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83501_cached_sizze_85546 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83501, &mem_83501_cached_sizze_85546, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83508_cached_sizze_85547 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_83508, &mem_83508_cached_sizze_85547, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83523_cached_sizze_85548 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83523, &mem_83523_cached_sizze_85548, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83524_cached_sizze_85549 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83524, &mem_83524_cached_sizze_85549, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83533_cached_sizze_85550 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83533, &mem_83533_cached_sizze_85550, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83534_cached_sizze_85551 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83534, &mem_83534_cached_sizze_85551, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83555_cached_sizze_85552 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83555, &mem_83555_cached_sizze_85552, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83556_cached_sizze_85553 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83556, &mem_83556_cached_sizze_85553, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83557_cached_sizze_85554 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83557, &mem_83557_cached_sizze_85554, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83569_cached_sizze_85555 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83569, &mem_83569_cached_sizze_85555, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83570_cached_sizze_85556 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83570, &mem_83570_cached_sizze_85556, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83594_cached_sizze_85557 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83594, &mem_83594_cached_sizze_85557, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83595_cached_sizze_85558 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83595, &mem_83595_cached_sizze_85558, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83596_cached_sizze_85559 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83596, &mem_83596_cached_sizze_85559, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83597_cached_sizze_85560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83597, &mem_83597_cached_sizze_85560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83598_cached_sizze_85561 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83598, &mem_83598_cached_sizze_85561, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83617_cached_sizze_85562 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83617, &mem_83617_cached_sizze_85562, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83618_cached_sizze_85563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83618, &mem_83618_cached_sizze_85563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83619_cached_sizze_85564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83619, &mem_83619_cached_sizze_85564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83656_cached_sizze_85565 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83656, &mem_83656_cached_sizze_85565, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83657_cached_sizze_85566 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83657, &mem_83657_cached_sizze_85566, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83658_cached_sizze_85567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83658, &mem_83658_cached_sizze_85567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83674_cached_sizze_85568 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83674, &mem_83674_cached_sizze_85568, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83675_cached_sizze_85569 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83675, &mem_83675_cached_sizze_85569, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83676_cached_sizze_85570 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83676, &mem_83676_cached_sizze_85570, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83689_cached_sizze_85571 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83689, &mem_83689_cached_sizze_85571, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83690_cached_sizze_85572 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83690, &mem_83690_cached_sizze_85572, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83691_cached_sizze_85573 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83691, &mem_83691_cached_sizze_85573, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83737_cached_sizze_85574 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83737, &mem_83737_cached_sizze_85574, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83738_cached_sizze_85575 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83738, &mem_83738_cached_sizze_85575, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83749_cached_sizze_85576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83749, &mem_83749_cached_sizze_85576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83750_cached_sizze_85577 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83750, &mem_83750_cached_sizze_85577, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83759_cached_sizze_85578 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83759, &mem_83759_cached_sizze_85578, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83760_cached_sizze_85579 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83760, &mem_83760_cached_sizze_85579, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83781_cached_sizze_85580 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83781, &mem_83781_cached_sizze_85580, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83786_cached_sizze_85581 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83786, &mem_83786_cached_sizze_85581, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83797_cached_sizze_85582 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83797, &mem_83797_cached_sizze_85582, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83802_cached_sizze_85583 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83802, &mem_83802_cached_sizze_85583, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83809_cached_sizze_85584 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83809, &mem_83809_cached_sizze_85584, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83816_cached_sizze_85585 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83816, &mem_83816_cached_sizze_85585, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83827_cached_sizze_85586 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83827, &mem_83827_cached_sizze_85586, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83832_cached_sizze_85587 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_83832, &mem_83832_cached_sizze_85587, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83853_cached_sizze_85588 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83853, &mem_83853_cached_sizze_85588, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83854_cached_sizze_85589 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83854, &mem_83854_cached_sizze_85589, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83862_cached_sizze_85590 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83862, &mem_83862_cached_sizze_85590, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83876_cached_sizze_85591 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83876, &mem_83876_cached_sizze_85591, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83881_cached_sizze_85592 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83881, &mem_83881_cached_sizze_85592, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83892_cached_sizze_85593 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83892, &mem_83892_cached_sizze_85593, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83897_cached_sizze_85594 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83897, &mem_83897_cached_sizze_85594, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83908_cached_sizze_85595 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83908, &mem_83908_cached_sizze_85595, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83909_cached_sizze_85596 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83909, &mem_83909_cached_sizze_85596, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83918_cached_sizze_85597 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83918, &mem_83918_cached_sizze_85597, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83919_cached_sizze_85598 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83919, &mem_83919_cached_sizze_85598, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83940_cached_sizze_85599 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83940, &mem_83940_cached_sizze_85599, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83941_cached_sizze_85600 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83941, &mem_83941_cached_sizze_85600, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83949_cached_sizze_85601 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83949, &mem_83949_cached_sizze_85601, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83963_cached_sizze_85602 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83963, &mem_83963_cached_sizze_85602, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83964_cached_sizze_85603 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_83964, &mem_83964_cached_sizze_85603, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83972_cached_sizze_85604 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_83972, &mem_83972_cached_sizze_85604, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83986_cached_sizze_85605 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_83986, &mem_83986_cached_sizze_85605, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_83991_cached_sizze_85606 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_83991, &mem_83991_cached_sizze_85606, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84002_cached_sizze_85607 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84002, &mem_84002_cached_sizze_85607, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84007_cached_sizze_85608 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84007, &mem_84007_cached_sizze_85608, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84018_cached_sizze_85609 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84018, &mem_84018_cached_sizze_85609, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84023_cached_sizze_85610 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84023, &mem_84023_cached_sizze_85610, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84034_cached_sizze_85611 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84034, &mem_84034_cached_sizze_85611, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84035_cached_sizze_85612 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84035, &mem_84035_cached_sizze_85612, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84044_cached_sizze_85613 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84044, &mem_84044_cached_sizze_85613, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84045_cached_sizze_85614 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84045, &mem_84045_cached_sizze_85614, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84058_cached_sizze_85615 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84058, &mem_84058_cached_sizze_85615, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84059_cached_sizze_85616 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84059, &mem_84059_cached_sizze_85616, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84072_cached_sizze_85617 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84072, &mem_84072_cached_sizze_85617, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84073_cached_sizze_85618 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84073, &mem_84073_cached_sizze_85618, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84094_cached_sizze_85619 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84094, &mem_84094_cached_sizze_85619, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84101_cached_sizze_85620 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84101, &mem_84101_cached_sizze_85620, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84106_cached_sizze_85621 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_84106, &mem_84106_cached_sizze_85621, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84117_cached_sizze_85622 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84117, &mem_84117_cached_sizze_85622, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84122_cached_sizze_85623 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84122, &mem_84122_cached_sizze_85623, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84133_cached_sizze_85624 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84133, &mem_84133_cached_sizze_85624, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84134_cached_sizze_85625 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84134, &mem_84134_cached_sizze_85625, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84143_cached_sizze_85626 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84143, &mem_84143_cached_sizze_85626, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84144_cached_sizze_85627 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84144, &mem_84144_cached_sizze_85627, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84165_cached_sizze_85628 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84165, &mem_84165_cached_sizze_85628, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84170_cached_sizze_85629 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84170, &mem_84170_cached_sizze_85629, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84181_cached_sizze_85630 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84181, &mem_84181_cached_sizze_85630, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84186_cached_sizze_85631 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84186, &mem_84186_cached_sizze_85631, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84197_cached_sizze_85632 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84197, &mem_84197_cached_sizze_85632, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84204_cached_sizze_85633 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84204, &mem_84204_cached_sizze_85633, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84211_cached_sizze_85634 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84211, &mem_84211_cached_sizze_85634, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84221_cached_sizze_85635 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84221, &mem_84221_cached_sizze_85635, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84226_cached_sizze_85636 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84226, &mem_84226_cached_sizze_85636, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84237_cached_sizze_85637 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84237, &mem_84237_cached_sizze_85637, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84238_cached_sizze_85638 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84238, &mem_84238_cached_sizze_85638, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84247_cached_sizze_85639 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84247, &mem_84247_cached_sizze_85639, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84248_cached_sizze_85640 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84248, &mem_84248_cached_sizze_85640, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84269_cached_sizze_85641 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84269, &mem_84269_cached_sizze_85641, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84270_cached_sizze_85642 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84270, &mem_84270_cached_sizze_85642, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84281_cached_sizze_85643 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84281, &mem_84281_cached_sizze_85643, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84282_cached_sizze_85644 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84282, &mem_84282_cached_sizze_85644, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84291_cached_sizze_85645 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84291, &mem_84291_cached_sizze_85645, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84298_cached_sizze_85646 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84298, &mem_84298_cached_sizze_85646, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84323_cached_sizze_85647 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84323, &mem_84323_cached_sizze_85647, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84324_cached_sizze_85648 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84324, &mem_84324_cached_sizze_85648, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84335_cached_sizze_85649 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84335, &mem_84335_cached_sizze_85649, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84336_cached_sizze_85650 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84336, &mem_84336_cached_sizze_85650, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84345_cached_sizze_85651 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84345, &mem_84345_cached_sizze_85651, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84352_cached_sizze_85652 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84352, &mem_84352_cached_sizze_85652, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84359_cached_sizze_85653 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84359, &mem_84359_cached_sizze_85653, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84366_cached_sizze_85654 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84366, &mem_84366_cached_sizze_85654, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84391_cached_sizze_85655 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84391, &mem_84391_cached_sizze_85655, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84392_cached_sizze_85656 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84392, &mem_84392_cached_sizze_85656, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84403_cached_sizze_85657 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84403, &mem_84403_cached_sizze_85657, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84404_cached_sizze_85658 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84404, &mem_84404_cached_sizze_85658, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84413_cached_sizze_85659 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84413, &mem_84413_cached_sizze_85659, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84420_cached_sizze_85660 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84420, &mem_84420_cached_sizze_85660, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84445_cached_sizze_85661 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84445, &mem_84445_cached_sizze_85661, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84450_cached_sizze_85662 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84450, &mem_84450_cached_sizze_85662, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84461_cached_sizze_85663 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84461, &mem_84461_cached_sizze_85663, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84467_cached_sizze_85664 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84467, &mem_84467_cached_sizze_85664, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84472_cached_sizze_85665 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84472, &mem_84472_cached_sizze_85665, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84488_cached_sizze_85666 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84488, &mem_84488_cached_sizze_85666, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84494_cached_sizze_85667 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84494, &mem_84494_cached_sizze_85667, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84499_cached_sizze_85668 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84499, &mem_84499_cached_sizze_85668, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84515_cached_sizze_85669 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84515, &mem_84515_cached_sizze_85669, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84516_cached_sizze_85670 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84516, &mem_84516_cached_sizze_85670, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84527_cached_sizze_85671 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84527, &mem_84527_cached_sizze_85671, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84528_cached_sizze_85672 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_84528, &mem_84528_cached_sizze_85672, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84537_cached_sizze_85673 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84537, &mem_84537_cached_sizze_85673, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84538_cached_sizze_85674 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_84538, &mem_84538_cached_sizze_85674, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84569_cached_sizze_85675 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84569, &mem_84569_cached_sizze_85675, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84570_cached_sizze_85676 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84570, &mem_84570_cached_sizze_85676, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84571_cached_sizze_85677 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84571, &mem_84571_cached_sizze_85677, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84584_cached_sizze_85678 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84584, &mem_84584_cached_sizze_85678, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84585_cached_sizze_85679 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84585, &mem_84585_cached_sizze_85679, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84586_cached_sizze_85680 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84586, &mem_84586_cached_sizze_85680, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84617_cached_sizze_85681 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84617, &mem_84617_cached_sizze_85681, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84618_cached_sizze_85682 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84618, &mem_84618_cached_sizze_85682, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84619_cached_sizze_85683 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84619, &mem_84619_cached_sizze_85683, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84620_cached_sizze_85684 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84620, &mem_84620_cached_sizze_85684, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84637_cached_sizze_85685 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84637, &mem_84637_cached_sizze_85685, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84638_cached_sizze_85686 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84638, &mem_84638_cached_sizze_85686, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84639_cached_sizze_85687 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84639, &mem_84639_cached_sizze_85687, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84640_cached_sizze_85688 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84640, &mem_84640_cached_sizze_85688, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84681_cached_sizze_85689 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84681, &mem_84681_cached_sizze_85689, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84688_cached_sizze_85690 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84688, &mem_84688_cached_sizze_85690, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84695_cached_sizze_85691 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84695, &mem_84695_cached_sizze_85691, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84705_cached_sizze_85692 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84705, &mem_84705_cached_sizze_85692, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84710_cached_sizze_85693 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84710, &mem_84710_cached_sizze_85693, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84721_cached_sizze_85694 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84721, &mem_84721_cached_sizze_85694, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84728_cached_sizze_85695 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84728, &mem_84728_cached_sizze_85695, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84735_cached_sizze_85696 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84735, &mem_84735_cached_sizze_85696, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84745_cached_sizze_85697 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84745, &mem_84745_cached_sizze_85697, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84750_cached_sizze_85698 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84750, &mem_84750_cached_sizze_85698, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84761_cached_sizze_85699 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84761, &mem_84761_cached_sizze_85699, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84762_cached_sizze_85700 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_84762, &mem_84762_cached_sizze_85700, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84771_cached_sizze_85701 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84771, &mem_84771_cached_sizze_85701, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84772_cached_sizze_85702 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84772, &mem_84772_cached_sizze_85702, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84793_cached_sizze_85703 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_84793, &mem_84793_cached_sizze_85703, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84798_cached_sizze_85704 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84798, &mem_84798_cached_sizze_85704, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84809_cached_sizze_85705 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84809, &mem_84809_cached_sizze_85705, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84810_cached_sizze_85706 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_84810, &mem_84810_cached_sizze_85706, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84819_cached_sizze_85707 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84819, &mem_84819_cached_sizze_85707, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_84820_cached_sizze_85708 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_84820, &mem_84820_cached_sizze_85708, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:435:5-440:51
    if (memblock_set(ctx, &mem_param_83354, &wdown_mem_83321, "wdown_mem_83321") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83358, &wkey_mem_83322, "wkey_mem_83322") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83362, &wout_mem_83323, "wout_mem_83323") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83366, &wpe_mem_83324, "wpe_mem_83324") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83370, &wqry_mem_83325, "wqry_mem_83325") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83374, &wte_mem_83326, "wte_mem_83326") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83378, &wup_mem_83327, "wup_mem_83327") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83382, &wval_mem_83328, "wval_mem_83328") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83386, &wvoc_mem_83329, "wvoc_mem_83329") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83390, &wdown_mem_83330, "wdown_mem_83330") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83394, &wkey_mem_83331, "wkey_mem_83331") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83398, &wout_mem_83332, "wout_mem_83332") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83402, &wpe_mem_83333, "wpe_mem_83333") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83406, &wqry_mem_83334, "wqry_mem_83334") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83410, &wte_mem_83335, "wte_mem_83335") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83414, &wup_mem_83336, "wup_mem_83336") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83418, &wval_mem_83337, "wval_mem_83337") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83422, &wvoc_mem_83338, "wvoc_mem_83338") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83426, &wdown_mem_83339, "wdown_mem_83339") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83430, &wkey_mem_83340, "wkey_mem_83340") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83434, &wout_mem_83341, "wout_mem_83341") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83438, &wpe_mem_83342, "wpe_mem_83342") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83442, &wqry_mem_83343, "wqry_mem_83343") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83446, &wte_mem_83344, "wte_mem_83344") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83450, &wup_mem_83345, "wup_mem_83345") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83454, &wval_mem_83346, "wval_mem_83346") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_83458, &wvoc_mem_83347, "wvoc_mem_83347") != 0)
        return 1;
    for (int64_t step_76675 = 0; step_76675 < (int64_t) 30000; step_76675++) {
        // futhark/microgpt.fut:437:16-25
        
        int64_t dl_76703 = ((int64_t *) dls_mem_83349.mem)[step_76675];
        
        // futhark/microgpt.fut:350:37-40
        
        int64_t zl_rhs_76704 = sub64(dl_76703, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82474 = 0; i_82474 < (int64_t) 16; i_82474++) {
            // futhark/microgpt.fut:350:25-81
            
            bool cond_78493 = slt64(i_82474, zl_rhs_76704);
            
            // futhark/microgpt.fut:350:56-59
            
            int64_t zeze_lhs_78494 = add64((int64_t) 1, i_82474);
            
            // futhark/microgpt.fut:350:47-60
            
            bool x_78495 = sle64((int64_t) 0, zeze_lhs_78494);
            
            // futhark/microgpt.fut:350:47-60
            
            bool y_78496 = slt64(zeze_lhs_78494, (int64_t) 16);
            
            // futhark/microgpt.fut:350:47-60
            
            bool bounds_check_78497 = x_78495 && y_78496;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_78498 = !cond_78493;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_78499 = bounds_check_78497 || loop_not_taken_78498;
            
            // futhark/microgpt.fut:350:47-60
            
            bool index_certs_78500;
            
            if (!protect_assert_disj_78499) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_78494, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:350:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:350:3-83\n   #6  futhark/microgpt.fut:408:18-38\n   #7  futhark/microgpt.fut:418:26-424:31\n   #8  futhark/microgpt.fut:440:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_78515 = ((int64_t *) seqs_mem_83350.mem)[step_76675 * (int64_t) 16 + i_82474];
            
            // futhark/microgpt.fut:410:37-51
            
            bool x_78516 = sle64((int64_t) 0, tmp_78515);
            
            // futhark/microgpt.fut:410:37-51
            
            bool y_78517 = slt64(tmp_78515, (int64_t) 27);
            
            // futhark/microgpt.fut:410:37-51
            
            bool bounds_check_78518 = x_78516 && y_78517;
            
            // futhark/microgpt.fut:410:37-51
            
            bool index_certs_78519;
            
            if (!bounds_check_78518) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_78515, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:410:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:410:16-55\n   #6  futhark/microgpt.fut:418:26-424:31\n   #7  futhark/microgpt.fut:440:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:350:47-60
            
            int64_t zeze_lhs_78501;
            
            if (cond_78493) {
                int64_t x_82283 = ((int64_t *) seqs_mem_83350.mem)[step_76675 * (int64_t) 16 + zeze_lhs_78494];
                
                zeze_lhs_78501 = x_82283;
            } else {
                zeze_lhs_78501 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82464 = 0; i_82464 < (int64_t) 27; i_82464++) {
                // futhark/microgpt.fut:350:61-65
                
                bool cond_t_res_78505 = zeze_lhs_78501 == i_82464;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_78506 = cond_78493 && cond_t_res_78505;
                
                // futhark/microgpt.fut:350:25-81
                
                double lifted_lambda_res_78507;
                
                if (x_78506) {
                    lifted_lambda_res_78507 = 1.0;
                } else {
                    lifted_lambda_res_78507 = 0.0;
                }
                ((double *) mem_83469)[i_82464] = lifted_lambda_res_78507;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82468 = 0; i_82468 < (int64_t) 16; i_82468++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_78526 = ((double *) mem_param_83374.mem)[tmp_78515 * (int64_t) 16 + i_82468];
                
                ((double *) mem_83476)[i_82468] = lifted_lambda_res_78526;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83459, i_82474 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83476, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83460, i_82474 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83469, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82489 = 0; i_82489 < (int64_t) 16; i_82489++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82479 = 0; i_82479 < (int64_t) 16; i_82479++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78551 = ((double *) mem_param_83366.mem)[i_82489 * (int64_t) 16 + i_82479];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_78552 = ((double *) mem_83459)[i_82489 * (int64_t) 16 + i_82479];
                
                // futhark/microgpt.fut:210:35-63
                
                double zp_res_78553 = zp_lhs_78551 + zp_rhs_78552;
                
                ((double *) mem_83501)[i_82479] = zp_res_78553;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82483 = 0; i_82483 < (int64_t) 27; i_82483++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78567 = ((double *) mem_83460)[i_82489 * (int64_t) 27 + i_82483];
                
                // futhark/microgpt.fut:242:51-87
                
                double zt_res_78568 = -6.25e-2 * zt_rhs_78567;
                
                ((double *) mem_83508)[i_82483] = zt_res_78568;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83491, i_82489 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83508, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83492, i_82489 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83501, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82503 = 0; i_82503 < (int64_t) 16; i_82503++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78587;
            double r_78589 = 0.0;
            
            for (int64_t i_78588 = 0; i_78588 < (int64_t) 16; i_78588++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78590 = ((double *) mem_83492)[i_82503 * (int64_t) 16 + i_78588];
                
                // futhark/microgpt.fut:211:58-83
                
                double zt_res_78591 = zt_lhs_78590 * zt_lhs_78590;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78592 = r_78589 + zt_res_78591;
                double r_tmp_85211 = zp_res_78592;
                
                r_78589 = r_tmp_85211;
            }
            defunc_0_lifted_lambda_res_78587 = r_78589;
            // futhark/microgpt.fut:211:40-101
            
            double zs_res_78593 = defunc_0_lifted_lambda_res_78587 / 16.0;
            
            // futhark/microgpt.fut:212:23-53
            
            double zp_res_78594 = 1.0e-5 + zs_res_78593;
            
            // futhark/microgpt.fut:212:15-53
            
            double sqrt_res_78595 = futrts_sqrt64(zp_res_78594);
            
            // futhark/microgpt.fut:213:39-49
            
            double zs_res_78596 = 1.0 / sqrt_res_78595;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82496 = 0; i_82496 < (int64_t) 16; i_82496++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80592 = ((double *) mem_83492)[i_82503 * (int64_t) 16 + i_82496];
                
                // futhark/microgpt.fut:213:23-49
                
                double zt_res_80593 = zs_res_78596 * zt_lhs_80592;
                
                // futhark/microgpt.fut:285:53-86
                
                double zt_res_80601 = zt_lhs_80592 * zt_lhs_80592;
                
                ((double *) mem_83533)[i_82496] = zt_res_80601;
                ((double *) mem_83534)[i_82496] = zt_res_80593;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83523, i_82503 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83533, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83524, i_82503 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83534, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82519 = 0; i_82519 < (int64_t) 16; i_82519++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78695;
            double r_78697 = 0.0;
            
            for (int64_t i_78696 = 0; i_78696 < (int64_t) 16; i_78696++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78698 = ((double *) mem_83524)[i_82519 * (int64_t) 16 + i_78696];
                
                // futhark/microgpt.fut:214:61-90
                
                double zt_res_78699 = zt_lhs_78698 * zt_lhs_78698;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78700 = r_78697 + zt_res_78699;
                double r_tmp_85217 = zp_res_78700;
                
                r_78697 = r_tmp_85217;
            }
            defunc_0_lifted_lambda_res_78695 = r_78697;
            // futhark/microgpt.fut:214:42-108
            
            double zs_res_78701 = defunc_0_lifted_lambda_res_78695 / 16.0;
            
            // futhark/microgpt.fut:215:24-55
            
            double zp_res_78702 = 1.0e-5 + zs_res_78701;
            
            // futhark/microgpt.fut:215:16-55
            
            double sqrt_res_78703 = futrts_sqrt64(zp_res_78702);
            
            // futhark/microgpt.fut:216:42-53
            
            double zs_res_78704 = 1.0 / sqrt_res_78703;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82510 = 0; i_82510 < (int64_t) 16; i_82510++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_80621 = ((double *) mem_83524)[i_82519 * (int64_t) 16 + i_82510];
                
                // futhark/microgpt.fut:216:24-53
                
                double zt_res_80622 = zs_res_78704 * zt_lhs_80621;
                
                // futhark/microgpt.fut:278:53-86
                
                double zt_res_80630 = zt_lhs_80621 * zt_lhs_80621;
                
                ((double *) mem_83569)[i_82510] = zt_res_80630;
                ((double *) mem_83570)[i_82510] = zt_res_80622;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78738;
            double r_78740 = 0.0;
            
            for (int64_t i_78739 = 0; i_78739 < (int64_t) 16; i_78739++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_78741 = ((double *) mem_83523)[i_82519 * (int64_t) 16 + i_78739];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78742 = r_78740 + lifted_lambda_res_78741;
                double r_tmp_85220 = zp_res_78742;
                
                r_78740 = r_tmp_85220;
            }
            defunc_0_lifted_lambda_res_78738 = r_78740;
            // futhark/microgpt.fut:286:34-86
            
            double zs_res_78743 = defunc_0_lifted_lambda_res_78738 / 16.0;
            
            ((double *) mem_83555)[i_82519] = zs_res_78743;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83556, i_82519 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83569, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83557, i_82519 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83570, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82543 = 0; i_82543 < (int64_t) 16; i_82543++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82529 = 0; i_82529 < (int64_t) 16; i_82529++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80693;
                double r_80695 = 0.0;
                
                for (int64_t i_80694 = 0; i_80694 < (int64_t) 16; i_80694++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80696 = ((double *) mem_param_83370.mem)[i_82529 * (int64_t) 16 + i_80694];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80697 = ((double *) mem_83557)[i_82543 * (int64_t) 16 + i_80694];
                    
                    // futhark/microgpt.fut:217:69-100
                    
                    double zt_res_80698 = zt_lhs_80696 * zt_rhs_80697;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80699 = r_80695 + zt_res_80698;
                    double r_tmp_85229 = zp_res_80699;
                    
                    r_80695 = r_tmp_85229;
                }
                defunc_0_lifted_lambda_res_80693 = r_80695;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80706;
                double r_80708 = 0.0;
                
                for (int64_t i_80707 = 0; i_80707 < (int64_t) 16; i_80707++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80709 = ((double *) mem_param_83358.mem)[i_82529 * (int64_t) 16 + i_80707];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80710 = ((double *) mem_83557)[i_82543 * (int64_t) 16 + i_80707];
                    
                    // futhark/microgpt.fut:218:69-100
                    
                    double zt_res_80711 = zt_lhs_80709 * zt_rhs_80710;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80712 = r_80708 + zt_res_80711;
                    double r_tmp_85230 = zp_res_80712;
                    
                    r_80708 = r_tmp_85230;
                }
                defunc_0_lifted_lambda_res_80706 = r_80708;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_80722;
                double r_80724 = 0.0;
                
                for (int64_t i_80723 = 0; i_80723 < (int64_t) 16; i_80723++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_80725 = ((double *) mem_param_83382.mem)[i_82529 * (int64_t) 16 + i_80723];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_80726 = ((double *) mem_83557)[i_82543 * (int64_t) 16 + i_80723];
                    
                    // futhark/microgpt.fut:219:69-100
                    
                    double zt_res_80727 = zt_lhs_80725 * zt_rhs_80726;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_80728 = r_80724 + zt_res_80727;
                    double r_tmp_85231 = zp_res_80728;
                    
                    r_80724 = r_tmp_85231;
                }
                defunc_0_lifted_lambda_res_80722 = r_80724;
                ((double *) mem_83617)[i_82529] = defunc_0_lifted_lambda_res_80722;
                ((double *) mem_83618)[i_82529] = defunc_0_lifted_lambda_res_80706;
                ((double *) mem_83619)[i_82529] = defunc_0_lifted_lambda_res_80693;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79085;
            double r_79087 = 0.0;
            
            for (int64_t i_79086 = 0; i_79086 < (int64_t) 16; i_79086++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79088 = ((double *) mem_83556)[i_82543 * (int64_t) 16 + i_79086];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79089 = r_79087 + lifted_lambda_res_79088;
                double r_tmp_85232 = zp_res_79089;
                
                r_79087 = r_tmp_85232;
            }
            defunc_0_lifted_lambda_res_79085 = r_79087;
            // futhark/microgpt.fut:279:34-86
            
            double zs_res_79090 = defunc_0_lifted_lambda_res_79085 / 16.0;
            
            // futhark/microgpt.fut:287:41-51
            
            double zp_lhs_79104 = ((double *) mem_83555)[i_82543];
            
            // futhark/microgpt.fut:287:41-79
            
            double zp_res_79105 = 1.0e-5 + zp_lhs_79104;
            
            // futhark/microgpt.fut:287:33-79
            
            double sqrt_res_79106 = futrts_sqrt64(zp_res_79105);
            
            ((double *) mem_83594)[i_82543] = sqrt_res_79106;
            ((double *) mem_83595)[i_82543] = zs_res_79090;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83596, i_82543 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83617, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83597, i_82543 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83618, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83598, i_82543 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83619, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82575 = 0; i_82575 < (int64_t) 4; i_82575++) {
            // futhark/microgpt.fut:220:81-84
            
            int64_t zp_lhs_79178 = mul64((int64_t) 4, i_82575);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82565 = 0; i_82565 < (int64_t) 16; i_82565++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82555 = 0; i_82555 < (int64_t) 4; i_82555++) {
                    // futhark/microgpt.fut:220:86-91
                    
                    int64_t tmp_80886 = add64(zp_lhs_79178, i_82555);
                    
                    // futhark/microgpt.fut:220:66-93
                    
                    bool x_80887 = sle64((int64_t) 0, tmp_80886);
                    
                    // futhark/microgpt.fut:220:66-93
                    
                    bool y_80888 = slt64(tmp_80886, (int64_t) 16);
                    
                    // futhark/microgpt.fut:220:66-93
                    
                    bool bounds_check_80889 = x_80887 && y_80888;
                    
                    // futhark/microgpt.fut:220:66-93
                    
                    bool index_certs_80890;
                    
                    if (!bounds_check_80889) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_80886, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:220:66-93\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:220:49-94\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:220:30-96\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:220:12-98\n   #10 futhark/microgpt.fut:413:5-76\n   #11 futhark/microgpt.fut:418:26-424:31\n   #12 futhark/microgpt.fut:440:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80891 = ((double *) mem_83598)[i_82565 * (int64_t) 16 + tmp_80886];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80899 = ((double *) mem_83597)[i_82565 * (int64_t) 16 + tmp_80886];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_80910 = ((double *) mem_83596)[i_82565 * (int64_t) 16 + tmp_80886];
                    
                    ((double *) mem_83689)[i_82555] = lifted_lambda_res_80910;
                    ((double *) mem_83690)[i_82555] = lifted_lambda_res_80899;
                    ((double *) mem_83691)[i_82555] = lifted_lambda_res_80891;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83674, i_82565 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83689, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83675, i_82565 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83690, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83676, i_82565 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83691, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83656, i_82575 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83674, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83657, i_82575 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83675, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83658, i_82575 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83676, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82631 = 0; i_82631 < (int64_t) 4; i_82631++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82590 = 0; i_82590 < (int64_t) 16; i_82590++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82583 = 0; i_82583 < (int64_t) 16; i_82583++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_80989;
                    double r_80991 = 0.0;
                    
                    for (int64_t i_80990 = 0; i_80990 < (int64_t) 4; i_80990++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_80992 = ((double *) mem_83658)[i_82631 * (int64_t) 64 + i_82590 * (int64_t) 4 + i_80990];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_80993 = ((double *) mem_83657)[i_82631 * (int64_t) 64 + i_82583 * (int64_t) 4 + i_80990];
                        
                        // futhark/microgpt.fut:223:97-138
                        
                        double zt_res_80994 = zt_lhs_80992 * zt_rhs_80993;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_80995 = r_80991 + zt_res_80994;
                        double r_tmp_85248 = zp_res_80995;
                        
                        r_80991 = r_tmp_85248;
                    }
                    defunc_0_lifted_lambda_res_80989 = r_80991;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81002;
                    double r_81004 = 0.0;
                    
                    for (int64_t i_81003 = 0; i_81003 < (int64_t) 4; i_81003++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81005 = ((double *) mem_83658)[i_82631 * (int64_t) 64 + i_82590 * (int64_t) 4 + i_81003];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81006 = ((double *) mem_83657)[i_82631 * (int64_t) 64 + i_82583 * (int64_t) 4 + i_81003];
                        
                        // futhark/microgpt.fut:262:91-138
                        
                        double zt_res_81007 = zt_lhs_81005 * zt_rhs_81006;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81008 = r_81004 + zt_res_81007;
                        double r_tmp_85249 = zp_res_81008;
                        
                        r_81004 = r_tmp_85249;
                    }
                    defunc_0_lifted_lambda_res_81002 = r_81004;
                    ((double *) mem_83759)[i_82583] = defunc_0_lifted_lambda_res_81002;
                    ((double *) mem_83760)[i_82583] = defunc_0_lifted_lambda_res_80989;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83749, i_82590 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83759, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83750, i_82590 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83760, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82599 = 0; i_82599 < (int64_t) 16; i_82599++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82595 = 0; i_82595 < (int64_t) 16; i_82595++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_79287 = ((double *) mem_83750)[i_82599 * (int64_t) 16 + i_82595];
                    
                    // futhark/microgpt.fut:224:43-70
                    
                    double zs_res_79288 = zs_lhs_79287 / 2.0;
                    double zp_rhs_79289 = ((double *) masks_mem_83348.mem)[step_76675 * (int64_t) 256 + i_82599 * (int64_t) 16 + i_82595];
                    
                    // futhark/microgpt.fut:224:57-90
                    
                    double zp_res_79290 = zs_res_79288 + zp_rhs_79289;
                    
                    ((double *) mem_83786)[i_82595] = zp_res_79290;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83781, i_82599 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83786, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82617 = 0; i_82617 < (int64_t) 16; i_82617++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_82304;
                double redout_82601 = -INFINITY;
                
                for (int64_t i_82602 = 0; i_82602 < (int64_t) 16; i_82602++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81026 = ((double *) mem_83781)[i_82617 * (int64_t) 16 + i_82602];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_79311 = fmax64(lifted_lambda_res_81026, redout_82601);
                    double redout_tmp_85253 = max_res_79311;
                    
                    redout_82601 = redout_tmp_85253;
                }
                defunc_0_reduce_res_82304 = redout_82601;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_79312 = -defunc_0_reduce_res_82304;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82605 = 0; i_82605 < (int64_t) 16; i_82605++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_79319 = ((double *) mem_83781)[i_82617 * (int64_t) 16 + i_82605];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_79320 = neg_res_79312 + lifted_lambda_res_79319;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_79321 = futrts_exp64(zp_res_79320);
                    
                    ((double *) mem_83802)[i_82605] = exp_res_79321;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79323;
                double r_79325 = 0.0;
                
                for (int64_t i_79324 = 0; i_79324 < (int64_t) 16; i_79324++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_79326 = ((double *) mem_83802)[i_79324];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79327 = r_79325 + lifted_lambda_res_79326;
                    double r_tmp_85255 = zp_res_79327;
                    
                    r_79325 = r_tmp_85255;
                }
                defunc_0_lifted_lambda_res_79323 = r_79325;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82609 = 0; i_82609 < (int64_t) 16; i_82609++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_79334 = ((double *) mem_83802)[i_82609];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_79335 = zs_lhs_79334 / defunc_0_lifted_lambda_res_79323;
                    
                    ((double *) mem_83809)[i_82609] = zs_res_79335;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82613 = 0; i_82613 < (int64_t) 16; i_82613++) {
                    // futhark/microgpt.fut:226:23-31
                    
                    double lifted_lambda_res_79343 = ((double *) mem_83809)[i_82613];
                    
                    ((double *) mem_83816)[i_82613] = lifted_lambda_res_79343;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83797, i_82617 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83816, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82625 = 0; i_82625 < (int64_t) 16; i_82625++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82621 = 0; i_82621 < (int64_t) 4; i_82621++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_79358;
                    double r_79360 = 0.0;
                    
                    for (int64_t i_79359 = 0; i_79359 < (int64_t) 16; i_79359++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_79361 = ((double *) mem_83797)[i_82625 * (int64_t) 16 + i_79359];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_79362 = ((double *) mem_83656)[i_82631 * (int64_t) 64 + i_79359 * (int64_t) 4 + i_82621];
                        
                        // futhark/microgpt.fut:227:61-97
                        
                        double zt_res_79363 = zt_lhs_79361 * zt_rhs_79362;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_79364 = r_79360 + zt_res_79363;
                        double r_tmp_85260 = zp_res_79364;
                        
                        r_79360 = r_tmp_85260;
                    }
                    defunc_0_lifted_lambda_res_79358 = r_79360;
                    ((double *) mem_83832)[i_82621] = defunc_0_lifted_lambda_res_79358;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_83827, i_82625 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83832, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83737, i_82631 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_83749, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_83738, i_82631 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_83827, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82642 = 0; i_82642 < (int64_t) 16; i_82642++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82636 = 0; i_82636 < (int64_t) 16; i_82636++) {
                // futhark/microgpt.fut:228:58-61
                
                int64_t tmp_79413 = sdiv64(i_82636, (int64_t) 4);
                
                // futhark/microgpt.fut:228:49-63
                
                bool x_79414 = sle64((int64_t) 0, tmp_79413);
                
                // futhark/microgpt.fut:228:49-63
                
                bool y_79415 = slt64(tmp_79413, (int64_t) 4);
                
                // futhark/microgpt.fut:228:49-63
                
                bool bounds_check_79416 = x_79414 && y_79415;
                
                // futhark/microgpt.fut:228:49-63
                
                bool index_certs_79417;
                
                if (!bounds_check_79416) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79413, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:228:49-63\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:228:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:228:12-82\n   #7  futhark/microgpt.fut:413:5-76\n   #8  futhark/microgpt.fut:418:26-424:31\n   #9  futhark/microgpt.fut:440:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:228:74-77
                
                int64_t tmp_79418 = smod64(i_82636, (int64_t) 4);
                
                // futhark/microgpt.fut:228:49-79
                
                bool x_79419 = sle64((int64_t) 0, tmp_79418);
                
                // futhark/microgpt.fut:228:49-79
                
                bool y_79420 = slt64(tmp_79418, (int64_t) 4);
                
                // futhark/microgpt.fut:228:49-79
                
                bool bounds_check_79421 = x_79419 && y_79420;
                
                // futhark/microgpt.fut:228:49-79
                
                bool index_certs_79422;
                
                if (!bounds_check_79421) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_79418, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:228:49-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:228:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:228:12-82\n   #7  futhark/microgpt.fut:413:5-76\n   #8  futhark/microgpt.fut:418:26-424:31\n   #9  futhark/microgpt.fut:440:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_79423 = ((double *) mem_83738)[tmp_79413 * (int64_t) 64 + i_82642 * (int64_t) 4 + tmp_79418];
                
                ((double *) mem_83862)[i_82636] = lifted_lambda_res_79423;
            }
            // futhark/microgpt.fut:280:41-51
            
            double zp_lhs_79431 = ((double *) mem_83595)[i_82642];
            
            // futhark/microgpt.fut:280:41-79
            
            double zp_res_79432 = 1.0e-5 + zp_lhs_79431;
            
            // futhark/microgpt.fut:280:33-79
            
            double sqrt_res_79433 = futrts_sqrt64(zp_res_79432);
            
            ((double *) mem_83853)[i_82642] = sqrt_res_79433;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83854, i_82642 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83862, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82651 = 0; i_82651 < (int64_t) 16; i_82651++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82647 = 0; i_82647 < (int64_t) 16; i_82647++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77089;
                double r_77091 = 0.0;
                
                for (int64_t i_77090 = 0; i_77090 < (int64_t) 16; i_77090++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77092 = ((double *) mem_param_83362.mem)[i_82647 * (int64_t) 16 + i_77090];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77093 = ((double *) mem_83854)[i_82651 * (int64_t) 16 + i_77090];
                    
                    // futhark/microgpt.fut:229:69-101
                    
                    double zt_res_77094 = zt_lhs_77092 * zt_rhs_77093;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77095 = r_77091 + zt_res_77094;
                    double r_tmp_85266 = zp_res_77095;
                    
                    r_77091 = r_tmp_85266;
                }
                defunc_0_lifted_lambda_res_77089 = r_77091;
                ((double *) mem_83881)[i_82647] = defunc_0_lifted_lambda_res_77089;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83876, i_82651 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83881, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82659 = 0; i_82659 < (int64_t) 16; i_82659++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82655 = 0; i_82655 < (int64_t) 16; i_82655++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77110 = ((double *) mem_83876)[i_82659 * (int64_t) 16 + i_82655];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77111 = ((double *) mem_83524)[i_82659 * (int64_t) 16 + i_82655];
                
                // futhark/microgpt.fut:230:38-68
                
                double zp_res_77112 = zp_lhs_77110 + zp_rhs_77111;
                
                ((double *) mem_83897)[i_82655] = zp_res_77112;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83892, i_82659 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83897, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82672 = 0; i_82672 < (int64_t) 16; i_82672++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79451;
            double r_79453 = 0.0;
            
            for (int64_t i_79452 = 0; i_79452 < (int64_t) 16; i_79452++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_79454 = ((double *) mem_83892)[i_82672 * (int64_t) 16 + i_79452];
                
                // futhark/microgpt.fut:231:62-93
                
                double zt_res_79455 = zt_lhs_79454 * zt_lhs_79454;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79456 = r_79453 + zt_res_79455;
                double r_tmp_85271 = zp_res_79456;
                
                r_79453 = r_tmp_85271;
            }
            defunc_0_lifted_lambda_res_79451 = r_79453;
            // futhark/microgpt.fut:231:43-111
            
            double zs_res_79457 = defunc_0_lifted_lambda_res_79451 / 16.0;
            
            // futhark/microgpt.fut:232:24-55
            
            double zp_res_79458 = 1.0e-5 + zs_res_79457;
            
            // futhark/microgpt.fut:232:16-55
            
            double sqrt_res_79459 = futrts_sqrt64(zp_res_79458);
            
            // futhark/microgpt.fut:233:43-54
            
            double zs_res_79460 = 1.0 / sqrt_res_79459;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82665 = 0; i_82665 < (int64_t) 16; i_82665++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81067 = ((double *) mem_83892)[i_82672 * (int64_t) 16 + i_82665];
                
                // futhark/microgpt.fut:233:24-54
                
                double zt_res_81068 = zs_res_79460 * zt_lhs_81067;
                
                // futhark/microgpt.fut:253:53-88
                
                double zt_res_81076 = zt_lhs_81067 * zt_lhs_81067;
                
                ((double *) mem_83918)[i_82665] = zt_res_81076;
                ((double *) mem_83919)[i_82665] = zt_res_81068;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83908, i_82672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83918, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83909, i_82672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83919, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82683 = 0; i_82683 < (int64_t) 16; i_82683++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82677 = 0; i_82677 < (int64_t) 64; i_82677++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_79508;
                double r_79510 = 0.0;
                
                for (int64_t i_79509 = 0; i_79509 < (int64_t) 16; i_79509++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_79511 = ((double *) mem_param_83378.mem)[i_82677 * (int64_t) 16 + i_79509];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_79512 = ((double *) mem_83909)[i_82683 * (int64_t) 16 + i_79509];
                    
                    // futhark/microgpt.fut:234:69-100
                    
                    double zt_res_79513 = zt_lhs_79511 * zt_rhs_79512;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_79514 = r_79510 + zt_res_79513;
                    double r_tmp_85277 = zp_res_79514;
                    
                    r_79510 = r_tmp_85277;
                }
                defunc_0_lifted_lambda_res_79508 = r_79510;
                ((double *) mem_83949)[i_82677] = defunc_0_lifted_lambda_res_79508;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79522;
            double r_79524 = 0.0;
            
            for (int64_t i_79523 = 0; i_79523 < (int64_t) 16; i_79523++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_79525 = ((double *) mem_83908)[i_82683 * (int64_t) 16 + i_79523];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79526 = r_79524 + lifted_lambda_res_79525;
                double r_tmp_85278 = zp_res_79526;
                
                r_79524 = r_tmp_85278;
            }
            defunc_0_lifted_lambda_res_79522 = r_79524;
            // futhark/microgpt.fut:254:34-86
            
            double zs_res_79527 = defunc_0_lifted_lambda_res_79522 / 16.0;
            
            ((double *) mem_83940)[i_82683] = zs_res_79527;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83941, i_82683 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83949, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82694 = 0; i_82694 < (int64_t) 16; i_82694++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82688 = 0; i_82688 < (int64_t) 64; i_82688++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_79551 = ((double *) mem_83941)[i_82694 * (int64_t) 64 + i_82688];
                
                // futhark/microgpt.fut:235:38-62
                
                double max_res_79552 = fmax64(0.0, max_arg0_79551);
                
                ((double *) mem_83972)[i_82688] = max_res_79552;
            }
            // futhark/microgpt.fut:255:41-51
            
            double zp_lhs_79560 = ((double *) mem_83940)[i_82694];
            
            // futhark/microgpt.fut:255:41-79
            
            double zp_res_79561 = 1.0e-5 + zp_lhs_79560;
            
            // futhark/microgpt.fut:255:33-79
            
            double sqrt_res_79562 = futrts_sqrt64(zp_res_79561);
            
            ((double *) mem_83963)[i_82694] = sqrt_res_79562;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83964, i_82694 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82703 = 0; i_82703 < (int64_t) 16; i_82703++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82699 = 0; i_82699 < (int64_t) 16; i_82699++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77190;
                double r_77192 = 0.0;
                
                for (int64_t i_77191 = 0; i_77191 < (int64_t) 64; i_77191++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77193 = ((double *) mem_param_83354.mem)[i_82699 * (int64_t) 64 + i_77191];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77194 = ((double *) mem_83964)[i_82703 * (int64_t) 64 + i_77191];
                    
                    // futhark/microgpt.fut:236:69-102
                    
                    double zt_res_77195 = zt_lhs_77193 * zt_rhs_77194;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77196 = r_77192 + zt_res_77195;
                    double r_tmp_85284 = zp_res_77196;
                    
                    r_77192 = r_tmp_85284;
                }
                defunc_0_lifted_lambda_res_77190 = r_77192;
                ((double *) mem_83991)[i_82699] = defunc_0_lifted_lambda_res_77190;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_83986, i_82703 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_83991, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82711 = 0; i_82711 < (int64_t) 16; i_82711++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82707 = 0; i_82707 < (int64_t) 16; i_82707++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77211 = ((double *) mem_83986)[i_82711 * (int64_t) 16 + i_82707];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77212 = ((double *) mem_83892)[i_82711 * (int64_t) 16 + i_82707];
                
                // futhark/microgpt.fut:237:38-69
                
                double zp_res_77213 = zp_lhs_77211 + zp_rhs_77212;
                
                ((double *) mem_84007)[i_82707] = zp_res_77213;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84002, i_82711 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84007, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82719 = 0; i_82719 < (int64_t) 16; i_82719++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82715 = 0; i_82715 < (int64_t) 27; i_82715++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77228;
                double r_77230 = 0.0;
                
                for (int64_t i_77229 = 0; i_77229 < (int64_t) 16; i_77229++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77231 = ((double *) mem_param_83386.mem)[i_82715 * (int64_t) 16 + i_77229];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77232 = ((double *) mem_84002)[i_82719 * (int64_t) 16 + i_77229];
                    
                    // futhark/microgpt.fut:238:69-101
                    
                    double zt_res_77233 = zt_lhs_77231 * zt_rhs_77232;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77234 = r_77230 + zt_res_77233;
                    double r_tmp_85289 = zp_res_77234;
                    
                    r_77230 = r_tmp_85289;
                }
                defunc_0_lifted_lambda_res_77228 = r_77230;
                ((double *) mem_84023)[i_82715] = defunc_0_lifted_lambda_res_77228;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84018, i_82719 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84023, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82749 = 0; i_82749 < (int64_t) 16; i_82749++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_82324;
            double defunc_0_reduce_res_82325;
            double redout_82721;
            double redout_82722;
            
            redout_82721 = -INFINITY;
            redout_82722 = -INFINITY;
            for (int64_t i_82723 = 0; i_82723 < (int64_t) 27; i_82723++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81144 = ((double *) mem_84018)[i_82749 * (int64_t) 27 + i_82723];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79592 = fmax64(lifted_lambda_res_81144, redout_82721);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_79644 = fmax64(lifted_lambda_res_81144, redout_82722);
                double redout_tmp_85292 = max_res_79592;
                double redout_tmp_85293 = max_res_79644;
                
                redout_82721 = redout_tmp_85292;
                redout_82722 = redout_tmp_85293;
            }
            defunc_0_reduce_res_82324 = redout_82721;
            defunc_0_reduce_res_82325 = redout_82722;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79593 = -defunc_0_reduce_res_82324;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_79645 = -defunc_0_reduce_res_82325;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82728 = 0; i_82728 < (int64_t) 27; i_82728++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_81183 = ((double *) mem_84018)[i_82749 * (int64_t) 27 + i_82728];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81184 = neg_res_79593 + lifted_lambda_res_81183;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81185 = futrts_exp64(zp_res_81184);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_81193 = neg_res_79645 + lifted_lambda_res_81183;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_81194 = futrts_exp64(zp_res_81193);
                
                ((double *) mem_84044)[i_82728] = exp_res_81194;
                ((double *) mem_84045)[i_82728] = exp_res_81185;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79604;
            double r_79606 = 0.0;
            
            for (int64_t i_79605 = 0; i_79605 < (int64_t) 27; i_79605++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79607 = ((double *) mem_84045)[i_79605];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79608 = r_79606 + lifted_lambda_res_79607;
                double r_tmp_85296 = zp_res_79608;
                
                r_79606 = r_tmp_85296;
            }
            defunc_0_lifted_lambda_res_79604 = r_79606;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_79656;
            double r_79658 = 0.0;
            
            for (int64_t i_79657 = 0; i_79657 < (int64_t) 27; i_79657++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_79659 = ((double *) mem_84044)[i_79657];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_79660 = r_79658 + lifted_lambda_res_79659;
                double r_tmp_85297 = zp_res_79660;
                
                r_79658 = r_tmp_85297;
            }
            defunc_0_lifted_lambda_res_79656 = r_79658;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82735 = 0; i_82735 < (int64_t) 27; i_82735++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81212 = ((double *) mem_84045)[i_82735];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81213 = zs_lhs_81212 / defunc_0_lifted_lambda_res_79604;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_81220 = ((double *) mem_84044)[i_82735];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_81221 = zs_lhs_81220 / defunc_0_lifted_lambda_res_79656;
                
                ((double *) mem_84058)[i_82735] = zs_res_81221;
                ((double *) mem_84059)[i_82735] = zs_res_81213;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82742 = 0; i_82742 < (int64_t) 27; i_82742++) {
                // futhark/microgpt.fut:244:24-34
                
                double lifted_lambda_res_81239 = ((double *) mem_84059)[i_82742];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81246 = ((double *) mem_83491)[i_82749 * (int64_t) 27 + i_82742];
                
                // futhark/microgpt.fut:246:4-14
                
                double zs_rhs_81247 = ((double *) mem_84058)[i_82742];
                
                // futhark/microgpt.fut:245:74-246:14
                
                double zs_res_81248 = 1.0 / zs_rhs_81247;
                
                // futhark/microgpt.fut:245:53-246:14
                
                double zt_res_81249 = zt_lhs_81246 * zs_res_81248;
                
                ((double *) mem_84072)[i_82742] = zt_res_81249;
                ((double *) mem_84073)[i_82742] = lifted_lambda_res_81239;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84034, i_82749 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84072, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84035, i_82749 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84073, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82754 = 0; i_82754 < (int64_t) 16; i_82754++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77368;
            double r_77370 = 0.0;
            
            for (int64_t i_77369 = 0; i_77369 < (int64_t) 27; i_77369++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77371 = ((double *) mem_84034)[i_82754 * (int64_t) 27 + i_77369];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77372 = ((double *) mem_84035)[i_82754 * (int64_t) 27 + i_77369];
                
                // futhark/microgpt.fut:247:53-90
                
                double zt_res_77373 = zt_lhs_77371 * zt_rhs_77372;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77374 = r_77370 + zt_res_77373;
                double r_tmp_85303 = zp_res_77374;
                
                r_77370 = r_tmp_85303;
            }
            defunc_0_lifted_lambda_res_77368 = r_77370;
            ((double *) mem_84094)[i_82754] = defunc_0_lifted_lambda_res_77368;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82762 = 0; i_82762 < (int64_t) 16; i_82762++) {
            // futhark/microgpt.fut:248:103-113
            
            double neg_arg0_77382 = ((double *) mem_84094)[i_82762];
            
            // futhark/microgpt.fut:248:97-113
            
            double neg_res_77383 = -neg_arg0_77382;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82758 = 0; i_82758 < (int64_t) 27; i_82758++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77390 = ((double *) mem_84035)[i_82762 * (int64_t) 27 + i_82758];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77391 = ((double *) mem_84034)[i_82762 * (int64_t) 27 + i_82758];
                
                // futhark/microgpt.fut:248:75-113
                
                double zp_res_77392 = neg_res_77383 + zp_lhs_77391;
                
                // futhark/microgpt.fut:248:53-113
                
                double zt_res_77393 = zt_lhs_77390 * zp_res_77392;
                
                ((double *) mem_84106)[i_82758] = zt_res_77393;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84101, i_82762 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84106, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82770 = 0; i_82770 < (int64_t) 16; i_82770++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82766 = 0; i_82766 < (int64_t) 16; i_82766++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77408;
                double r_77410 = 0.0;
                
                for (int64_t i_77409 = 0; i_77409 < (int64_t) 27; i_77409++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77411 = ((double *) mem_param_83386.mem)[i_77409 * (int64_t) 16 + i_82766];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77412 = ((double *) mem_84101)[i_82770 * (int64_t) 27 + i_77409];
                    
                    // futhark/microgpt.fut:249:73-110
                    
                    double zt_res_77413 = zt_lhs_77411 * zt_rhs_77412;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77414 = r_77410 + zt_res_77413;
                    double r_tmp_85308 = zp_res_77414;
                    
                    r_77410 = r_tmp_85308;
                }
                defunc_0_lifted_lambda_res_77408 = r_77410;
                ((double *) mem_84122)[i_82766] = defunc_0_lifted_lambda_res_77408;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84117, i_82770 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84122, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82783 = 0; i_82783 < (int64_t) 16; i_82783++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82776 = 0; i_82776 < (int64_t) 64; i_82776++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81277;
                double r_81279 = 0.0;
                
                for (int64_t i_81278 = 0; i_81278 < (int64_t) 16; i_81278++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81280 = ((double *) mem_param_83354.mem)[i_81278 * (int64_t) 64 + i_82776];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81281 = ((double *) mem_84117)[i_82783 * (int64_t) 16 + i_81278];
                    
                    // futhark/microgpt.fut:250:73-111
                    
                    double zt_res_81282 = zt_lhs_81280 * zt_rhs_81281;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81283 = r_81279 + zt_res_81282;
                    double r_tmp_85313 = zp_res_81283;
                    
                    r_81279 = r_tmp_85313;
                }
                defunc_0_lifted_lambda_res_81277 = r_81279;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81290;
                double r_81292 = 0.0;
                
                for (int64_t i_81291 = 0; i_81291 < (int64_t) 16; i_81291++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81293 = ((double *) mem_84117)[i_81291 * (int64_t) 16 + i_82783];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81294 = ((double *) mem_83964)[i_81291 * (int64_t) 64 + i_82776];
                    
                    // futhark/microgpt.fut:300:75-111
                    
                    double zt_res_81295 = zt_lhs_81293 * zt_rhs_81294;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81296 = r_81292 + zt_res_81295;
                    double r_tmp_85314 = zp_res_81296;
                    
                    r_81292 = r_tmp_85314;
                }
                defunc_0_lifted_lambda_res_81290 = r_81292;
                ((double *) mem_84143)[i_82776] = defunc_0_lifted_lambda_res_81290;
                ((double *) mem_84144)[i_82776] = defunc_0_lifted_lambda_res_81277;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84133, i_82783 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84143, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84134, i_82783 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84144, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82792 = 0; i_82792 < (int64_t) 16; i_82792++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82788 = 0; i_82788 < (int64_t) 64; i_82788++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_77450 = ((double *) mem_83941)[i_82792 * (int64_t) 64 + i_82788];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_77451 = fmax64(0.0, indicatorp_arg0_77450);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_77452 = fsignum64(max_res_77451);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77453 = ((double *) mem_84134)[i_82792 * (int64_t) 64 + i_82788];
                
                // futhark/microgpt.fut:251:42-90
                
                double zt_res_77454 = sgn_res_77452 * zt_rhs_77453;
                
                ((double *) mem_84170)[i_82788] = zt_res_77454;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84165, i_82792 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82800 = 0; i_82800 < (int64_t) 16; i_82800++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82796 = 0; i_82796 < (int64_t) 16; i_82796++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77469;
                double r_77471 = 0.0;
                
                for (int64_t i_77470 = 0; i_77470 < (int64_t) 64; i_77470++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77472 = ((double *) mem_param_83378.mem)[i_77470 * (int64_t) 16 + i_82796];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77473 = ((double *) mem_84165)[i_82800 * (int64_t) 64 + i_77470];
                    
                    // futhark/microgpt.fut:252:73-109
                    
                    double zt_res_77474 = zt_lhs_77472 * zt_rhs_77473;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77475 = r_77471 + zt_res_77474;
                    double r_tmp_85319 = zp_res_77475;
                    
                    r_77471 = r_tmp_85319;
                }
                defunc_0_lifted_lambda_res_77469 = r_77471;
                ((double *) mem_84186)[i_82796] = defunc_0_lifted_lambda_res_77469;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84181, i_82800 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84186, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82804 = 0; i_82804 < (int64_t) 16; i_82804++) {
            // futhark/microgpt.fut:256:49-59
            
            double zs_rhs_77523 = ((double *) mem_83963)[i_82804];
            
            // futhark/microgpt.fut:256:41-59
            
            double zs_res_77524 = 1.0 / zs_rhs_77523;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77525;
            double r_77527 = 0.0;
            
            for (int64_t i_77526 = 0; i_77526 < (int64_t) 16; i_77526++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77528 = ((double *) mem_83892)[i_82804 * (int64_t) 16 + i_77526];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77529 = ((double *) mem_84181)[i_82804 * (int64_t) 16 + i_77526];
                
                // futhark/microgpt.fut:256:87-123
                
                double zt_res_77530 = zt_lhs_77528 * zt_rhs_77529;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77531 = r_77527 + zt_res_77530;
                double r_tmp_85321 = zp_res_77531;
                
                r_77527 = r_tmp_85321;
            }
            defunc_0_lifted_lambda_res_77525 = r_77527;
            // futhark/microgpt.fut:256:67-150
            
            double zt_res_77532 = zs_res_77524 * defunc_0_lifted_lambda_res_77525;
            
            // futhark/microgpt.fut:256:45-150
            
            double zt_res_77533 = zs_res_77524 * zt_res_77532;
            
            // futhark/microgpt.fut:256:33-150
            
            double neg_res_77534 = -zt_res_77533;
            
            ((double *) mem_84197)[i_82804] = neg_res_77534;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82808 = 0; i_82808 < (int64_t) 16; i_82808++) {
            // futhark/microgpt.fut:257:33-43
            
            double zt_lhs_77542 = ((double *) mem_84197)[i_82808];
            
            // futhark/microgpt.fut:257:85-95
            
            double zp_lhs_77543 = ((double *) mem_83940)[i_82808];
            
            // futhark/microgpt.fut:257:85-123
            
            double zp_res_77544 = 1.0e-5 + zp_lhs_77543;
            
            // futhark/microgpt.fut:257:77-123
            
            double sqrt_res_77545 = futrts_sqrt64(zp_res_77544);
            
            // futhark/microgpt.fut:257:63-125
            
            double zt_res_77546 = 2.0 * sqrt_res_77545;
            
            // futhark/microgpt.fut:257:49-125
            
            double zs_res_77547 = 1.0 / zt_res_77546;
            
            // futhark/microgpt.fut:257:33-125
            
            double zt_res_77548 = zt_lhs_77542 * zs_res_77547;
            
            ((double *) mem_84204)[i_82808] = zt_res_77548;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82812 = 0; i_82812 < (int64_t) 16; i_82812++) {
            // futhark/microgpt.fut:258:53-63
            
            double zs_lhs_77556 = ((double *) mem_84204)[i_82812];
            
            // futhark/microgpt.fut:258:53-78
            
            double zs_res_77557 = zs_lhs_77556 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85324 = 0; nest_i_85324 < (int64_t) 16; nest_i_85324++) {
                ((double *) mem_84211)[i_82812 * (int64_t) 16 + nest_i_85324] = zs_res_77557;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82820 = 0; i_82820 < (int64_t) 16; i_82820++) {
            // futhark/microgpt.fut:259:107-117
            
            double zs_rhs_77566 = ((double *) mem_83963)[i_82820];
            
            // futhark/microgpt.fut:259:99-117
            
            double zs_res_77567 = 1.0 / zs_rhs_77566;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82816 = 0; i_82816 < (int64_t) 16; i_82816++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77574 = ((double *) mem_84117)[i_82820 * (int64_t) 16 + i_82816];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77575 = ((double *) mem_84181)[i_82820 * (int64_t) 16 + i_82816];
                
                // futhark/microgpt.fut:259:77-117
                
                double zt_res_77576 = zs_res_77567 * zt_lhs_77575;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_77577 = ((double *) mem_83892)[i_82820 * (int64_t) 16 + i_82816];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_77578 = ((double *) mem_84211)[i_82820 * (int64_t) 16 + i_82816];
                
                // futhark/microgpt.fut:259:125-161
                
                double zt_res_77579 = zt_lhs_77577 * zt_rhs_77578;
                
                // futhark/microgpt.fut:259:94-161
                
                double zp_res_77580 = zt_res_77576 + zt_res_77579;
                
                // futhark/microgpt.fut:259:120-205
                
                double zp_res_77581 = zt_res_77579 + zp_res_77580;
                
                // futhark/microgpt.fut:259:53-205
                
                double zp_res_77582 = zp_lhs_77574 + zp_res_77581;
                
                ((double *) mem_84226)[i_82816] = zp_res_77582;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84221, i_82820 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84226, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82833 = 0; i_82833 < (int64_t) 16; i_82833++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82826 = 0; i_82826 < (int64_t) 16; i_82826++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81319;
                double r_81321 = 0.0;
                
                for (int64_t i_81320 = 0; i_81320 < (int64_t) 16; i_81320++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81322 = ((double *) mem_param_83362.mem)[i_81320 * (int64_t) 16 + i_82826];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81323 = ((double *) mem_84221)[i_82833 * (int64_t) 16 + i_81320];
                    
                    // futhark/microgpt.fut:260:73-110
                    
                    double zt_res_81324 = zt_lhs_81322 * zt_rhs_81323;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81325 = r_81321 + zt_res_81324;
                    double r_tmp_85331 = zp_res_81325;
                    
                    r_81321 = r_tmp_85331;
                }
                defunc_0_lifted_lambda_res_81319 = r_81321;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81332;
                double r_81334 = 0.0;
                
                for (int64_t i_81333 = 0; i_81333 < (int64_t) 16; i_81333++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81335 = ((double *) mem_84221)[i_81333 * (int64_t) 16 + i_82833];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81336 = ((double *) mem_83854)[i_81333 * (int64_t) 16 + i_82826];
                    
                    // futhark/microgpt.fut:298:74-110
                    
                    double zt_res_81337 = zt_lhs_81335 * zt_rhs_81336;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81338 = r_81334 + zt_res_81337;
                    double r_tmp_85332 = zp_res_81338;
                    
                    r_81334 = r_tmp_85332;
                }
                defunc_0_lifted_lambda_res_81332 = r_81334;
                ((double *) mem_84247)[i_82826] = defunc_0_lifted_lambda_res_81332;
                ((double *) mem_84248)[i_82826] = defunc_0_lifted_lambda_res_81319;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84237, i_82833 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84247, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84238, i_82833 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84248, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82855 = 0; i_82855 < (int64_t) 4; i_82855++) {
            // futhark/microgpt.fut:261:88-91
            
            int64_t zp_lhs_79796 = mul64((int64_t) 4, i_82855);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82848 = 0; i_82848 < (int64_t) 16; i_82848++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82838 = 0; i_82838 < (int64_t) 4; i_82838++) {
                    // futhark/microgpt.fut:261:93-99
                    
                    int64_t tmp_81360 = add64(zp_lhs_79796, i_82838);
                    
                    // futhark/microgpt.fut:261:70-101
                    
                    bool x_81361 = sle64((int64_t) 0, tmp_81360);
                    
                    // futhark/microgpt.fut:261:70-101
                    
                    bool y_81362 = slt64(tmp_81360, (int64_t) 16);
                    
                    // futhark/microgpt.fut:261:70-101
                    
                    bool bounds_check_81363 = x_81361 && y_81362;
                    
                    // futhark/microgpt.fut:261:70-101
                    
                    bool index_certs_81364;
                    
                    if (!bounds_check_81363) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81360, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:261:70-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:261:52-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:261:32-104\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:261:13-106\n   #10 futhark/microgpt.fut:413:5-76\n   #11 futhark/microgpt.fut:418:26-424:31\n   #12 futhark/microgpt.fut:440:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81365 = ((double *) mem_84238)[i_82848 * (int64_t) 16 + tmp_81360];
                    
                    ((double *) mem_84291)[i_82838] = lifted_lambda_res_81365;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82842 = 0; i_82842 < (int64_t) 16; i_82842++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_81379 = ((double *) mem_83737)[i_82855 * (int64_t) 256 + i_82848 * (int64_t) 16 + i_82842];
                    
                    // futhark/microgpt.fut:263:61-97
                    
                    double zs_res_81380 = zs_lhs_81379 / 2.0;
                    double zp_rhs_81381 = ((double *) masks_mem_83348.mem)[step_76675 * (int64_t) 256 + i_82848 * (int64_t) 16 + i_82842];
                    
                    // futhark/microgpt.fut:263:84-119
                    
                    double zp_res_81382 = zs_res_81380 + zp_rhs_81381;
                    
                    ((double *) mem_84298)[i_82842] = zp_res_81382;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84281, i_82848 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84298, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84282, i_82848 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84291, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84269, i_82855 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84281, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84270, i_82855 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84282, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82886 = 0; i_82886 < (int64_t) 4; i_82886++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82879 = 0; i_82879 < (int64_t) 16; i_82879++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_82345;
                double redout_82859 = -INFINITY;
                
                for (int64_t i_82861 = 0; i_82861 < (int64_t) 16; i_82861++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81508 = ((double *) mem_84269)[i_82886 * (int64_t) 256 + i_82879 * (int64_t) 16 + i_82861];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81519;
                    double r_81521 = 0.0;
                    
                    for (int64_t i_81520 = 0; i_81520 < (int64_t) 4; i_81520++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81522 = ((double *) mem_84270)[i_82886 * (int64_t) 64 + i_82879 * (int64_t) 4 + i_81520];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81523 = ((double *) mem_83656)[i_82886 * (int64_t) 64 + i_82861 * (int64_t) 4 + i_81520];
                        
                        // futhark/microgpt.fut:266:91-139
                        
                        double zt_res_81524 = zt_lhs_81522 * zt_rhs_81523;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81525 = r_81521 + zt_res_81524;
                        double r_tmp_85345 = zp_res_81525;
                        
                        r_81521 = r_tmp_85345;
                    }
                    defunc_0_lifted_lambda_res_81519 = r_81521;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_81419 = fmax64(lifted_lambda_res_81508, redout_82859);
                    
                    ((double *) mem_84345)[i_82861] = defunc_0_lifted_lambda_res_81519;
                    
                    double redout_tmp_85343 = max_res_81419;
                    
                    redout_82859 = redout_tmp_85343;
                }
                defunc_0_reduce_res_82345 = redout_82859;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_81420 = -defunc_0_reduce_res_82345;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82865 = 0; i_82865 < (int64_t) 16; i_82865++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_81427 = ((double *) mem_84269)[i_82886 * (int64_t) 256 + i_82879 * (int64_t) 16 + i_82865];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_81428 = neg_res_81420 + lifted_lambda_res_81427;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_81429 = futrts_exp64(zp_res_81428);
                    
                    ((double *) mem_84352)[i_82865] = exp_res_81429;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81431;
                double r_81433 = 0.0;
                
                for (int64_t i_81432 = 0; i_81432 < (int64_t) 16; i_81432++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_81434 = ((double *) mem_84352)[i_81432];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81435 = r_81433 + lifted_lambda_res_81434;
                    double r_tmp_85347 = zp_res_81435;
                    
                    r_81433 = r_tmp_85347;
                }
                defunc_0_lifted_lambda_res_81431 = r_81433;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82869 = 0; i_82869 < (int64_t) 16; i_82869++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_81442 = ((double *) mem_84352)[i_82869];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_81443 = zs_lhs_81442 / defunc_0_lifted_lambda_res_81431;
                    
                    ((double *) mem_84359)[i_82869] = zs_res_81443;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82873 = 0; i_82873 < (int64_t) 16; i_82873++) {
                    // futhark/microgpt.fut:265:24-34
                    
                    double lifted_lambda_res_81451 = ((double *) mem_84359)[i_82873];
                    
                    ((double *) mem_84366)[i_82873] = lifted_lambda_res_81451;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84335, i_82879 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84345, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84336, i_82879 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84366, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84323, i_82886 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84335, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84324, i_82886 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84336, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82908 = 0; i_82908 < (int64_t) 4; i_82908++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82901 = 0; i_82901 < (int64_t) 16; i_82901++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82891 = 0; i_82891 < (int64_t) 16; i_82891++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_81561 = ((double *) mem_84323)[i_82908 * (int64_t) 256 + i_82901 * (int64_t) 16 + i_82891];
                    
                    ((double *) mem_84413)[i_82891] = lifted_lambda_res_81561;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82895 = 0; i_82895 < (int64_t) 4; i_82895++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81575;
                    double r_81577 = 0.0;
                    
                    for (int64_t i_81576 = 0; i_81576 < (int64_t) 16; i_81576++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81578 = ((double *) mem_84324)[i_82908 * (int64_t) 256 + i_81576 * (int64_t) 16 + i_82901];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81579 = ((double *) mem_84270)[i_82908 * (int64_t) 64 + i_81576 * (int64_t) 4 + i_82895];
                        
                        // futhark/microgpt.fut:271:91-140
                        
                        double zt_res_81580 = zt_lhs_81578 * zt_rhs_81579;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81581 = r_81577 + zt_res_81580;
                        double r_tmp_85356 = zp_res_81581;
                        
                        r_81577 = r_tmp_85356;
                    }
                    defunc_0_lifted_lambda_res_81575 = r_81577;
                    ((double *) mem_84420)[i_82895] = defunc_0_lifted_lambda_res_81575;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84403, i_82901 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84420, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84404, i_82901 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84413, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84391, i_82908 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84403, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84392, i_82908 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84404, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82917 = 0; i_82917 < (int64_t) 4; i_82917++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82913 = 0; i_82913 < (int64_t) 16; i_82913++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77801;
                double r_77803 = 0.0;
                
                for (int64_t i_77802 = 0; i_77802 < (int64_t) 16; i_77802++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77804 = ((double *) mem_84392)[i_82917 * (int64_t) 256 + i_82913 * (int64_t) 16 + i_77802];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77805 = ((double *) mem_84324)[i_82917 * (int64_t) 256 + i_82913 * (int64_t) 16 + i_77802];
                    
                    // futhark/microgpt.fut:268:72-121
                    
                    double zt_res_77806 = zt_lhs_77804 * zt_rhs_77805;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77807 = r_77803 + zt_res_77806;
                    double r_tmp_85359 = zp_res_77807;
                    
                    r_77803 = r_tmp_85359;
                }
                defunc_0_lifted_lambda_res_77801 = r_77803;
                ((double *) mem_84450)[i_82913] = defunc_0_lifted_lambda_res_77801;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84445, i_82917 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84450, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82929 = 0; i_82929 < (int64_t) 4; i_82929++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82925 = 0; i_82925 < (int64_t) 16; i_82925++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_77822 = ((double *) mem_84445)[i_82929 * (int64_t) 16 + i_82925];
                
                // futhark/microgpt.fut:269:128-150
                
                double neg_res_77823 = -neg_arg0_77822;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82921 = 0; i_82921 < (int64_t) 16; i_82921++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_77830 = ((double *) mem_84324)[i_82929 * (int64_t) 256 + i_82925 * (int64_t) 16 + i_82921];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_77831 = ((double *) mem_84392)[i_82929 * (int64_t) 256 + i_82925 * (int64_t) 16 + i_82921];
                    
                    // futhark/microgpt.fut:269:100-150
                    
                    double zp_res_77832 = neg_res_77823 + zp_lhs_77831;
                    
                    // futhark/microgpt.fut:269:72-150
                    
                    double zt_res_77833 = zt_lhs_77830 * zp_res_77832;
                    
                    ((double *) mem_84472)[i_82921] = zt_res_77833;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84467, i_82925 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84472, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84461, i_82929 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84467, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82941 = 0; i_82941 < (int64_t) 4; i_82941++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82937 = 0; i_82937 < (int64_t) 16; i_82937++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82933 = 0; i_82933 < (int64_t) 16; i_82933++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_77855 = ((double *) mem_84461)[i_82941 * (int64_t) 256 + i_82937 * (int64_t) 16 + i_82933];
                    
                    // futhark/microgpt.fut:270:60-96
                    
                    double zs_res_77856 = zs_lhs_77855 / 2.0;
                    
                    ((double *) mem_84499)[i_82933] = zs_res_77856;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84494, i_82937 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84499, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84488, i_82941 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84494, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82961 = 0; i_82961 < (int64_t) 4; i_82961++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82954 = 0; i_82954 < (int64_t) 16; i_82954++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_82947 = 0; i_82947 < (int64_t) 4; i_82947++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81662;
                    double r_81664 = 0.0;
                    
                    for (int64_t i_81663 = 0; i_81663 < (int64_t) 16; i_81663++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81665 = ((double *) mem_83658)[i_82961 * (int64_t) 64 + i_81663 * (int64_t) 4 + i_82947];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81666 = ((double *) mem_84488)[i_82961 * (int64_t) 256 + i_81663 * (int64_t) 16 + i_82954];
                        
                        // futhark/microgpt.fut:272:91-139
                        
                        double zt_res_81667 = zt_lhs_81665 * zt_rhs_81666;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81668 = r_81664 + zt_res_81667;
                        double r_tmp_85372 = zp_res_81668;
                        
                        r_81664 = r_tmp_85372;
                    }
                    defunc_0_lifted_lambda_res_81662 = r_81664;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_81675;
                    double r_81677 = 0.0;
                    
                    for (int64_t i_81676 = 0; i_81676 < (int64_t) 16; i_81676++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_81678 = ((double *) mem_84488)[i_82961 * (int64_t) 256 + i_82954 * (int64_t) 16 + i_81676];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_81679 = ((double *) mem_83657)[i_82961 * (int64_t) 64 + i_81676 * (int64_t) 4 + i_82947];
                        
                        // futhark/microgpt.fut:273:91-139
                        
                        double zt_res_81680 = zt_lhs_81678 * zt_rhs_81679;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_81681 = r_81677 + zt_res_81680;
                        double r_tmp_85373 = zp_res_81681;
                        
                        r_81677 = r_tmp_85373;
                    }
                    defunc_0_lifted_lambda_res_81675 = r_81677;
                    ((double *) mem_84537)[i_82947] = defunc_0_lifted_lambda_res_81675;
                    ((double *) mem_84538)[i_82947] = defunc_0_lifted_lambda_res_81662;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84527, i_82954 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84537, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_84528, i_82954 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84538, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84515, i_82961 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84527, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_84516, i_82961 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_84528, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_82980 = 0; i_82980 < (int64_t) 16; i_82980++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82970 = 0; i_82970 < (int64_t) 16; i_82970++) {
                // futhark/microgpt.fut:274:63-66
                
                int64_t tmp_81744 = sdiv64(i_82970, (int64_t) 4);
                
                // futhark/microgpt.fut:274:52-68
                
                bool x_81745 = sle64((int64_t) 0, tmp_81744);
                
                // futhark/microgpt.fut:274:52-68
                
                bool y_81746 = slt64(tmp_81744, (int64_t) 4);
                
                // futhark/microgpt.fut:274:52-68
                
                bool bounds_check_81747 = x_81745 && y_81746;
                
                // futhark/microgpt.fut:274:52-68
                
                bool index_certs_81748;
                
                if (!bounds_check_81747) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81744, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:274:52-68\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:274:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:274:13-89\n   #7  futhark/microgpt.fut:413:5-76\n   #8  futhark/microgpt.fut:418:26-424:31\n   #9  futhark/microgpt.fut:440:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:274:81-84
                
                int64_t tmp_81749 = smod64(i_82970, (int64_t) 4);
                
                // futhark/microgpt.fut:274:52-86
                
                bool x_81750 = sle64((int64_t) 0, tmp_81749);
                
                // futhark/microgpt.fut:274:52-86
                
                bool y_81751 = slt64(tmp_81749, (int64_t) 4);
                
                // futhark/microgpt.fut:274:52-86
                
                bool bounds_check_81752 = x_81750 && y_81751;
                
                // futhark/microgpt.fut:274:52-86
                
                bool index_certs_81753;
                
                if (!bounds_check_81752) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_81749, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:274:52-86\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:274:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:274:13-89\n   #7  futhark/microgpt.fut:413:5-76\n   #8  futhark/microgpt.fut:418:26-424:31\n   #9  futhark/microgpt.fut:440:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81754 = ((double *) mem_84391)[tmp_81744 * (int64_t) 64 + i_82980 * (int64_t) 4 + tmp_81749];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81767 = ((double *) mem_84516)[tmp_81744 * (int64_t) 64 + i_82980 * (int64_t) 4 + tmp_81749];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_81783 = ((double *) mem_84515)[tmp_81744 * (int64_t) 64 + i_82980 * (int64_t) 4 + tmp_81749];
                
                ((double *) mem_84584)[i_82970] = lifted_lambda_res_81783;
                ((double *) mem_84585)[i_82970] = lifted_lambda_res_81767;
                ((double *) mem_84586)[i_82970] = lifted_lambda_res_81754;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84569, i_82980 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84584, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84570, i_82980 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84585, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84571, i_82980 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84586, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83005 = 0; i_83005 < (int64_t) 16; i_83005++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_82992 = 0; i_82992 < (int64_t) 16; i_82992++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81946;
                double r_81948 = 0.0;
                
                for (int64_t i_81947 = 0; i_81947 < (int64_t) 16; i_81947++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81949 = ((double *) mem_param_83382.mem)[i_81947 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81950 = ((double *) mem_84571)[i_83005 * (int64_t) 16 + i_81947];
                    
                    // futhark/microgpt.fut:277:75-112
                    
                    double zt_res_81951 = zt_lhs_81949 * zt_rhs_81950;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81952 = r_81948 + zt_res_81951;
                    double r_tmp_85388 = zp_res_81952;
                    
                    r_81948 = r_tmp_85388;
                }
                defunc_0_lifted_lambda_res_81946 = r_81948;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81953;
                double r_81955 = 0.0;
                
                for (int64_t i_81954 = 0; i_81954 < (int64_t) 16; i_81954++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81956 = ((double *) mem_param_83358.mem)[i_81954 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81957 = ((double *) mem_84570)[i_83005 * (int64_t) 16 + i_81954];
                    
                    // futhark/microgpt.fut:277:141-178
                    
                    double zt_res_81958 = zt_lhs_81956 * zt_rhs_81957;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81959 = r_81955 + zt_res_81958;
                    double r_tmp_85389 = zp_res_81959;
                    
                    r_81955 = r_tmp_85389;
                }
                defunc_0_lifted_lambda_res_81953 = r_81955;
                // futhark/microgpt.fut:277:55-180
                
                double zp_res_81960 = defunc_0_lifted_lambda_res_81946 + defunc_0_lifted_lambda_res_81953;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81961;
                double r_81963 = 0.0;
                
                for (int64_t i_81962 = 0; i_81962 < (int64_t) 16; i_81962++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81964 = ((double *) mem_param_83370.mem)[i_81962 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81965 = ((double *) mem_84569)[i_83005 * (int64_t) 16 + i_81962];
                    
                    // futhark/microgpt.fut:277:208-245
                    
                    double zt_res_81966 = zt_lhs_81964 * zt_rhs_81965;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81967 = r_81963 + zt_res_81966;
                    double r_tmp_85390 = zp_res_81967;
                    
                    r_81963 = r_tmp_85390;
                }
                defunc_0_lifted_lambda_res_81961 = r_81963;
                // futhark/microgpt.fut:277:116-247
                
                double zp_res_81968 = zp_res_81960 + defunc_0_lifted_lambda_res_81961;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81975;
                double r_81977 = 0.0;
                
                for (int64_t i_81976 = 0; i_81976 < (int64_t) 16; i_81976++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81978 = ((double *) mem_84569)[i_81976 * (int64_t) 16 + i_83005];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81979 = ((double *) mem_83557)[i_81976 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:295:74-109
                    
                    double zt_res_81980 = zt_lhs_81978 * zt_rhs_81979;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81981 = r_81977 + zt_res_81980;
                    double r_tmp_85391 = zp_res_81981;
                    
                    r_81977 = r_tmp_85391;
                }
                defunc_0_lifted_lambda_res_81975 = r_81977;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81991;
                double r_81993 = 0.0;
                
                for (int64_t i_81992 = 0; i_81992 < (int64_t) 16; i_81992++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81994 = ((double *) mem_84570)[i_81992 * (int64_t) 16 + i_83005];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81995 = ((double *) mem_83557)[i_81992 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:296:74-109
                    
                    double zt_res_81996 = zt_lhs_81994 * zt_rhs_81995;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81997 = r_81993 + zt_res_81996;
                    double r_tmp_85392 = zp_res_81997;
                    
                    r_81993 = r_tmp_85392;
                }
                defunc_0_lifted_lambda_res_81991 = r_81993;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82009;
                double r_82011 = 0.0;
                
                for (int64_t i_82010 = 0; i_82010 < (int64_t) 16; i_82010++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82012 = ((double *) mem_84571)[i_82010 * (int64_t) 16 + i_83005];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82013 = ((double *) mem_83557)[i_82010 * (int64_t) 16 + i_82992];
                    
                    // futhark/microgpt.fut:297:74-109
                    
                    double zt_res_82014 = zt_lhs_82012 * zt_rhs_82013;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82015 = r_82011 + zt_res_82014;
                    double r_tmp_85393 = zp_res_82015;
                    
                    r_82011 = r_tmp_85393;
                }
                defunc_0_lifted_lambda_res_82009 = r_82011;
                ((double *) mem_84637)[i_82992] = defunc_0_lifted_lambda_res_82009;
                ((double *) mem_84638)[i_82992] = defunc_0_lifted_lambda_res_81991;
                ((double *) mem_84639)[i_82992] = defunc_0_lifted_lambda_res_81975;
                ((double *) mem_84640)[i_82992] = zp_res_81968;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84617, i_83005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84637, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84618, i_83005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84638, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84619, i_83005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84639, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84620, i_83005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84640, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83012 = 0; i_83012 < (int64_t) 16; i_83012++) {
            // futhark/microgpt.fut:281:49-59
            
            double zs_rhs_78089 = ((double *) mem_83853)[i_83012];
            
            // futhark/microgpt.fut:281:41-59
            
            double zs_res_78090 = 1.0 / zs_rhs_78089;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78091;
            double r_78093 = 0.0;
            
            for (int64_t i_78092 = 0; i_78092 < (int64_t) 16; i_78092++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78094 = ((double *) mem_83524)[i_83012 * (int64_t) 16 + i_78092];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78095 = ((double *) mem_84620)[i_83012 * (int64_t) 16 + i_78092];
                
                // futhark/microgpt.fut:281:87-122
                
                double zt_res_78096 = zt_lhs_78094 * zt_rhs_78095;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78097 = r_78093 + zt_res_78096;
                double r_tmp_85395 = zp_res_78097;
                
                r_78093 = r_tmp_85395;
            }
            defunc_0_lifted_lambda_res_78091 = r_78093;
            // futhark/microgpt.fut:281:67-149
            
            double zt_res_78098 = zs_res_78090 * defunc_0_lifted_lambda_res_78091;
            
            // futhark/microgpt.fut:281:45-149
            
            double zt_res_78099 = zs_res_78090 * zt_res_78098;
            
            // futhark/microgpt.fut:281:33-149
            
            double neg_res_78100 = -zt_res_78099;
            
            ((double *) mem_84681)[i_83012] = neg_res_78100;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83016 = 0; i_83016 < (int64_t) 16; i_83016++) {
            // futhark/microgpt.fut:282:33-43
            
            double zt_lhs_78108 = ((double *) mem_84681)[i_83016];
            
            // futhark/microgpt.fut:282:85-95
            
            double zp_lhs_78109 = ((double *) mem_83595)[i_83016];
            
            // futhark/microgpt.fut:282:85-123
            
            double zp_res_78110 = 1.0e-5 + zp_lhs_78109;
            
            // futhark/microgpt.fut:282:77-123
            
            double sqrt_res_78111 = futrts_sqrt64(zp_res_78110);
            
            // futhark/microgpt.fut:282:63-125
            
            double zt_res_78112 = 2.0 * sqrt_res_78111;
            
            // futhark/microgpt.fut:282:49-125
            
            double zs_res_78113 = 1.0 / zt_res_78112;
            
            // futhark/microgpt.fut:282:33-125
            
            double zt_res_78114 = zt_lhs_78108 * zs_res_78113;
            
            ((double *) mem_84688)[i_83016] = zt_res_78114;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83020 = 0; i_83020 < (int64_t) 16; i_83020++) {
            // futhark/microgpt.fut:283:53-63
            
            double zs_lhs_78122 = ((double *) mem_84688)[i_83020];
            
            // futhark/microgpt.fut:283:53-78
            
            double zs_res_78123 = zs_lhs_78122 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85398 = 0; nest_i_85398 < (int64_t) 16; nest_i_85398++) {
                ((double *) mem_84695)[i_83020 * (int64_t) 16 + nest_i_85398] = zs_res_78123;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83028 = 0; i_83028 < (int64_t) 16; i_83028++) {
            // futhark/microgpt.fut:284:107-117
            
            double zs_rhs_78132 = ((double *) mem_83853)[i_83028];
            
            // futhark/microgpt.fut:284:99-117
            
            double zs_res_78133 = 1.0 / zs_rhs_78132;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83024 = 0; i_83024 < (int64_t) 16; i_83024++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_78140 = ((double *) mem_84221)[i_83028 * (int64_t) 16 + i_83024];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78141 = ((double *) mem_84620)[i_83028 * (int64_t) 16 + i_83024];
                
                // futhark/microgpt.fut:284:77-117
                
                double zt_res_78142 = zs_res_78133 * zt_lhs_78141;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78143 = ((double *) mem_83524)[i_83028 * (int64_t) 16 + i_83024];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78144 = ((double *) mem_84695)[i_83028 * (int64_t) 16 + i_83024];
                
                // futhark/microgpt.fut:284:125-160
                
                double zt_res_78145 = zt_lhs_78143 * zt_rhs_78144;
                
                // futhark/microgpt.fut:284:94-160
                
                double zp_res_78146 = zt_res_78142 + zt_res_78145;
                
                // futhark/microgpt.fut:284:120-203
                
                double zp_res_78147 = zt_res_78145 + zp_res_78146;
                
                // futhark/microgpt.fut:284:53-203
                
                double zp_res_78148 = zp_lhs_78140 + zp_res_78147;
                
                ((double *) mem_84710)[i_83024] = zp_res_78148;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84705, i_83028 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84710, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83032 = 0; i_83032 < (int64_t) 16; i_83032++) {
            // futhark/microgpt.fut:288:49-59
            
            double zs_rhs_78196 = ((double *) mem_83594)[i_83032];
            
            // futhark/microgpt.fut:288:41-59
            
            double zs_res_78197 = 1.0 / zs_rhs_78196;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_78198;
            double r_78200 = 0.0;
            
            for (int64_t i_78199 = 0; i_78199 < (int64_t) 16; i_78199++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_78201 = ((double *) mem_83492)[i_83032 * (int64_t) 16 + i_78199];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_78202 = ((double *) mem_84705)[i_83032 * (int64_t) 16 + i_78199];
                
                // futhark/microgpt.fut:288:87-122
                
                double zt_res_78203 = zt_lhs_78201 * zt_rhs_78202;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_78204 = r_78200 + zt_res_78203;
                double r_tmp_85402 = zp_res_78204;
                
                r_78200 = r_tmp_85402;
            }
            defunc_0_lifted_lambda_res_78198 = r_78200;
            // futhark/microgpt.fut:288:67-149
            
            double zt_res_78205 = zs_res_78197 * defunc_0_lifted_lambda_res_78198;
            
            // futhark/microgpt.fut:288:45-149
            
            double zt_res_78206 = zs_res_78197 * zt_res_78205;
            
            // futhark/microgpt.fut:288:33-149
            
            double neg_res_78207 = -zt_res_78206;
            
            ((double *) mem_84721)[i_83032] = neg_res_78207;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83036 = 0; i_83036 < (int64_t) 16; i_83036++) {
            // futhark/microgpt.fut:289:33-43
            
            double zt_lhs_78215 = ((double *) mem_84721)[i_83036];
            
            // futhark/microgpt.fut:289:85-95
            
            double zp_lhs_78216 = ((double *) mem_83555)[i_83036];
            
            // futhark/microgpt.fut:289:85-123
            
            double zp_res_78217 = 1.0e-5 + zp_lhs_78216;
            
            // futhark/microgpt.fut:289:77-123
            
            double sqrt_res_78218 = futrts_sqrt64(zp_res_78217);
            
            // futhark/microgpt.fut:289:63-125
            
            double zt_res_78219 = 2.0 * sqrt_res_78218;
            
            // futhark/microgpt.fut:289:49-125
            
            double zs_res_78220 = 1.0 / zt_res_78219;
            
            // futhark/microgpt.fut:289:33-125
            
            double zt_res_78221 = zt_lhs_78215 * zs_res_78220;
            
            ((double *) mem_84728)[i_83036] = zt_res_78221;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83040 = 0; i_83040 < (int64_t) 16; i_83040++) {
            // futhark/microgpt.fut:290:53-63
            
            double zs_lhs_78229 = ((double *) mem_84728)[i_83040];
            
            // futhark/microgpt.fut:290:53-78
            
            double zs_res_78230 = zs_lhs_78229 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_85405 = 0; nest_i_85405 < (int64_t) 16; nest_i_85405++) {
                ((double *) mem_84735)[i_83040 * (int64_t) 16 + nest_i_85405] = zs_res_78230;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83048 = 0; i_83048 < (int64_t) 16; i_83048++) {
            // futhark/microgpt.fut:291:85-95
            
            double zs_rhs_78239 = ((double *) mem_83594)[i_83048];
            
            // futhark/microgpt.fut:291:77-95
            
            double zs_res_78240 = 1.0 / zs_rhs_78239;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83044 = 0; i_83044 < (int64_t) 16; i_83044++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78247 = ((double *) mem_84705)[i_83048 * (int64_t) 16 + i_83044];
                
                // futhark/microgpt.fut:291:55-95
                
                double zt_res_78248 = zs_res_78240 * zt_lhs_78247;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_78249 = ((double *) mem_83492)[i_83048 * (int64_t) 16 + i_83044];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_78250 = ((double *) mem_84735)[i_83048 * (int64_t) 16 + i_83044];
                
                // futhark/microgpt.fut:291:103-138
                
                double zt_res_78251 = zt_lhs_78249 * zt_rhs_78250;
                
                // futhark/microgpt.fut:291:72-138
                
                double zp_res_78252 = zt_res_78248 + zt_res_78251;
                
                // futhark/microgpt.fut:291:98-181
                
                double zp_res_78253 = zt_res_78251 + zp_res_78252;
                
                ((double *) mem_84750)[i_83044] = zp_res_78253;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84745, i_83048 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84750, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83061 = 0; i_83061 < (int64_t) 16; i_83061++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83054 = 0; i_83054 < (int64_t) 16; i_83054++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_82041 = ((double *) mem_84745)[i_83061 * (int64_t) 16 + i_83054];
                
                ((double *) mem_84771)[i_83054] = lifted_lambda_res_82041;
                ((double *) mem_84772)[i_83054] = lifted_lambda_res_82041;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84761, i_83061 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84771, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84762, i_83061 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84772, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83070 = 0; i_83070 < (int64_t) 64; i_83070++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83066 = 0; i_83066 < (int64_t) 16; i_83066++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_78367;
                double r_78369 = 0.0;
                
                for (int64_t i_78368 = 0; i_78368 < (int64_t) 16; i_78368++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_78370 = ((double *) mem_84165)[i_78368 * (int64_t) 64 + i_83070];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_78371 = ((double *) mem_83909)[i_78368 * (int64_t) 16 + i_83066];
                    
                    // futhark/microgpt.fut:299:73-109
                    
                    double zt_res_78372 = zt_lhs_78370 * zt_rhs_78371;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_78373 = r_78369 + zt_res_78372;
                    double r_tmp_85414 = zp_res_78373;
                    
                    r_78369 = r_tmp_85414;
                }
                defunc_0_lifted_lambda_res_78367 = r_78369;
                ((double *) mem_84798)[i_83066] = defunc_0_lifted_lambda_res_78367;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84793, i_83070 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84798, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_83083 = 0; i_83083 < (int64_t) 27; i_83083++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_83076 = 0; i_83076 < (int64_t) 16; i_83076++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82069;
                double r_82071 = 0.0;
                
                for (int64_t i_82070 = 0; i_82070 < (int64_t) 16; i_82070++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82072 = ((double *) mem_84101)[i_82070 * (int64_t) 27 + i_83083];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82073 = ((double *) mem_84002)[i_82070 * (int64_t) 16 + i_83076];
                    
                    // futhark/microgpt.fut:301:74-110
                    
                    double zt_res_82074 = zt_lhs_82072 * zt_rhs_82073;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82075 = r_82071 + zt_res_82074;
                    double r_tmp_85419 = zp_res_82075;
                    
                    r_82071 = r_tmp_85419;
                }
                defunc_0_lifted_lambda_res_82069 = r_82071;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82078;
                double r_82080 = 0.0;
                
                for (int64_t i_82079 = 0; i_82079 < (int64_t) 16; i_82079++) {
                    int64_t zeze_lhs_82081 = ((int64_t *) seqs_mem_83350.mem)[step_76675 * (int64_t) 16 + i_82079];
                    
                    // futhark/microgpt.fut:414:58-109
                    
                    bool cond_82082 = zeze_lhs_82081 == i_83083;
                    
                    // futhark/microgpt.fut:414:58-109
                    
                    double lifted_lambda_res_82083;
                    
                    if (cond_82082) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_82381 = ((double *) mem_84761)[i_82079 * (int64_t) 16 + i_83076];
                        
                        lifted_lambda_res_82083 = lifted_lambda_res_t_res_82381;
                    } else {
                        lifted_lambda_res_82083 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82089 = r_82080 + lifted_lambda_res_82083;
                    double r_tmp_85420 = zp_res_82089;
                    
                    r_82080 = r_tmp_85420;
                }
                defunc_0_lifted_lambda_res_82078 = r_82080;
                ((double *) mem_84819)[i_83076] = defunc_0_lifted_lambda_res_82078;
                ((double *) mem_84820)[i_83076] = defunc_0_lifted_lambda_res_82069;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84809, i_83083 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84819, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_84810, i_83083 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_84820, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_78451 = sitofp_i64_f64(step_76675);
        
        // futhark/microgpt.fut:370:46-67
        
        double zm_rhs_78452 = i64_res_78451 / 30000.0;
        
        // futhark/microgpt.fut:370:24-67
        
        double zt_rhs_78453 = 1.0 - zm_rhs_78452;
        
        // futhark/microgpt.fut:370:19-67
        
        double lt_r_78454 = 1.0e-2 * zt_rhs_78453;
        
        // futhark/microgpt.fut:372:5-52
        if (memblock_alloc(ctx, &mem_84841, (int64_t) 3456, "mem_84841")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:372:5-52
        // futhark/microgpt.fut:372:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84841.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83374.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:372:5-52
        if (memblock_alloc(ctx, &mem_84843, (int64_t) 3456, "mem_84843")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:372:5-52
        // futhark/microgpt.fut:372:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84843.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83410.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:372:5-52
        if (memblock_alloc(ctx, &mem_84845, (int64_t) 3456, "mem_84845")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:372:5-52
        // futhark/microgpt.fut:372:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84845.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83446.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:372:5-52
        if (memblock_alloc(ctx, &mem_84847, (int64_t) 3456, "mem_84847")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:372:5-52
        // futhark/microgpt.fut:372:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84847.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84809, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:372:5-52
        if (futrts_adam_opt_w_10347(ctx, &ext_mem_84851, &ext_mem_84850, &ext_mem_84849, mem_84841, mem_84843, mem_84845, mem_84847, (int64_t) 27, (int64_t) 16, step_76675, lt_r_78454) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84841, "mem_84841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84843, "mem_84843") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84845, "mem_84845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84847, "mem_84847") != 0)
            return 1;
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84852, (int64_t) 2048, "mem_84852")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84852.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83366.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84854, (int64_t) 2048, "mem_84854")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84854.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83402.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84856, (int64_t) 2048, "mem_84856")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84856.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83438.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (memblock_alloc(ctx, &mem_84858, (int64_t) 2048, "mem_84858")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:374:5-52
        // futhark/microgpt.fut:374:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84858.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84762, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:374:5-52
        if (futrts_adam_opt_w_10348(ctx, &ext_mem_84862, &ext_mem_84861, &ext_mem_84860, mem_84852, mem_84854, mem_84856, mem_84858, (int64_t) 16, (int64_t) 16, step_76675, lt_r_78454) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_84852, "mem_84852") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84854, "mem_84854") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84856, "mem_84856") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84858, "mem_84858") != 0)
            return 1;
        // futhark/microgpt.fut:376:5-56
        if (memblock_alloc(ctx, &mem_84863, (int64_t) 2048, "mem_84863")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-56
        // futhark/microgpt.fut:376:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84863.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83370.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-56
        if (memblock_alloc(ctx, &mem_84865, (int64_t) 2048, "mem_84865")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-56
        // futhark/microgpt.fut:376:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84865.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83406.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-56
        if (memblock_alloc(ctx, &mem_84867, (int64_t) 2048, "mem_84867")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-56
        // futhark/microgpt.fut:376:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84867.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83442.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-56
        if (memblock_alloc(ctx, &mem_84869, (int64_t) 2048, "mem_84869")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:376:5-56
        // futhark/microgpt.fut:376:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84869.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84619, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:376:5-56
        if (futrts_adam_opt_w_10348(ctx, &ext_mem_84873, &ext_mem_84872, &ext_mem_84871, mem_84863, mem_84865, mem_84867, mem_84869, (int64_t) 16, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84874, (int64_t) 2048, "mem_84874")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84874.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83358.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84876, (int64_t) 2048, "mem_84876")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84876.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83394.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84878, (int64_t) 2048, "mem_84878")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84878.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83430.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (memblock_alloc(ctx, &mem_84880, (int64_t) 2048, "mem_84880")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:378:5-56
        // futhark/microgpt.fut:378:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84880.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84618, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:378:5-56
        if (futrts_adam_opt_w_10348(ctx, &ext_mem_84884, &ext_mem_84883, &ext_mem_84882, mem_84874, mem_84876, mem_84878, mem_84880, (int64_t) 16, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84885, (int64_t) 2048, "mem_84885")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84885.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83382.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84887, (int64_t) 2048, "mem_84887")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84887.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83418.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84889, (int64_t) 2048, "mem_84889")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84889.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83454.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (memblock_alloc(ctx, &mem_84891, (int64_t) 2048, "mem_84891")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:380:5-56
        // futhark/microgpt.fut:380:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84891.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84617, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:380:5-56
        if (futrts_adam_opt_w_10348(ctx, &ext_mem_84895, &ext_mem_84894, &ext_mem_84893, mem_84885, mem_84887, mem_84889, mem_84891, (int64_t) 16, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84896, (int64_t) 2048, "mem_84896")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84896.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83362.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84898, (int64_t) 2048, "mem_84898")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83398.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84900, (int64_t) 2048, "mem_84900")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84900.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83434.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (memblock_alloc(ctx, &mem_84902, (int64_t) 2048, "mem_84902")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:382:5-56
        // futhark/microgpt.fut:382:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84902.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84237, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:382:5-56
        if (futrts_adam_opt_w_10348(ctx, &ext_mem_84906, &ext_mem_84905, &ext_mem_84904, mem_84896, mem_84898, mem_84900, mem_84902, (int64_t) 16, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:384:5-52
        if (memblock_alloc(ctx, &mem_84907, (int64_t) 8192, "mem_84907")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-52
        // futhark/microgpt.fut:384:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84907.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83378.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:384:5-52
        if (memblock_alloc(ctx, &mem_84909, (int64_t) 8192, "mem_84909")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-52
        // futhark/microgpt.fut:384:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84909.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83414.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:384:5-52
        if (memblock_alloc(ctx, &mem_84911, (int64_t) 8192, "mem_84911")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-52
        // futhark/microgpt.fut:384:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84911.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83450.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:384:5-52
        if (memblock_alloc(ctx, &mem_84913, (int64_t) 8192, "mem_84913")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:384:5-52
        // futhark/microgpt.fut:384:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84913.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84793, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:384:5-52
        if (futrts_adam_opt_w_10347(ctx, &ext_mem_84917, &ext_mem_84916, &ext_mem_84915, mem_84907, mem_84909, mem_84911, mem_84913, (int64_t) 64, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:386:5-60
        if (memblock_alloc(ctx, &mem_84918, (int64_t) 8192, "mem_84918")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-60
        // futhark/microgpt.fut:386:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84918.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83354.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:386:5-60
        if (memblock_alloc(ctx, &mem_84920, (int64_t) 8192, "mem_84920")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-60
        // futhark/microgpt.fut:386:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84920.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83390.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:386:5-60
        if (memblock_alloc(ctx, &mem_84922, (int64_t) 8192, "mem_84922")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-60
        // futhark/microgpt.fut:386:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84922.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_83426.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:386:5-60
        if (memblock_alloc(ctx, &mem_84924, (int64_t) 8192, "mem_84924")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:386:5-60
        // futhark/microgpt.fut:386:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84924.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_84133, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:386:5-60
        if (futrts_adam_opt_w_10347(ctx, &ext_mem_84928, &ext_mem_84927, &ext_mem_84926, mem_84918, mem_84920, mem_84922, mem_84924, (int64_t) 16, (int64_t) 64, step_76675, lt_r_78454) != 0) {
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
        // futhark/microgpt.fut:388:5-56
        if (memblock_alloc(ctx, &mem_84929, (int64_t) 3456, "mem_84929")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-56
        // futhark/microgpt.fut:388:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84929.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83386.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:388:5-56
        if (memblock_alloc(ctx, &mem_84931, (int64_t) 3456, "mem_84931")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-56
        // futhark/microgpt.fut:388:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84931.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83422.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:388:5-56
        if (memblock_alloc(ctx, &mem_84933, (int64_t) 3456, "mem_84933")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-56
        // futhark/microgpt.fut:388:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84933.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_83458.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:388:5-56
        if (memblock_alloc(ctx, &mem_84935, (int64_t) 3456, "mem_84935")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:388:5-56
        // futhark/microgpt.fut:388:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_84935.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_84810, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:388:5-56
        if (futrts_adam_opt_w_10347(ctx, &ext_mem_84939, &ext_mem_84938, &ext_mem_84937, mem_84929, mem_84931, mem_84933, mem_84935, (int64_t) 27, (int64_t) 16, step_76675, lt_r_78454) != 0) {
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
        if (memblock_set(ctx, &mem_param_tmp_85147, &ext_mem_84928, "ext_mem_84928") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85148, &ext_mem_84884, "ext_mem_84884") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85149, &ext_mem_84906, "ext_mem_84906") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85150, &ext_mem_84862, "ext_mem_84862") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85151, &ext_mem_84873, "ext_mem_84873") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85152, &ext_mem_84851, "ext_mem_84851") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85153, &ext_mem_84917, "ext_mem_84917") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85154, &ext_mem_84895, "ext_mem_84895") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85155, &ext_mem_84939, "ext_mem_84939") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85156, &ext_mem_84927, "ext_mem_84927") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85157, &ext_mem_84883, "ext_mem_84883") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85158, &ext_mem_84905, "ext_mem_84905") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85159, &ext_mem_84861, "ext_mem_84861") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85160, &ext_mem_84872, "ext_mem_84872") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85161, &ext_mem_84850, "ext_mem_84850") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85162, &ext_mem_84916, "ext_mem_84916") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85163, &ext_mem_84894, "ext_mem_84894") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85164, &ext_mem_84938, "ext_mem_84938") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85165, &ext_mem_84926, "ext_mem_84926") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85166, &ext_mem_84882, "ext_mem_84882") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85167, &ext_mem_84904, "ext_mem_84904") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85168, &ext_mem_84860, "ext_mem_84860") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85169, &ext_mem_84871, "ext_mem_84871") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85170, &ext_mem_84849, "ext_mem_84849") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85171, &ext_mem_84915, "ext_mem_84915") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85172, &ext_mem_84893, "ext_mem_84893") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_85173, &ext_mem_84937, "ext_mem_84937") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83354, &mem_param_tmp_85147, "mem_param_tmp_85147") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83358, &mem_param_tmp_85148, "mem_param_tmp_85148") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83362, &mem_param_tmp_85149, "mem_param_tmp_85149") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83366, &mem_param_tmp_85150, "mem_param_tmp_85150") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83370, &mem_param_tmp_85151, "mem_param_tmp_85151") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83374, &mem_param_tmp_85152, "mem_param_tmp_85152") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83378, &mem_param_tmp_85153, "mem_param_tmp_85153") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83382, &mem_param_tmp_85154, "mem_param_tmp_85154") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83386, &mem_param_tmp_85155, "mem_param_tmp_85155") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83390, &mem_param_tmp_85156, "mem_param_tmp_85156") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83394, &mem_param_tmp_85157, "mem_param_tmp_85157") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83398, &mem_param_tmp_85158, "mem_param_tmp_85158") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83402, &mem_param_tmp_85159, "mem_param_tmp_85159") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83406, &mem_param_tmp_85160, "mem_param_tmp_85160") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83410, &mem_param_tmp_85161, "mem_param_tmp_85161") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83414, &mem_param_tmp_85162, "mem_param_tmp_85162") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83418, &mem_param_tmp_85163, "mem_param_tmp_85163") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83422, &mem_param_tmp_85164, "mem_param_tmp_85164") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83426, &mem_param_tmp_85165, "mem_param_tmp_85165") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83430, &mem_param_tmp_85166, "mem_param_tmp_85166") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83434, &mem_param_tmp_85167, "mem_param_tmp_85167") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83438, &mem_param_tmp_85168, "mem_param_tmp_85168") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83442, &mem_param_tmp_85169, "mem_param_tmp_85169") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83446, &mem_param_tmp_85170, "mem_param_tmp_85170") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83450, &mem_param_tmp_85171, "mem_param_tmp_85171") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83454, &mem_param_tmp_85172, "mem_param_tmp_85172") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_83458, &mem_param_tmp_85173, "mem_param_tmp_85173") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_85047, &mem_param_83354, "mem_param_83354") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85046, &mem_param_83358, "mem_param_83358") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85045, &mem_param_83362, "mem_param_83362") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85044, &mem_param_83366, "mem_param_83366") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85043, &mem_param_83370, "mem_param_83370") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85042, &mem_param_83374, "mem_param_83374") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85041, &mem_param_83378, "mem_param_83378") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85040, &mem_param_83382, "mem_param_83382") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85039, &mem_param_83386, "mem_param_83386") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85038, &mem_param_83390, "mem_param_83390") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85037, &mem_param_83394, "mem_param_83394") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85036, &mem_param_83398, "mem_param_83398") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85035, &mem_param_83402, "mem_param_83402") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85034, &mem_param_83406, "mem_param_83406") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85033, &mem_param_83410, "mem_param_83410") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85032, &mem_param_83414, "mem_param_83414") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85031, &mem_param_83418, "mem_param_83418") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85030, &mem_param_83422, "mem_param_83422") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85029, &mem_param_83426, "mem_param_83426") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85028, &mem_param_83430, "mem_param_83430") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85027, &mem_param_83434, "mem_param_83434") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85026, &mem_param_83438, "mem_param_83438") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85025, &mem_param_83442, "mem_param_83442") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85024, &mem_param_83446, "mem_param_83446") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85023, &mem_param_83450, "mem_param_83450") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85022, &mem_param_83454, "mem_param_83454") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_85021, &mem_param_83458, "mem_param_83458") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85120, &ext_mem_85042, "ext_mem_85042") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85121, &ext_mem_85044, "ext_mem_85044") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85122, &ext_mem_85043, "ext_mem_85043") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85123, &ext_mem_85046, "ext_mem_85046") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85124, &ext_mem_85040, "ext_mem_85040") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85125, &ext_mem_85045, "ext_mem_85045") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85126, &ext_mem_85041, "ext_mem_85041") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &ext_mem_85047, "ext_mem_85047") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &ext_mem_85039, "ext_mem_85039") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85129, &ext_mem_85033, "ext_mem_85033") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85130, &ext_mem_85035, "ext_mem_85035") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85131, &ext_mem_85034, "ext_mem_85034") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85132, &ext_mem_85037, "ext_mem_85037") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85133, &ext_mem_85031, "ext_mem_85031") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85134, &ext_mem_85036, "ext_mem_85036") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85135, &ext_mem_85032, "ext_mem_85032") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85136, &ext_mem_85038, "ext_mem_85038") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85137, &ext_mem_85030, "ext_mem_85030") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85138, &ext_mem_85024, "ext_mem_85024") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85139, &ext_mem_85026, "ext_mem_85026") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85140, &ext_mem_85025, "ext_mem_85025") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85141, &ext_mem_85028, "ext_mem_85028") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85142, &ext_mem_85022, "ext_mem_85022") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85143, &ext_mem_85027, "ext_mem_85027") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85144, &ext_mem_85023, "ext_mem_85023") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85145, &ext_mem_85029, "ext_mem_85029") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85146, &ext_mem_85021, "ext_mem_85021") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85513, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85514, &mem_out_85121, "mem_out_85121") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85515, &mem_out_85122, "mem_out_85122") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85516, &mem_out_85123, "mem_out_85123") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85517, &mem_out_85124, "mem_out_85124") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85518, &mem_out_85125, "mem_out_85125") != 0)
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
    
  cleanup:
    {
        free(mem_83459);
        free(mem_83460);
        free(mem_83469);
        free(mem_83476);
        free(mem_83491);
        free(mem_83492);
        free(mem_83501);
        free(mem_83508);
        free(mem_83523);
        free(mem_83524);
        free(mem_83533);
        free(mem_83534);
        free(mem_83555);
        free(mem_83556);
        free(mem_83557);
        free(mem_83569);
        free(mem_83570);
        free(mem_83594);
        free(mem_83595);
        free(mem_83596);
        free(mem_83597);
        free(mem_83598);
        free(mem_83617);
        free(mem_83618);
        free(mem_83619);
        free(mem_83656);
        free(mem_83657);
        free(mem_83658);
        free(mem_83674);
        free(mem_83675);
        free(mem_83676);
        free(mem_83689);
        free(mem_83690);
        free(mem_83691);
        free(mem_83737);
        free(mem_83738);
        free(mem_83749);
        free(mem_83750);
        free(mem_83759);
        free(mem_83760);
        free(mem_83781);
        free(mem_83786);
        free(mem_83797);
        free(mem_83802);
        free(mem_83809);
        free(mem_83816);
        free(mem_83827);
        free(mem_83832);
        free(mem_83853);
        free(mem_83854);
        free(mem_83862);
        free(mem_83876);
        free(mem_83881);
        free(mem_83892);
        free(mem_83897);
        free(mem_83908);
        free(mem_83909);
        free(mem_83918);
        free(mem_83919);
        free(mem_83940);
        free(mem_83941);
        free(mem_83949);
        free(mem_83963);
        free(mem_83964);
        free(mem_83972);
        free(mem_83986);
        free(mem_83991);
        free(mem_84002);
        free(mem_84007);
        free(mem_84018);
        free(mem_84023);
        free(mem_84034);
        free(mem_84035);
        free(mem_84044);
        free(mem_84045);
        free(mem_84058);
        free(mem_84059);
        free(mem_84072);
        free(mem_84073);
        free(mem_84094);
        free(mem_84101);
        free(mem_84106);
        free(mem_84117);
        free(mem_84122);
        free(mem_84133);
        free(mem_84134);
        free(mem_84143);
        free(mem_84144);
        free(mem_84165);
        free(mem_84170);
        free(mem_84181);
        free(mem_84186);
        free(mem_84197);
        free(mem_84204);
        free(mem_84211);
        free(mem_84221);
        free(mem_84226);
        free(mem_84237);
        free(mem_84238);
        free(mem_84247);
        free(mem_84248);
        free(mem_84269);
        free(mem_84270);
        free(mem_84281);
        free(mem_84282);
        free(mem_84291);
        free(mem_84298);
        free(mem_84323);
        free(mem_84324);
        free(mem_84335);
        free(mem_84336);
        free(mem_84345);
        free(mem_84352);
        free(mem_84359);
        free(mem_84366);
        free(mem_84391);
        free(mem_84392);
        free(mem_84403);
        free(mem_84404);
        free(mem_84413);
        free(mem_84420);
        free(mem_84445);
        free(mem_84450);
        free(mem_84461);
        free(mem_84467);
        free(mem_84472);
        free(mem_84488);
        free(mem_84494);
        free(mem_84499);
        free(mem_84515);
        free(mem_84516);
        free(mem_84527);
        free(mem_84528);
        free(mem_84537);
        free(mem_84538);
        free(mem_84569);
        free(mem_84570);
        free(mem_84571);
        free(mem_84584);
        free(mem_84585);
        free(mem_84586);
        free(mem_84617);
        free(mem_84618);
        free(mem_84619);
        free(mem_84620);
        free(mem_84637);
        free(mem_84638);
        free(mem_84639);
        free(mem_84640);
        free(mem_84681);
        free(mem_84688);
        free(mem_84695);
        free(mem_84705);
        free(mem_84710);
        free(mem_84721);
        free(mem_84728);
        free(mem_84735);
        free(mem_84745);
        free(mem_84750);
        free(mem_84761);
        free(mem_84762);
        free(mem_84771);
        free(mem_84772);
        free(mem_84793);
        free(mem_84798);
        free(mem_84809);
        free(mem_84810);
        free(mem_84819);
        free(mem_84820);
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
        if (memblock_unref(ctx, &mem_param_tmp_85152, "mem_param_tmp_85152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85151, "mem_param_tmp_85151") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85150, "mem_param_tmp_85150") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85149, "mem_param_tmp_85149") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85148, "mem_param_tmp_85148") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_85147, "mem_param_tmp_85147") != 0)
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
        if (memblock_unref(ctx, &ext_mem_84860, "ext_mem_84860") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84861, "ext_mem_84861") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84862, "ext_mem_84862") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84858, "mem_84858") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84856, "mem_84856") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84854, "mem_84854") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84852, "mem_84852") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84849, "ext_mem_84849") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84850, "ext_mem_84850") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_84851, "ext_mem_84851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84847, "mem_84847") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84845, "mem_84845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84843, "mem_84843") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_84841, "mem_84841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83458, "mem_param_83458") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83454, "mem_param_83454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83450, "mem_param_83450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83446, "mem_param_83446") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83442, "mem_param_83442") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83438, "mem_param_83438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83434, "mem_param_83434") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83430, "mem_param_83430") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83426, "mem_param_83426") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83422, "mem_param_83422") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83418, "mem_param_83418") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83414, "mem_param_83414") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83410, "mem_param_83410") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83406, "mem_param_83406") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83402, "mem_param_83402") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83398, "mem_param_83398") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83394, "mem_param_83394") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83390, "mem_param_83390") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83386, "mem_param_83386") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83382, "mem_param_83382") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83378, "mem_param_83378") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83374, "mem_param_83374") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83370, "mem_param_83370") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83366, "mem_param_83366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83362, "mem_param_83362") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83358, "mem_param_83358") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_83354, "mem_param_83354") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85021, "ext_mem_85021") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85022, "ext_mem_85022") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85023, "ext_mem_85023") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85024, "ext_mem_85024") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85025, "ext_mem_85025") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_85026, "ext_mem_85026") != 0)
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
        if (memblock_unref(ctx, &mem_out_85125, "mem_out_85125") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85124, "mem_out_85124") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85123, "mem_out_85123") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85122, "mem_out_85122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85121, "mem_out_85121") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_85709, struct memblock *mem_out_p_85710, struct memblock *mem_out_p_85711, struct memblock *mem_out_p_85712, struct memblock *mem_out_p_85713, struct memblock *mem_out_p_85714, struct memblock *mem_out_p_85715, struct memblock *mem_out_p_85716, struct memblock *mem_out_p_85717)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mem_83312 = ctx->constants->mem_83312;
    struct memblock mem_83313 = ctx->constants->mem_83313;
    struct memblock mem_83314 = ctx->constants->mem_83314;
    struct memblock mem_83315 = ctx->constants->mem_83315;
    struct memblock mem_83316 = ctx->constants->mem_83316;
    struct memblock mem_83317 = ctx->constants->mem_83317;
    struct memblock mem_83318 = ctx->constants->mem_83318;
    struct memblock mem_83319 = ctx->constants->mem_83319;
    struct memblock mem_83320 = ctx->constants->mem_83320;
    
    if (memblock_set(ctx, &mem_out_85120, &mem_83319, "mem_83319") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85121, &mem_83315, "mem_83315") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85122, &mem_83317, "mem_83317") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85123, &mem_83313, "mem_83313") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85124, &mem_83314, "mem_83314") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85125, &mem_83312, "mem_83312") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85126, &mem_83318, "mem_83318") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85127, &mem_83316, "mem_83316") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_85128, &mem_83320, "mem_83320") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85709, &mem_out_85120, "mem_out_85120") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85710, &mem_out_85121, "mem_out_85121") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85711, &mem_out_85122, "mem_out_85122") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85712, &mem_out_85123, "mem_out_85123") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85713, &mem_out_85124, "mem_out_85124") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85714, &mem_out_85125, "mem_out_85125") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85715, &mem_out_85126, "mem_out_85126") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85716, &mem_out_85127, "mem_out_85127") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_85717, &mem_out_85128, "mem_out_85128") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_85128, "mem_out_85128") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85127, "mem_out_85127") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85126, "mem_out_85126") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85125, "mem_out_85125") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85124, "mem_out_85124") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85123, "mem_out_85123") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85122, "mem_out_85122") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85121, "mem_out_85121") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_85120, "mem_out_85120") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock mask_mem_83331;
    
    mask_mem_83331.references = NULL;
    
    struct memblock tokens_mem_83330;
    
    tokens_mem_83330.references = NULL;
    
    struct memblock wvoc_mem_83329;
    
    wvoc_mem_83329.references = NULL;
    
    struct memblock wval_mem_83328;
    
    wval_mem_83328.references = NULL;
    
    struct memblock wup_mem_83327;
    
    wup_mem_83327.references = NULL;
    
    struct memblock wte_mem_83326;
    
    wte_mem_83326.references = NULL;
    
    struct memblock wqry_mem_83325;
    
    wqry_mem_83325.references = NULL;
    
    struct memblock wpe_mem_83324;
    
    wpe_mem_83324.references = NULL;
    
    struct memblock wout_mem_83323;
    
    wout_mem_83323.references = NULL;
    
    struct memblock wkey_mem_83322;
    
    wkey_mem_83322.references = NULL;
    
    struct memblock wdown_mem_83321;
    
    wdown_mem_83321.references = NULL;
    wdown_mem_83321 = in0->v0->mem;
    wkey_mem_83322 = in0->v1->mem;
    wout_mem_83323 = in0->v2->mem;
    wpe_mem_83324 = in0->v3->mem;
    wqry_mem_83325 = in0->v4->mem;
    wte_mem_83326 = in0->v5->mem;
    wup_mem_83327 = in0->v6->mem;
    wval_mem_83328 = in0->v7->mem;
    wvoc_mem_83329 = in0->v8->mem;
    tokens_mem_83330 = in1->mem;
    mask_mem_83331 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_85120, wdown_mem_83321, wkey_mem_83322, wout_mem_83323, wpe_mem_83324, wqry_mem_83325, wte_mem_83326, wup_mem_83327, wval_mem_83328, wvoc_mem_83329, tokens_mem_83330, mask_mem_83331);
        if (ret == 0) {
            struct memblock mem_83312 = ctx->constants->mem_83312;
            struct memblock mem_83313 = ctx->constants->mem_83313;
            struct memblock mem_83314 = ctx->constants->mem_83314;
            struct memblock mem_83315 = ctx->constants->mem_83315;
            struct memblock mem_83316 = ctx->constants->mem_83316;
            struct memblock mem_83317 = ctx->constants->mem_83317;
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_85120;
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
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock wvoc_mem_83329;
    
    wvoc_mem_83329.references = NULL;
    
    struct memblock wdown_mem_83328;
    
    wdown_mem_83328.references = NULL;
    
    struct memblock wup_mem_83327;
    
    wup_mem_83327.references = NULL;
    
    struct memblock wout_mem_83326;
    
    wout_mem_83326.references = NULL;
    
    struct memblock wval_mem_83325;
    
    wval_mem_83325.references = NULL;
    
    struct memblock wkey_mem_83324;
    
    wkey_mem_83324.references = NULL;
    
    struct memblock wqry_mem_83323;
    
    wqry_mem_83323.references = NULL;
    
    struct memblock wpe_mem_83322;
    
    wpe_mem_83322.references = NULL;
    
    struct memblock wte_mem_83321;
    
    wte_mem_83321.references = NULL;
    wte_mem_83321 = in0->mem;
    wpe_mem_83322 = in1->mem;
    wqry_mem_83323 = in2->mem;
    wkey_mem_83324 = in3->mem;
    wval_mem_83325 = in4->mem;
    wout_mem_83326 = in5->mem;
    wup_mem_83327 = in6->mem;
    wdown_mem_83328 = in7->mem;
    wvoc_mem_83329 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_85120, &mem_out_85121, &mem_out_85122, &mem_out_85123, &mem_out_85124, &mem_out_85125, &mem_out_85126, &mem_out_85127, &mem_out_85128, wte_mem_83321, wpe_mem_83322, wqry_mem_83323, wkey_mem_83324, wval_mem_83325, wout_mem_83326, wup_mem_83327, wdown_mem_83328, wvoc_mem_83329);
        if (ret == 0) {
            struct memblock mem_83312 = ctx->constants->mem_83312;
            struct memblock mem_83313 = ctx->constants->mem_83313;
            struct memblock mem_83314 = ctx->constants->mem_83314;
            struct memblock mem_83315 = ctx->constants->mem_83315;
            struct memblock mem_83316 = ctx->constants->mem_83316;
            struct memblock mem_83317 = ctx->constants->mem_83317;
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85120;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85121;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85122;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85123;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85124;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85125;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85126;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85127;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85128;
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
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    
    struct memblock seqs_mem_83350;
    
    seqs_mem_83350.references = NULL;
    
    struct memblock dls_mem_83349;
    
    dls_mem_83349.references = NULL;
    
    struct memblock masks_mem_83348;
    
    masks_mem_83348.references = NULL;
    
    struct memblock wvoc_mem_83347;
    
    wvoc_mem_83347.references = NULL;
    
    struct memblock wval_mem_83346;
    
    wval_mem_83346.references = NULL;
    
    struct memblock wup_mem_83345;
    
    wup_mem_83345.references = NULL;
    
    struct memblock wte_mem_83344;
    
    wte_mem_83344.references = NULL;
    
    struct memblock wqry_mem_83343;
    
    wqry_mem_83343.references = NULL;
    
    struct memblock wpe_mem_83342;
    
    wpe_mem_83342.references = NULL;
    
    struct memblock wout_mem_83341;
    
    wout_mem_83341.references = NULL;
    
    struct memblock wkey_mem_83340;
    
    wkey_mem_83340.references = NULL;
    
    struct memblock wdown_mem_83339;
    
    wdown_mem_83339.references = NULL;
    
    struct memblock wvoc_mem_83338;
    
    wvoc_mem_83338.references = NULL;
    
    struct memblock wval_mem_83337;
    
    wval_mem_83337.references = NULL;
    
    struct memblock wup_mem_83336;
    
    wup_mem_83336.references = NULL;
    
    struct memblock wte_mem_83335;
    
    wte_mem_83335.references = NULL;
    
    struct memblock wqry_mem_83334;
    
    wqry_mem_83334.references = NULL;
    
    struct memblock wpe_mem_83333;
    
    wpe_mem_83333.references = NULL;
    
    struct memblock wout_mem_83332;
    
    wout_mem_83332.references = NULL;
    
    struct memblock wkey_mem_83331;
    
    wkey_mem_83331.references = NULL;
    
    struct memblock wdown_mem_83330;
    
    wdown_mem_83330.references = NULL;
    
    struct memblock wvoc_mem_83329;
    
    wvoc_mem_83329.references = NULL;
    
    struct memblock wval_mem_83328;
    
    wval_mem_83328.references = NULL;
    
    struct memblock wup_mem_83327;
    
    wup_mem_83327.references = NULL;
    
    struct memblock wte_mem_83326;
    
    wte_mem_83326.references = NULL;
    
    struct memblock wqry_mem_83325;
    
    wqry_mem_83325.references = NULL;
    
    struct memblock wpe_mem_83324;
    
    wpe_mem_83324.references = NULL;
    
    struct memblock wout_mem_83323;
    
    wout_mem_83323.references = NULL;
    
    struct memblock wkey_mem_83322;
    
    wkey_mem_83322.references = NULL;
    
    struct memblock wdown_mem_83321;
    
    wdown_mem_83321.references = NULL;
    wdown_mem_83321 = in0->v0->mem;
    wkey_mem_83322 = in0->v1->mem;
    wout_mem_83323 = in0->v2->mem;
    wpe_mem_83324 = in0->v3->mem;
    wqry_mem_83325 = in0->v4->mem;
    wte_mem_83326 = in0->v5->mem;
    wup_mem_83327 = in0->v6->mem;
    wval_mem_83328 = in0->v7->mem;
    wvoc_mem_83329 = in0->v8->mem;
    wdown_mem_83330 = in1->v0->mem;
    wkey_mem_83331 = in1->v1->mem;
    wout_mem_83332 = in1->v2->mem;
    wpe_mem_83333 = in1->v3->mem;
    wqry_mem_83334 = in1->v4->mem;
    wte_mem_83335 = in1->v5->mem;
    wup_mem_83336 = in1->v6->mem;
    wval_mem_83337 = in1->v7->mem;
    wvoc_mem_83338 = in1->v8->mem;
    wdown_mem_83339 = in2->v0->mem;
    wkey_mem_83340 = in2->v1->mem;
    wout_mem_83341 = in2->v2->mem;
    wpe_mem_83342 = in2->v3->mem;
    wqry_mem_83343 = in2->v4->mem;
    wte_mem_83344 = in2->v5->mem;
    wup_mem_83345 = in2->v6->mem;
    wval_mem_83346 = in2->v7->mem;
    wvoc_mem_83347 = in2->v8->mem;
    masks_mem_83348 = in3->mem;
    dls_mem_83349 = in4->mem;
    seqs_mem_83350 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 30000 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 30000 == in4->shape[0] && ((int64_t) 30000 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_85120, &mem_out_85121, &mem_out_85122, &mem_out_85123, &mem_out_85124, &mem_out_85125, &mem_out_85126, &mem_out_85127, &mem_out_85128, &mem_out_85129, &mem_out_85130, &mem_out_85131, &mem_out_85132, &mem_out_85133, &mem_out_85134, &mem_out_85135, &mem_out_85136, &mem_out_85137, &mem_out_85138, &mem_out_85139, &mem_out_85140, &mem_out_85141, &mem_out_85142, &mem_out_85143, &mem_out_85144, &mem_out_85145, &mem_out_85146, wdown_mem_83321, wkey_mem_83322, wout_mem_83323, wpe_mem_83324, wqry_mem_83325, wte_mem_83326, wup_mem_83327, wval_mem_83328, wvoc_mem_83329, wdown_mem_83330, wkey_mem_83331, wout_mem_83332, wpe_mem_83333, wqry_mem_83334, wte_mem_83335, wup_mem_83336, wval_mem_83337, wvoc_mem_83338, wdown_mem_83339, wkey_mem_83340, wout_mem_83341, wpe_mem_83342, wqry_mem_83343, wte_mem_83344, wup_mem_83345, wval_mem_83346, wvoc_mem_83347, masks_mem_83348, dls_mem_83349, seqs_mem_83350);
        if (ret == 0) {
            struct memblock mem_83312 = ctx->constants->mem_83312;
            struct memblock mem_83313 = ctx->constants->mem_83313;
            struct memblock mem_83314 = ctx->constants->mem_83314;
            struct memblock mem_83315 = ctx->constants->mem_83315;
            struct memblock mem_83316 = ctx->constants->mem_83316;
            struct memblock mem_83317 = ctx->constants->mem_83317;
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85120;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85121;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85122;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85123;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85124;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85125;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85126;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85127;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85128;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_85129;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_85130;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_85131;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_85132;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_85133;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_85134;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_85135;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_85136;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_85137;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_85138;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_85139;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_85140;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_85141;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_85142;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_85143;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_85144;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_85145;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_85146;
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
    
    struct memblock mem_out_85128;
    
    mem_out_85128.references = NULL;
    
    struct memblock mem_out_85127;
    
    mem_out_85127.references = NULL;
    
    struct memblock mem_out_85126;
    
    mem_out_85126.references = NULL;
    
    struct memblock mem_out_85125;
    
    mem_out_85125.references = NULL;
    
    struct memblock mem_out_85124;
    
    mem_out_85124.references = NULL;
    
    struct memblock mem_out_85123;
    
    mem_out_85123.references = NULL;
    
    struct memblock mem_out_85122;
    
    mem_out_85122.references = NULL;
    
    struct memblock mem_out_85121;
    
    mem_out_85121.references = NULL;
    
    struct memblock mem_out_85120;
    
    mem_out_85120.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_85120, &mem_out_85121, &mem_out_85122, &mem_out_85123, &mem_out_85124, &mem_out_85125, &mem_out_85126, &mem_out_85127, &mem_out_85128);
        if (ret == 0) {
            struct memblock mem_83312 = ctx->constants->mem_83312;
            struct memblock mem_83313 = ctx->constants->mem_83313;
            struct memblock mem_83314 = ctx->constants->mem_83314;
            struct memblock mem_83315 = ctx->constants->mem_83315;
            struct memblock mem_83316 = ctx->constants->mem_83316;
            struct memblock mem_83317 = ctx->constants->mem_83317;
            struct memblock mem_83318 = ctx->constants->mem_83318;
            struct memblock mem_83319 = ctx->constants->mem_83319;
            struct memblock mem_83320 = ctx->constants->mem_83320;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_85120;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_85121;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_85122;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_85123;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_85124;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_85125;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_85126;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_85127;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_85128;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
