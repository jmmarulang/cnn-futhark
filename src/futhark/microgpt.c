
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
struct entry_point entry_points[] = {{.name ="to_params", .f =call_to_params, .tuning_params =to_params_tuning_params, .in_types =to_params_in_types, .out_type =&type_params, .in_unique =to_params_in_unique, .out_unique =false, .attrs =to_params_attrs}, {.name ="train", .f =call_train, .tuning_params =train_tuning_params, .in_types =train_in_types, .out_type =&type_ZLZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRz2cUz20UZLZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64z2cUz20UZMZNZMZNf64ZRZR, .in_unique =train_in_unique, .out_unique =false, .attrs =train_attrs}, {.name ="zero_params", .f =call_zzero_params, .tuning_params =zzero_params_tuning_params, .in_types =zzero_params_in_types, .out_type =&type_params, .in_unique =zzero_params_in_unique, .out_unique =false, .attrs =zzero_params_attrs}, {.name =NULL}};
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
    struct memblock mem_70833;
    struct memblock mem_70834;
    struct memblock mem_70835;
    struct memblock mem_70836;
    struct memblock mem_70837;
    struct memblock mem_70838;
    struct memblock mem_70839;
    struct memblock mem_70840;
    struct memblock mem_70841;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_9496(struct futhark_context *ctx, struct memblock *mem_out_p_72960, struct memblock *mem_out_p_72961, struct memblock *mem_out_p_72962, struct memblock w_mem_70842, struct memblock mw_mem_70843, struct memblock vw_mem_70844, struct memblock dw_mem_70845, int64_t n_47868, int64_t m_47869, int64_t step_47874, double lt_r_47875);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_9497(struct futhark_context *ctx, struct memblock *mem_out_p_72965, struct memblock *mem_out_p_72966, struct memblock *mem_out_p_72967, struct memblock w_mem_70842, struct memblock mw_mem_70843, struct memblock vw_mem_70844, struct memblock dw_mem_70845, int64_t n_48901, int64_t m_48902, int64_t step_48907, double lt_r_48908);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_72970, struct memblock *mem_out_p_72971, struct memblock *mem_out_p_72972, struct memblock *mem_out_p_72973, struct memblock *mem_out_p_72974, struct memblock *mem_out_p_72975, struct memblock *mem_out_p_72976, struct memblock *mem_out_p_72977, struct memblock *mem_out_p_72978, struct memblock wte_mem_70842, struct memblock wpe_mem_70843, struct memblock wqry_mem_70844, struct memblock wkey_mem_70845, struct memblock wval_mem_70846, struct memblock wout_mem_70847, struct memblock wup_mem_70848, struct memblock wdown_mem_70849, struct memblock wvoc_mem_70850);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_72979, struct memblock *mem_out_p_72980, struct memblock *mem_out_p_72981, struct memblock *mem_out_p_72982, struct memblock *mem_out_p_72983, struct memblock *mem_out_p_72984, struct memblock *mem_out_p_72985, struct memblock *mem_out_p_72986, struct memblock *mem_out_p_72987, struct memblock *mem_out_p_72988, struct memblock *mem_out_p_72989, struct memblock *mem_out_p_72990, struct memblock *mem_out_p_72991, struct memblock *mem_out_p_72992, struct memblock *mem_out_p_72993, struct memblock *mem_out_p_72994, struct memblock *mem_out_p_72995, struct memblock *mem_out_p_72996, struct memblock *mem_out_p_72997, struct memblock *mem_out_p_72998, struct memblock *mem_out_p_72999, struct memblock *mem_out_p_73000, struct memblock *mem_out_p_73001, struct memblock *mem_out_p_73002, struct memblock *mem_out_p_73003, struct memblock *mem_out_p_73004, struct memblock *mem_out_p_73005, struct memblock wdown_mem_70842, struct memblock wkey_mem_70843, struct memblock wout_mem_70844, struct memblock wpe_mem_70845, struct memblock wqry_mem_70846, struct memblock wte_mem_70847, struct memblock wup_mem_70848, struct memblock wval_mem_70849, struct memblock wvoc_mem_70850, struct memblock wdown_mem_70851, struct memblock wkey_mem_70852, struct memblock wout_mem_70853, struct memblock wpe_mem_70854, struct memblock wqry_mem_70855, struct memblock wte_mem_70856, struct memblock wup_mem_70857, struct memblock wval_mem_70858, struct memblock wvoc_mem_70859, struct memblock wdown_mem_70860, struct memblock wkey_mem_70861, struct memblock wout_mem_70862, struct memblock wpe_mem_70863, struct memblock wqry_mem_70864, struct memblock wte_mem_70865, struct memblock wup_mem_70866, struct memblock wval_mem_70867, struct memblock wvoc_mem_70868, struct memblock masks_mem_70869, struct memblock dls_mem_70870, struct memblock seqs_mem_70871);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_73175, struct memblock *mem_out_p_73176, struct memblock *mem_out_p_73177, struct memblock *mem_out_p_73178, struct memblock *mem_out_p_73179, struct memblock *mem_out_p_73180, struct memblock *mem_out_p_73181, struct memblock *mem_out_p_73182, struct memblock *mem_out_p_73183);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_70833 (ctx->constants->mem_70833)
    #define mem_70834 (ctx->constants->mem_70834)
    #define mem_70835 (ctx->constants->mem_70835)
    #define mem_70836 (ctx->constants->mem_70836)
    #define mem_70837 (ctx->constants->mem_70837)
    #define mem_70838 (ctx->constants->mem_70838)
    #define mem_70839 (ctx->constants->mem_70839)
    #define mem_70840 (ctx->constants->mem_70840)
    #define mem_70841 (ctx->constants->mem_70841)
    mem_70833.references = NULL;
    mem_70834.references = NULL;
    mem_70835.references = NULL;
    mem_70836.references = NULL;
    mem_70837.references = NULL;
    mem_70838.references = NULL;
    mem_70839.references = NULL;
    mem_70840.references = NULL;
    mem_70841.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70833, (int64_t) 3456, "mem_70833")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72942 = 0; nest_i_72942 < (int64_t) 27; nest_i_72942++) {
        for (int64_t nest_i_72943 = 0; nest_i_72943 < (int64_t) 16; nest_i_72943++) {
            ((double *) mem_70833.mem)[nest_i_72942 * (int64_t) 16 + nest_i_72943] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70834, (int64_t) 2048, "mem_70834")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72944 = 0; nest_i_72944 < (int64_t) 16; nest_i_72944++) {
        for (int64_t nest_i_72945 = 0; nest_i_72945 < (int64_t) 16; nest_i_72945++) {
            ((double *) mem_70834.mem)[nest_i_72944 * (int64_t) 16 + nest_i_72945] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70835, (int64_t) 2048, "mem_70835")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72946 = 0; nest_i_72946 < (int64_t) 16; nest_i_72946++) {
        for (int64_t nest_i_72947 = 0; nest_i_72947 < (int64_t) 16; nest_i_72947++) {
            ((double *) mem_70835.mem)[nest_i_72946 * (int64_t) 16 + nest_i_72947] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70836, (int64_t) 2048, "mem_70836")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72948 = 0; nest_i_72948 < (int64_t) 16; nest_i_72948++) {
        for (int64_t nest_i_72949 = 0; nest_i_72949 < (int64_t) 16; nest_i_72949++) {
            ((double *) mem_70836.mem)[nest_i_72948 * (int64_t) 16 + nest_i_72949] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70837, (int64_t) 2048, "mem_70837")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72950 = 0; nest_i_72950 < (int64_t) 16; nest_i_72950++) {
        for (int64_t nest_i_72951 = 0; nest_i_72951 < (int64_t) 16; nest_i_72951++) {
            ((double *) mem_70837.mem)[nest_i_72950 * (int64_t) 16 + nest_i_72951] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70838, (int64_t) 2048, "mem_70838")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72952 = 0; nest_i_72952 < (int64_t) 16; nest_i_72952++) {
        for (int64_t nest_i_72953 = 0; nest_i_72953 < (int64_t) 16; nest_i_72953++) {
            ((double *) mem_70838.mem)[nest_i_72952 * (int64_t) 16 + nest_i_72953] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70839, (int64_t) 8192, "mem_70839")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72954 = 0; nest_i_72954 < (int64_t) 64; nest_i_72954++) {
        for (int64_t nest_i_72955 = 0; nest_i_72955 < (int64_t) 16; nest_i_72955++) {
            ((double *) mem_70839.mem)[nest_i_72954 * (int64_t) 16 + nest_i_72955] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70840, (int64_t) 8192, "mem_70840")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72956 = 0; nest_i_72956 < (int64_t) 16; nest_i_72956++) {
        for (int64_t nest_i_72957 = 0; nest_i_72957 < (int64_t) 64; nest_i_72957++) {
            ((double *) mem_70840.mem)[nest_i_72956 * (int64_t) 64 + nest_i_72957] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70841, (int64_t) 3456, "mem_70841")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_72958 = 0; nest_i_72958 < (int64_t) 27; nest_i_72958++) {
        for (int64_t nest_i_72959 = 0; nest_i_72959 < (int64_t) 16; nest_i_72959++) {
            ((double *) mem_70841.mem)[nest_i_72958 * (int64_t) 16 + nest_i_72959] = 0.0;
        }
    }
    #undef mem_70833
    #undef mem_70834
    #undef mem_70835
    #undef mem_70836
    #undef mem_70837
    #undef mem_70838
    #undef mem_70839
    #undef mem_70840
    #undef mem_70841
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_70833, "ctx->constants->mem_70833") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70834, "ctx->constants->mem_70834") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70835, "ctx->constants->mem_70835") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70836, "ctx->constants->mem_70836") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70837, "ctx->constants->mem_70837") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70838, "ctx->constants->mem_70838") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70839, "ctx->constants->mem_70839") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70840, "ctx->constants->mem_70840") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_70841, "ctx->constants->mem_70841") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_9496(struct futhark_context *ctx, struct memblock *mem_out_p_72960, struct memblock *mem_out_p_72961, struct memblock *mem_out_p_72962, struct memblock w_mem_70842, struct memblock mw_mem_70843, struct memblock vw_mem_70844, struct memblock dw_mem_70845, int64_t n_47868, int64_t m_47869, int64_t step_47874, double lt_r_47875)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_70886_cached_sizze_72963 = 0;
    unsigned char *mem_70886 = NULL;
    int64_t mem_70889_cached_sizze_72964 = 0;
    unsigned char *mem_70889 = NULL;
    struct memblock mem_70924;
    
    mem_70924.references = NULL;
    
    struct memblock mem_70851;
    
    mem_70851.references = NULL;
    
    struct memblock mem_70848;
    
    mem_70848.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock mem_70833 = ctx->constants->mem_70833;
    struct memblock mem_70834 = ctx->constants->mem_70834;
    struct memblock mem_70835 = ctx->constants->mem_70835;
    struct memblock mem_70836 = ctx->constants->mem_70836;
    struct memblock mem_70837 = ctx->constants->mem_70837;
    struct memblock mem_70838 = ctx->constants->mem_70838;
    struct memblock mem_70839 = ctx->constants->mem_70839;
    struct memblock mem_70840 = ctx->constants->mem_70840;
    struct memblock mem_70841 = ctx->constants->mem_70841;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_70846 = (int64_t) 8 * n_47868;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_70847 = m_47869 * binop_x_70846;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70848, bytes_70847, "mem_70848")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70851, bytes_70847, "mem_70851")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_69994 = 0; i_69994 < n_47868; i_69994++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_69987 = 0; i_69987 < m_47869; i_69987++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66107 = ((double *) mw_mem_70843.mem)[i_69994 * m_47869 + i_69987];
            
            // futhark/microgpt.fut:417:10-20
            
            double zp_lhs_66108 = 0.85 * zt_rhs_66107;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66109 = ((double *) dw_mem_70845.mem)[i_69994 * m_47869 + i_69987];
            
            // futhark/microgpt.fut:417:35-45
            
            double zp_rhs_66110 = 0.15000000000000002 * zt_rhs_66109;
            
            // futhark/microgpt.fut:417:21-45
            
            double lifted_lambda_res_66111 = zp_lhs_66108 + zp_rhs_66110;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66118 = ((double *) vw_mem_70844.mem)[i_69994 * m_47869 + i_69987];
            
            // futhark/microgpt.fut:419:10-20
            
            double zp_lhs_66119 = 0.99 * zt_rhs_66118;
            
            // futhark/microgpt.fut:419:35-45
            
            double zt_lhs_66121 = 1.0000000000000009e-2 * zt_rhs_66109;
            
            // futhark/microgpt.fut:419:46-56
            
            double zp_rhs_66122 = zt_rhs_66109 * zt_lhs_66121;
            
            // futhark/microgpt.fut:419:21-56
            
            double lifted_lambda_res_66123 = zp_lhs_66119 + zp_rhs_66122;
            
            ((double *) mem_70848.mem)[i_69994 * m_47869 + i_69987] = lifted_lambda_res_66123;
            ((double *) mem_70851.mem)[i_69994 * m_47869 + i_69987] = lifted_lambda_res_66111;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_52873 = sitofp_i64_f64(step_47874);
    
    // futhark/microgpt.fut:421:54-57
    
    double ztzt_rhs_52874 = 1.0 + i64_res_52873;
    
    // futhark/microgpt.fut:421:30-57
    
    double zm_rhs_52875 = fpow64(0.85, ztzt_rhs_52874);
    
    // futhark/microgpt.fut:421:23-57
    
    double zs_rhs_52876 = 1.0 - zm_rhs_52875;
    
    // futhark/microgpt.fut:423:31-58
    
    double zm_rhs_52914 = fpow64(0.99, ztzt_rhs_52874);
    
    // futhark/microgpt.fut:423:23-58
    
    double zs_rhs_52915 = 1.0 - zm_rhs_52914;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_70886_cached_sizze_72963 < bytes_70847) {
        err = lexical_realloc(ctx, &mem_70886, &mem_70886_cached_sizze_72963, bytes_70847);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_70889_cached_sizze_72964 < bytes_70847) {
        err = lexical_realloc(ctx, &mem_70889, &mem_70889_cached_sizze_72964, bytes_70847);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_70008 = 0; i_70008 < n_47868; i_70008++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70001 = 0; i_70001 < m_47869; i_70001++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_66143 = ((double *) mem_70851.mem)[i_70008 * m_47869 + i_70001];
            
            // futhark/microgpt.fut:421:18-57
            
            double lifted_lambda_res_66144 = zs_lhs_66143 / zs_rhs_52876;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_66151 = ((double *) mem_70848.mem)[i_70008 * m_47869 + i_70001];
            
            // futhark/microgpt.fut:423:18-58
            
            double lifted_lambda_res_66152 = zs_lhs_66151 / zs_rhs_52915;
            
            ((double *) mem_70886)[i_70008 * m_47869 + i_70001] = lifted_lambda_res_66152;
            ((double *) mem_70889)[i_70008 * m_47869 + i_70001] = lifted_lambda_res_66144;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70924, bytes_70847, "mem_70924")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_70017 = 0; i_70017 < n_47868; i_70017++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70013 = 0; i_70013 < m_47869; i_70013++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_52037 = ((double *) w_mem_70842.mem)[i_70017 * m_47869 + i_70013];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_52038 = ((double *) mem_70889)[i_70017 * m_47869 + i_70013];
            
            // futhark/microgpt.fut:425:21-34
            
            double zs_lhs_52039 = lt_r_47875 * zt_rhs_52038;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_52040 = ((double *) mem_70886)[i_70017 * m_47869 + i_70013];
            
            // futhark/microgpt.fut:425:51-57
            
            double zp_lhs_52041 = fpow64(ztzt_lhs_52040, 0.5);
            
            // futhark/microgpt.fut:425:59-71
            
            double zs_rhs_52042 = 1.0e-8 + zp_lhs_52041;
            
            // futhark/microgpt.fut:425:35-71
            
            double zm_rhs_52043 = zs_lhs_52039 / zs_rhs_52042;
            
            // futhark/microgpt.fut:425:13-71
            
            double lifted_lambda_res_52044 = zm_lhs_52037 - zm_rhs_52043;
            
            ((double *) mem_70924.mem)[i_70017 * m_47869 + i_70013] = lifted_lambda_res_52044;
        }
    }
    if (memblock_set(ctx, &mem_out_72641, &mem_70924, "mem_70924") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72642, &mem_70851, "mem_70851") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72643, &mem_70848, "mem_70848") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72960, &mem_out_72641, "mem_out_72641") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72961, &mem_out_72642, "mem_out_72642") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72962, &mem_out_72643, "mem_out_72643") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_70886);
        free(mem_70889);
        if (memblock_unref(ctx, &mem_70924, "mem_70924") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_70851, "mem_70851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_70848, "mem_70848") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72643, "mem_out_72643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72642, "mem_out_72642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72641, "mem_out_72641") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_9497(struct futhark_context *ctx, struct memblock *mem_out_p_72965, struct memblock *mem_out_p_72966, struct memblock *mem_out_p_72967, struct memblock w_mem_70842, struct memblock mw_mem_70843, struct memblock vw_mem_70844, struct memblock dw_mem_70845, int64_t n_48901, int64_t m_48902, int64_t step_48907, double lt_r_48908)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_70886_cached_sizze_72968 = 0;
    unsigned char *mem_70886 = NULL;
    int64_t mem_70889_cached_sizze_72969 = 0;
    unsigned char *mem_70889 = NULL;
    struct memblock mem_70924;
    
    mem_70924.references = NULL;
    
    struct memblock mem_70851;
    
    mem_70851.references = NULL;
    
    struct memblock mem_70848;
    
    mem_70848.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock mem_70833 = ctx->constants->mem_70833;
    struct memblock mem_70834 = ctx->constants->mem_70834;
    struct memblock mem_70835 = ctx->constants->mem_70835;
    struct memblock mem_70836 = ctx->constants->mem_70836;
    struct memblock mem_70837 = ctx->constants->mem_70837;
    struct memblock mem_70838 = ctx->constants->mem_70838;
    struct memblock mem_70839 = ctx->constants->mem_70839;
    struct memblock mem_70840 = ctx->constants->mem_70840;
    struct memblock mem_70841 = ctx->constants->mem_70841;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_70846 = (int64_t) 8 * n_48901;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_70847 = m_48902 * binop_x_70846;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70848, bytes_70847, "mem_70848")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70851, bytes_70847, "mem_70851")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_69994 = 0; i_69994 < n_48901; i_69994++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_69987 = 0; i_69987 < m_48902; i_69987++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66107 = ((double *) mw_mem_70843.mem)[i_69994 * m_48902 + i_69987];
            
            // futhark/microgpt.fut:417:10-20
            
            double zp_lhs_66108 = 0.85 * zt_rhs_66107;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66109 = ((double *) dw_mem_70845.mem)[i_69994 * m_48902 + i_69987];
            
            // futhark/microgpt.fut:417:35-45
            
            double zp_rhs_66110 = 0.15000000000000002 * zt_rhs_66109;
            
            // futhark/microgpt.fut:417:21-45
            
            double lifted_lambda_res_66111 = zp_lhs_66108 + zp_rhs_66110;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_66118 = ((double *) vw_mem_70844.mem)[i_69994 * m_48902 + i_69987];
            
            // futhark/microgpt.fut:419:10-20
            
            double zp_lhs_66119 = 0.99 * zt_rhs_66118;
            
            // futhark/microgpt.fut:419:35-45
            
            double zt_lhs_66121 = 1.0000000000000009e-2 * zt_rhs_66109;
            
            // futhark/microgpt.fut:419:46-56
            
            double zp_rhs_66122 = zt_rhs_66109 * zt_lhs_66121;
            
            // futhark/microgpt.fut:419:21-56
            
            double lifted_lambda_res_66123 = zp_lhs_66119 + zp_rhs_66122;
            
            ((double *) mem_70848.mem)[i_69994 * m_48902 + i_69987] = lifted_lambda_res_66123;
            ((double *) mem_70851.mem)[i_69994 * m_48902 + i_69987] = lifted_lambda_res_66111;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_52873 = sitofp_i64_f64(step_48907);
    
    // futhark/microgpt.fut:421:54-57
    
    double ztzt_rhs_52874 = 1.0 + i64_res_52873;
    
    // futhark/microgpt.fut:421:30-57
    
    double zm_rhs_52875 = fpow64(0.85, ztzt_rhs_52874);
    
    // futhark/microgpt.fut:421:23-57
    
    double zs_rhs_52876 = 1.0 - zm_rhs_52875;
    
    // futhark/microgpt.fut:423:31-58
    
    double zm_rhs_52914 = fpow64(0.99, ztzt_rhs_52874);
    
    // futhark/microgpt.fut:423:23-58
    
    double zs_rhs_52915 = 1.0 - zm_rhs_52914;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_70886_cached_sizze_72968 < bytes_70847) {
        err = lexical_realloc(ctx, &mem_70886, &mem_70886_cached_sizze_72968, bytes_70847);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_70889_cached_sizze_72969 < bytes_70847) {
        err = lexical_realloc(ctx, &mem_70889, &mem_70889_cached_sizze_72969, bytes_70847);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_70008 = 0; i_70008 < n_48901; i_70008++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70001 = 0; i_70001 < m_48902; i_70001++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_66143 = ((double *) mem_70851.mem)[i_70008 * m_48902 + i_70001];
            
            // futhark/microgpt.fut:421:18-57
            
            double lifted_lambda_res_66144 = zs_lhs_66143 / zs_rhs_52876;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_66151 = ((double *) mem_70848.mem)[i_70008 * m_48902 + i_70001];
            
            // futhark/microgpt.fut:423:18-58
            
            double lifted_lambda_res_66152 = zs_lhs_66151 / zs_rhs_52915;
            
            ((double *) mem_70886)[i_70008 * m_48902 + i_70001] = lifted_lambda_res_66152;
            ((double *) mem_70889)[i_70008 * m_48902 + i_70001] = lifted_lambda_res_66144;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_70924, bytes_70847, "mem_70924")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_70017 = 0; i_70017 < n_48901; i_70017++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70013 = 0; i_70013 < m_48902; i_70013++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_52037 = ((double *) w_mem_70842.mem)[i_70017 * m_48902 + i_70013];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_52038 = ((double *) mem_70889)[i_70017 * m_48902 + i_70013];
            
            // futhark/microgpt.fut:425:21-34
            
            double zs_lhs_52039 = lt_r_48908 * zt_rhs_52038;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_52040 = ((double *) mem_70886)[i_70017 * m_48902 + i_70013];
            
            // futhark/microgpt.fut:425:51-57
            
            double zp_lhs_52041 = fpow64(ztzt_lhs_52040, 0.5);
            
            // futhark/microgpt.fut:425:59-71
            
            double zs_rhs_52042 = 1.0e-8 + zp_lhs_52041;
            
            // futhark/microgpt.fut:425:35-71
            
            double zm_rhs_52043 = zs_lhs_52039 / zs_rhs_52042;
            
            // futhark/microgpt.fut:425:13-71
            
            double lifted_lambda_res_52044 = zm_lhs_52037 - zm_rhs_52043;
            
            ((double *) mem_70924.mem)[i_70017 * m_48902 + i_70013] = lifted_lambda_res_52044;
        }
    }
    if (memblock_set(ctx, &mem_out_72641, &mem_70924, "mem_70924") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72642, &mem_70851, "mem_70851") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72643, &mem_70848, "mem_70848") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72965, &mem_out_72641, "mem_out_72641") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72966, &mem_out_72642, "mem_out_72642") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72967, &mem_out_72643, "mem_out_72643") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_70886);
        free(mem_70889);
        if (memblock_unref(ctx, &mem_70924, "mem_70924") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_70851, "mem_70851") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_70848, "mem_70848") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72643, "mem_out_72643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72642, "mem_out_72642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72641, "mem_out_72641") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_72970, struct memblock *mem_out_p_72971, struct memblock *mem_out_p_72972, struct memblock *mem_out_p_72973, struct memblock *mem_out_p_72974, struct memblock *mem_out_p_72975, struct memblock *mem_out_p_72976, struct memblock *mem_out_p_72977, struct memblock *mem_out_p_72978, struct memblock wte_mem_70842, struct memblock wpe_mem_70843, struct memblock wqry_mem_70844, struct memblock wkey_mem_70845, struct memblock wval_mem_70846, struct memblock wout_mem_70847, struct memblock wup_mem_70848, struct memblock wdown_mem_70849, struct memblock wvoc_mem_70850)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock mem_70833 = ctx->constants->mem_70833;
    struct memblock mem_70834 = ctx->constants->mem_70834;
    struct memblock mem_70835 = ctx->constants->mem_70835;
    struct memblock mem_70836 = ctx->constants->mem_70836;
    struct memblock mem_70837 = ctx->constants->mem_70837;
    struct memblock mem_70838 = ctx->constants->mem_70838;
    struct memblock mem_70839 = ctx->constants->mem_70839;
    struct memblock mem_70840 = ctx->constants->mem_70840;
    struct memblock mem_70841 = ctx->constants->mem_70841;
    
    if (memblock_set(ctx, &mem_out_72641, &wdown_mem_70849, "wdown_mem_70849") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72642, &wkey_mem_70845, "wkey_mem_70845") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72643, &wout_mem_70847, "wout_mem_70847") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72644, &wpe_mem_70843, "wpe_mem_70843") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72645, &wqry_mem_70844, "wqry_mem_70844") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72646, &wte_mem_70842, "wte_mem_70842") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72647, &wup_mem_70848, "wup_mem_70848") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72648, &wval_mem_70846, "wval_mem_70846") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72649, &wvoc_mem_70850, "wvoc_mem_70850") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72970, &mem_out_72641, "mem_out_72641") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72971, &mem_out_72642, "mem_out_72642") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72972, &mem_out_72643, "mem_out_72643") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72973, &mem_out_72644, "mem_out_72644") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72974, &mem_out_72645, "mem_out_72645") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72975, &mem_out_72646, "mem_out_72646") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72976, &mem_out_72647, "mem_out_72647") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72977, &mem_out_72648, "mem_out_72648") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72978, &mem_out_72649, "mem_out_72649") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_72649, "mem_out_72649") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72648, "mem_out_72648") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72647, "mem_out_72647") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72646, "mem_out_72646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72645, "mem_out_72645") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72644, "mem_out_72644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72643, "mem_out_72643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72642, "mem_out_72642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72641, "mem_out_72641") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_72979, struct memblock *mem_out_p_72980, struct memblock *mem_out_p_72981, struct memblock *mem_out_p_72982, struct memblock *mem_out_p_72983, struct memblock *mem_out_p_72984, struct memblock *mem_out_p_72985, struct memblock *mem_out_p_72986, struct memblock *mem_out_p_72987, struct memblock *mem_out_p_72988, struct memblock *mem_out_p_72989, struct memblock *mem_out_p_72990, struct memblock *mem_out_p_72991, struct memblock *mem_out_p_72992, struct memblock *mem_out_p_72993, struct memblock *mem_out_p_72994, struct memblock *mem_out_p_72995, struct memblock *mem_out_p_72996, struct memblock *mem_out_p_72997, struct memblock *mem_out_p_72998, struct memblock *mem_out_p_72999, struct memblock *mem_out_p_73000, struct memblock *mem_out_p_73001, struct memblock *mem_out_p_73002, struct memblock *mem_out_p_73003, struct memblock *mem_out_p_73004, struct memblock *mem_out_p_73005, struct memblock wdown_mem_70842, struct memblock wkey_mem_70843, struct memblock wout_mem_70844, struct memblock wpe_mem_70845, struct memblock wqry_mem_70846, struct memblock wte_mem_70847, struct memblock wup_mem_70848, struct memblock wval_mem_70849, struct memblock wvoc_mem_70850, struct memblock wdown_mem_70851, struct memblock wkey_mem_70852, struct memblock wout_mem_70853, struct memblock wpe_mem_70854, struct memblock wqry_mem_70855, struct memblock wte_mem_70856, struct memblock wup_mem_70857, struct memblock wval_mem_70858, struct memblock wvoc_mem_70859, struct memblock wdown_mem_70860, struct memblock wkey_mem_70861, struct memblock wout_mem_70862, struct memblock wpe_mem_70863, struct memblock wqry_mem_70864, struct memblock wte_mem_70865, struct memblock wup_mem_70866, struct memblock wval_mem_70867, struct memblock wvoc_mem_70868, struct memblock masks_mem_70869, struct memblock dls_mem_70870, struct memblock seqs_mem_70871)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_70980_cached_sizze_73006 = 0;
    unsigned char *mem_70980 = NULL;
    int64_t mem_70981_cached_sizze_73007 = 0;
    unsigned char *mem_70981 = NULL;
    int64_t mem_70990_cached_sizze_73008 = 0;
    unsigned char *mem_70990 = NULL;
    int64_t mem_70997_cached_sizze_73009 = 0;
    unsigned char *mem_70997 = NULL;
    int64_t mem_71012_cached_sizze_73010 = 0;
    unsigned char *mem_71012 = NULL;
    int64_t mem_71013_cached_sizze_73011 = 0;
    unsigned char *mem_71013 = NULL;
    int64_t mem_71022_cached_sizze_73012 = 0;
    unsigned char *mem_71022 = NULL;
    int64_t mem_71029_cached_sizze_73013 = 0;
    unsigned char *mem_71029 = NULL;
    int64_t mem_71044_cached_sizze_73014 = 0;
    unsigned char *mem_71044 = NULL;
    int64_t mem_71045_cached_sizze_73015 = 0;
    unsigned char *mem_71045 = NULL;
    int64_t mem_71054_cached_sizze_73016 = 0;
    unsigned char *mem_71054 = NULL;
    int64_t mem_71055_cached_sizze_73017 = 0;
    unsigned char *mem_71055 = NULL;
    int64_t mem_71076_cached_sizze_73018 = 0;
    unsigned char *mem_71076 = NULL;
    int64_t mem_71077_cached_sizze_73019 = 0;
    unsigned char *mem_71077 = NULL;
    int64_t mem_71078_cached_sizze_73020 = 0;
    unsigned char *mem_71078 = NULL;
    int64_t mem_71090_cached_sizze_73021 = 0;
    unsigned char *mem_71090 = NULL;
    int64_t mem_71091_cached_sizze_73022 = 0;
    unsigned char *mem_71091 = NULL;
    int64_t mem_71115_cached_sizze_73023 = 0;
    unsigned char *mem_71115 = NULL;
    int64_t mem_71116_cached_sizze_73024 = 0;
    unsigned char *mem_71116 = NULL;
    int64_t mem_71117_cached_sizze_73025 = 0;
    unsigned char *mem_71117 = NULL;
    int64_t mem_71118_cached_sizze_73026 = 0;
    unsigned char *mem_71118 = NULL;
    int64_t mem_71119_cached_sizze_73027 = 0;
    unsigned char *mem_71119 = NULL;
    int64_t mem_71138_cached_sizze_73028 = 0;
    unsigned char *mem_71138 = NULL;
    int64_t mem_71139_cached_sizze_73029 = 0;
    unsigned char *mem_71139 = NULL;
    int64_t mem_71140_cached_sizze_73030 = 0;
    unsigned char *mem_71140 = NULL;
    int64_t mem_71177_cached_sizze_73031 = 0;
    unsigned char *mem_71177 = NULL;
    int64_t mem_71178_cached_sizze_73032 = 0;
    unsigned char *mem_71178 = NULL;
    int64_t mem_71179_cached_sizze_73033 = 0;
    unsigned char *mem_71179 = NULL;
    int64_t mem_71195_cached_sizze_73034 = 0;
    unsigned char *mem_71195 = NULL;
    int64_t mem_71196_cached_sizze_73035 = 0;
    unsigned char *mem_71196 = NULL;
    int64_t mem_71197_cached_sizze_73036 = 0;
    unsigned char *mem_71197 = NULL;
    int64_t mem_71210_cached_sizze_73037 = 0;
    unsigned char *mem_71210 = NULL;
    int64_t mem_71211_cached_sizze_73038 = 0;
    unsigned char *mem_71211 = NULL;
    int64_t mem_71212_cached_sizze_73039 = 0;
    unsigned char *mem_71212 = NULL;
    int64_t mem_71258_cached_sizze_73040 = 0;
    unsigned char *mem_71258 = NULL;
    int64_t mem_71259_cached_sizze_73041 = 0;
    unsigned char *mem_71259 = NULL;
    int64_t mem_71270_cached_sizze_73042 = 0;
    unsigned char *mem_71270 = NULL;
    int64_t mem_71271_cached_sizze_73043 = 0;
    unsigned char *mem_71271 = NULL;
    int64_t mem_71280_cached_sizze_73044 = 0;
    unsigned char *mem_71280 = NULL;
    int64_t mem_71281_cached_sizze_73045 = 0;
    unsigned char *mem_71281 = NULL;
    int64_t mem_71302_cached_sizze_73046 = 0;
    unsigned char *mem_71302 = NULL;
    int64_t mem_71307_cached_sizze_73047 = 0;
    unsigned char *mem_71307 = NULL;
    int64_t mem_71318_cached_sizze_73048 = 0;
    unsigned char *mem_71318 = NULL;
    int64_t mem_71323_cached_sizze_73049 = 0;
    unsigned char *mem_71323 = NULL;
    int64_t mem_71330_cached_sizze_73050 = 0;
    unsigned char *mem_71330 = NULL;
    int64_t mem_71337_cached_sizze_73051 = 0;
    unsigned char *mem_71337 = NULL;
    int64_t mem_71348_cached_sizze_73052 = 0;
    unsigned char *mem_71348 = NULL;
    int64_t mem_71353_cached_sizze_73053 = 0;
    unsigned char *mem_71353 = NULL;
    int64_t mem_71374_cached_sizze_73054 = 0;
    unsigned char *mem_71374 = NULL;
    int64_t mem_71375_cached_sizze_73055 = 0;
    unsigned char *mem_71375 = NULL;
    int64_t mem_71383_cached_sizze_73056 = 0;
    unsigned char *mem_71383 = NULL;
    int64_t mem_71397_cached_sizze_73057 = 0;
    unsigned char *mem_71397 = NULL;
    int64_t mem_71402_cached_sizze_73058 = 0;
    unsigned char *mem_71402 = NULL;
    int64_t mem_71413_cached_sizze_73059 = 0;
    unsigned char *mem_71413 = NULL;
    int64_t mem_71418_cached_sizze_73060 = 0;
    unsigned char *mem_71418 = NULL;
    int64_t mem_71429_cached_sizze_73061 = 0;
    unsigned char *mem_71429 = NULL;
    int64_t mem_71430_cached_sizze_73062 = 0;
    unsigned char *mem_71430 = NULL;
    int64_t mem_71439_cached_sizze_73063 = 0;
    unsigned char *mem_71439 = NULL;
    int64_t mem_71440_cached_sizze_73064 = 0;
    unsigned char *mem_71440 = NULL;
    int64_t mem_71461_cached_sizze_73065 = 0;
    unsigned char *mem_71461 = NULL;
    int64_t mem_71462_cached_sizze_73066 = 0;
    unsigned char *mem_71462 = NULL;
    int64_t mem_71470_cached_sizze_73067 = 0;
    unsigned char *mem_71470 = NULL;
    int64_t mem_71484_cached_sizze_73068 = 0;
    unsigned char *mem_71484 = NULL;
    int64_t mem_71485_cached_sizze_73069 = 0;
    unsigned char *mem_71485 = NULL;
    int64_t mem_71493_cached_sizze_73070 = 0;
    unsigned char *mem_71493 = NULL;
    int64_t mem_71507_cached_sizze_73071 = 0;
    unsigned char *mem_71507 = NULL;
    int64_t mem_71512_cached_sizze_73072 = 0;
    unsigned char *mem_71512 = NULL;
    int64_t mem_71523_cached_sizze_73073 = 0;
    unsigned char *mem_71523 = NULL;
    int64_t mem_71528_cached_sizze_73074 = 0;
    unsigned char *mem_71528 = NULL;
    int64_t mem_71539_cached_sizze_73075 = 0;
    unsigned char *mem_71539 = NULL;
    int64_t mem_71544_cached_sizze_73076 = 0;
    unsigned char *mem_71544 = NULL;
    int64_t mem_71555_cached_sizze_73077 = 0;
    unsigned char *mem_71555 = NULL;
    int64_t mem_71556_cached_sizze_73078 = 0;
    unsigned char *mem_71556 = NULL;
    int64_t mem_71565_cached_sizze_73079 = 0;
    unsigned char *mem_71565 = NULL;
    int64_t mem_71566_cached_sizze_73080 = 0;
    unsigned char *mem_71566 = NULL;
    int64_t mem_71579_cached_sizze_73081 = 0;
    unsigned char *mem_71579 = NULL;
    int64_t mem_71580_cached_sizze_73082 = 0;
    unsigned char *mem_71580 = NULL;
    int64_t mem_71593_cached_sizze_73083 = 0;
    unsigned char *mem_71593 = NULL;
    int64_t mem_71594_cached_sizze_73084 = 0;
    unsigned char *mem_71594 = NULL;
    int64_t mem_71615_cached_sizze_73085 = 0;
    unsigned char *mem_71615 = NULL;
    int64_t mem_71622_cached_sizze_73086 = 0;
    unsigned char *mem_71622 = NULL;
    int64_t mem_71627_cached_sizze_73087 = 0;
    unsigned char *mem_71627 = NULL;
    int64_t mem_71638_cached_sizze_73088 = 0;
    unsigned char *mem_71638 = NULL;
    int64_t mem_71643_cached_sizze_73089 = 0;
    unsigned char *mem_71643 = NULL;
    int64_t mem_71654_cached_sizze_73090 = 0;
    unsigned char *mem_71654 = NULL;
    int64_t mem_71655_cached_sizze_73091 = 0;
    unsigned char *mem_71655 = NULL;
    int64_t mem_71664_cached_sizze_73092 = 0;
    unsigned char *mem_71664 = NULL;
    int64_t mem_71665_cached_sizze_73093 = 0;
    unsigned char *mem_71665 = NULL;
    int64_t mem_71686_cached_sizze_73094 = 0;
    unsigned char *mem_71686 = NULL;
    int64_t mem_71691_cached_sizze_73095 = 0;
    unsigned char *mem_71691 = NULL;
    int64_t mem_71702_cached_sizze_73096 = 0;
    unsigned char *mem_71702 = NULL;
    int64_t mem_71707_cached_sizze_73097 = 0;
    unsigned char *mem_71707 = NULL;
    int64_t mem_71718_cached_sizze_73098 = 0;
    unsigned char *mem_71718 = NULL;
    int64_t mem_71725_cached_sizze_73099 = 0;
    unsigned char *mem_71725 = NULL;
    int64_t mem_71732_cached_sizze_73100 = 0;
    unsigned char *mem_71732 = NULL;
    int64_t mem_71742_cached_sizze_73101 = 0;
    unsigned char *mem_71742 = NULL;
    int64_t mem_71747_cached_sizze_73102 = 0;
    unsigned char *mem_71747 = NULL;
    int64_t mem_71758_cached_sizze_73103 = 0;
    unsigned char *mem_71758 = NULL;
    int64_t mem_71759_cached_sizze_73104 = 0;
    unsigned char *mem_71759 = NULL;
    int64_t mem_71768_cached_sizze_73105 = 0;
    unsigned char *mem_71768 = NULL;
    int64_t mem_71769_cached_sizze_73106 = 0;
    unsigned char *mem_71769 = NULL;
    int64_t mem_71790_cached_sizze_73107 = 0;
    unsigned char *mem_71790 = NULL;
    int64_t mem_71791_cached_sizze_73108 = 0;
    unsigned char *mem_71791 = NULL;
    int64_t mem_71802_cached_sizze_73109 = 0;
    unsigned char *mem_71802 = NULL;
    int64_t mem_71803_cached_sizze_73110 = 0;
    unsigned char *mem_71803 = NULL;
    int64_t mem_71812_cached_sizze_73111 = 0;
    unsigned char *mem_71812 = NULL;
    int64_t mem_71819_cached_sizze_73112 = 0;
    unsigned char *mem_71819 = NULL;
    int64_t mem_71844_cached_sizze_73113 = 0;
    unsigned char *mem_71844 = NULL;
    int64_t mem_71845_cached_sizze_73114 = 0;
    unsigned char *mem_71845 = NULL;
    int64_t mem_71856_cached_sizze_73115 = 0;
    unsigned char *mem_71856 = NULL;
    int64_t mem_71857_cached_sizze_73116 = 0;
    unsigned char *mem_71857 = NULL;
    int64_t mem_71866_cached_sizze_73117 = 0;
    unsigned char *mem_71866 = NULL;
    int64_t mem_71873_cached_sizze_73118 = 0;
    unsigned char *mem_71873 = NULL;
    int64_t mem_71880_cached_sizze_73119 = 0;
    unsigned char *mem_71880 = NULL;
    int64_t mem_71887_cached_sizze_73120 = 0;
    unsigned char *mem_71887 = NULL;
    int64_t mem_71912_cached_sizze_73121 = 0;
    unsigned char *mem_71912 = NULL;
    int64_t mem_71913_cached_sizze_73122 = 0;
    unsigned char *mem_71913 = NULL;
    int64_t mem_71924_cached_sizze_73123 = 0;
    unsigned char *mem_71924 = NULL;
    int64_t mem_71925_cached_sizze_73124 = 0;
    unsigned char *mem_71925 = NULL;
    int64_t mem_71934_cached_sizze_73125 = 0;
    unsigned char *mem_71934 = NULL;
    int64_t mem_71941_cached_sizze_73126 = 0;
    unsigned char *mem_71941 = NULL;
    int64_t mem_71966_cached_sizze_73127 = 0;
    unsigned char *mem_71966 = NULL;
    int64_t mem_71971_cached_sizze_73128 = 0;
    unsigned char *mem_71971 = NULL;
    int64_t mem_71982_cached_sizze_73129 = 0;
    unsigned char *mem_71982 = NULL;
    int64_t mem_71988_cached_sizze_73130 = 0;
    unsigned char *mem_71988 = NULL;
    int64_t mem_71993_cached_sizze_73131 = 0;
    unsigned char *mem_71993 = NULL;
    int64_t mem_72009_cached_sizze_73132 = 0;
    unsigned char *mem_72009 = NULL;
    int64_t mem_72015_cached_sizze_73133 = 0;
    unsigned char *mem_72015 = NULL;
    int64_t mem_72020_cached_sizze_73134 = 0;
    unsigned char *mem_72020 = NULL;
    int64_t mem_72036_cached_sizze_73135 = 0;
    unsigned char *mem_72036 = NULL;
    int64_t mem_72037_cached_sizze_73136 = 0;
    unsigned char *mem_72037 = NULL;
    int64_t mem_72048_cached_sizze_73137 = 0;
    unsigned char *mem_72048 = NULL;
    int64_t mem_72049_cached_sizze_73138 = 0;
    unsigned char *mem_72049 = NULL;
    int64_t mem_72058_cached_sizze_73139 = 0;
    unsigned char *mem_72058 = NULL;
    int64_t mem_72059_cached_sizze_73140 = 0;
    unsigned char *mem_72059 = NULL;
    int64_t mem_72090_cached_sizze_73141 = 0;
    unsigned char *mem_72090 = NULL;
    int64_t mem_72091_cached_sizze_73142 = 0;
    unsigned char *mem_72091 = NULL;
    int64_t mem_72092_cached_sizze_73143 = 0;
    unsigned char *mem_72092 = NULL;
    int64_t mem_72105_cached_sizze_73144 = 0;
    unsigned char *mem_72105 = NULL;
    int64_t mem_72106_cached_sizze_73145 = 0;
    unsigned char *mem_72106 = NULL;
    int64_t mem_72107_cached_sizze_73146 = 0;
    unsigned char *mem_72107 = NULL;
    int64_t mem_72138_cached_sizze_73147 = 0;
    unsigned char *mem_72138 = NULL;
    int64_t mem_72139_cached_sizze_73148 = 0;
    unsigned char *mem_72139 = NULL;
    int64_t mem_72140_cached_sizze_73149 = 0;
    unsigned char *mem_72140 = NULL;
    int64_t mem_72141_cached_sizze_73150 = 0;
    unsigned char *mem_72141 = NULL;
    int64_t mem_72158_cached_sizze_73151 = 0;
    unsigned char *mem_72158 = NULL;
    int64_t mem_72159_cached_sizze_73152 = 0;
    unsigned char *mem_72159 = NULL;
    int64_t mem_72160_cached_sizze_73153 = 0;
    unsigned char *mem_72160 = NULL;
    int64_t mem_72161_cached_sizze_73154 = 0;
    unsigned char *mem_72161 = NULL;
    int64_t mem_72202_cached_sizze_73155 = 0;
    unsigned char *mem_72202 = NULL;
    int64_t mem_72209_cached_sizze_73156 = 0;
    unsigned char *mem_72209 = NULL;
    int64_t mem_72216_cached_sizze_73157 = 0;
    unsigned char *mem_72216 = NULL;
    int64_t mem_72226_cached_sizze_73158 = 0;
    unsigned char *mem_72226 = NULL;
    int64_t mem_72231_cached_sizze_73159 = 0;
    unsigned char *mem_72231 = NULL;
    int64_t mem_72242_cached_sizze_73160 = 0;
    unsigned char *mem_72242 = NULL;
    int64_t mem_72249_cached_sizze_73161 = 0;
    unsigned char *mem_72249 = NULL;
    int64_t mem_72256_cached_sizze_73162 = 0;
    unsigned char *mem_72256 = NULL;
    int64_t mem_72266_cached_sizze_73163 = 0;
    unsigned char *mem_72266 = NULL;
    int64_t mem_72271_cached_sizze_73164 = 0;
    unsigned char *mem_72271 = NULL;
    int64_t mem_72282_cached_sizze_73165 = 0;
    unsigned char *mem_72282 = NULL;
    int64_t mem_72283_cached_sizze_73166 = 0;
    unsigned char *mem_72283 = NULL;
    int64_t mem_72292_cached_sizze_73167 = 0;
    unsigned char *mem_72292 = NULL;
    int64_t mem_72293_cached_sizze_73168 = 0;
    unsigned char *mem_72293 = NULL;
    int64_t mem_72314_cached_sizze_73169 = 0;
    unsigned char *mem_72314 = NULL;
    int64_t mem_72319_cached_sizze_73170 = 0;
    unsigned char *mem_72319 = NULL;
    int64_t mem_72330_cached_sizze_73171 = 0;
    unsigned char *mem_72330 = NULL;
    int64_t mem_72331_cached_sizze_73172 = 0;
    unsigned char *mem_72331 = NULL;
    int64_t mem_72340_cached_sizze_73173 = 0;
    unsigned char *mem_72340 = NULL;
    int64_t mem_72341_cached_sizze_73174 = 0;
    unsigned char *mem_72341 = NULL;
    struct memblock mem_param_tmp_72694;
    
    mem_param_tmp_72694.references = NULL;
    
    struct memblock mem_param_tmp_72693;
    
    mem_param_tmp_72693.references = NULL;
    
    struct memblock mem_param_tmp_72692;
    
    mem_param_tmp_72692.references = NULL;
    
    struct memblock mem_param_tmp_72691;
    
    mem_param_tmp_72691.references = NULL;
    
    struct memblock mem_param_tmp_72690;
    
    mem_param_tmp_72690.references = NULL;
    
    struct memblock mem_param_tmp_72689;
    
    mem_param_tmp_72689.references = NULL;
    
    struct memblock mem_param_tmp_72688;
    
    mem_param_tmp_72688.references = NULL;
    
    struct memblock mem_param_tmp_72687;
    
    mem_param_tmp_72687.references = NULL;
    
    struct memblock mem_param_tmp_72686;
    
    mem_param_tmp_72686.references = NULL;
    
    struct memblock mem_param_tmp_72685;
    
    mem_param_tmp_72685.references = NULL;
    
    struct memblock mem_param_tmp_72684;
    
    mem_param_tmp_72684.references = NULL;
    
    struct memblock mem_param_tmp_72683;
    
    mem_param_tmp_72683.references = NULL;
    
    struct memblock mem_param_tmp_72682;
    
    mem_param_tmp_72682.references = NULL;
    
    struct memblock mem_param_tmp_72681;
    
    mem_param_tmp_72681.references = NULL;
    
    struct memblock mem_param_tmp_72680;
    
    mem_param_tmp_72680.references = NULL;
    
    struct memblock mem_param_tmp_72679;
    
    mem_param_tmp_72679.references = NULL;
    
    struct memblock mem_param_tmp_72678;
    
    mem_param_tmp_72678.references = NULL;
    
    struct memblock mem_param_tmp_72677;
    
    mem_param_tmp_72677.references = NULL;
    
    struct memblock mem_param_tmp_72676;
    
    mem_param_tmp_72676.references = NULL;
    
    struct memblock mem_param_tmp_72675;
    
    mem_param_tmp_72675.references = NULL;
    
    struct memblock mem_param_tmp_72674;
    
    mem_param_tmp_72674.references = NULL;
    
    struct memblock mem_param_tmp_72673;
    
    mem_param_tmp_72673.references = NULL;
    
    struct memblock mem_param_tmp_72672;
    
    mem_param_tmp_72672.references = NULL;
    
    struct memblock mem_param_tmp_72671;
    
    mem_param_tmp_72671.references = NULL;
    
    struct memblock mem_param_tmp_72670;
    
    mem_param_tmp_72670.references = NULL;
    
    struct memblock mem_param_tmp_72669;
    
    mem_param_tmp_72669.references = NULL;
    
    struct memblock mem_param_tmp_72668;
    
    mem_param_tmp_72668.references = NULL;
    
    struct memblock ext_mem_72458;
    
    ext_mem_72458.references = NULL;
    
    struct memblock ext_mem_72459;
    
    ext_mem_72459.references = NULL;
    
    struct memblock ext_mem_72460;
    
    ext_mem_72460.references = NULL;
    
    struct memblock mem_72456;
    
    mem_72456.references = NULL;
    
    struct memblock mem_72454;
    
    mem_72454.references = NULL;
    
    struct memblock mem_72452;
    
    mem_72452.references = NULL;
    
    struct memblock mem_72450;
    
    mem_72450.references = NULL;
    
    struct memblock ext_mem_72447;
    
    ext_mem_72447.references = NULL;
    
    struct memblock ext_mem_72448;
    
    ext_mem_72448.references = NULL;
    
    struct memblock ext_mem_72449;
    
    ext_mem_72449.references = NULL;
    
    struct memblock mem_72445;
    
    mem_72445.references = NULL;
    
    struct memblock mem_72443;
    
    mem_72443.references = NULL;
    
    struct memblock mem_72441;
    
    mem_72441.references = NULL;
    
    struct memblock mem_72439;
    
    mem_72439.references = NULL;
    
    struct memblock ext_mem_72436;
    
    ext_mem_72436.references = NULL;
    
    struct memblock ext_mem_72437;
    
    ext_mem_72437.references = NULL;
    
    struct memblock ext_mem_72438;
    
    ext_mem_72438.references = NULL;
    
    struct memblock mem_72434;
    
    mem_72434.references = NULL;
    
    struct memblock mem_72432;
    
    mem_72432.references = NULL;
    
    struct memblock mem_72430;
    
    mem_72430.references = NULL;
    
    struct memblock mem_72428;
    
    mem_72428.references = NULL;
    
    struct memblock ext_mem_72425;
    
    ext_mem_72425.references = NULL;
    
    struct memblock ext_mem_72426;
    
    ext_mem_72426.references = NULL;
    
    struct memblock ext_mem_72427;
    
    ext_mem_72427.references = NULL;
    
    struct memblock mem_72423;
    
    mem_72423.references = NULL;
    
    struct memblock mem_72421;
    
    mem_72421.references = NULL;
    
    struct memblock mem_72419;
    
    mem_72419.references = NULL;
    
    struct memblock mem_72417;
    
    mem_72417.references = NULL;
    
    struct memblock ext_mem_72414;
    
    ext_mem_72414.references = NULL;
    
    struct memblock ext_mem_72415;
    
    ext_mem_72415.references = NULL;
    
    struct memblock ext_mem_72416;
    
    ext_mem_72416.references = NULL;
    
    struct memblock mem_72412;
    
    mem_72412.references = NULL;
    
    struct memblock mem_72410;
    
    mem_72410.references = NULL;
    
    struct memblock mem_72408;
    
    mem_72408.references = NULL;
    
    struct memblock mem_72406;
    
    mem_72406.references = NULL;
    
    struct memblock ext_mem_72403;
    
    ext_mem_72403.references = NULL;
    
    struct memblock ext_mem_72404;
    
    ext_mem_72404.references = NULL;
    
    struct memblock ext_mem_72405;
    
    ext_mem_72405.references = NULL;
    
    struct memblock mem_72401;
    
    mem_72401.references = NULL;
    
    struct memblock mem_72399;
    
    mem_72399.references = NULL;
    
    struct memblock mem_72397;
    
    mem_72397.references = NULL;
    
    struct memblock mem_72395;
    
    mem_72395.references = NULL;
    
    struct memblock ext_mem_72392;
    
    ext_mem_72392.references = NULL;
    
    struct memblock ext_mem_72393;
    
    ext_mem_72393.references = NULL;
    
    struct memblock ext_mem_72394;
    
    ext_mem_72394.references = NULL;
    
    struct memblock mem_72390;
    
    mem_72390.references = NULL;
    
    struct memblock mem_72388;
    
    mem_72388.references = NULL;
    
    struct memblock mem_72386;
    
    mem_72386.references = NULL;
    
    struct memblock mem_72384;
    
    mem_72384.references = NULL;
    
    struct memblock ext_mem_72381;
    
    ext_mem_72381.references = NULL;
    
    struct memblock ext_mem_72382;
    
    ext_mem_72382.references = NULL;
    
    struct memblock ext_mem_72383;
    
    ext_mem_72383.references = NULL;
    
    struct memblock mem_72379;
    
    mem_72379.references = NULL;
    
    struct memblock mem_72377;
    
    mem_72377.references = NULL;
    
    struct memblock mem_72375;
    
    mem_72375.references = NULL;
    
    struct memblock mem_72373;
    
    mem_72373.references = NULL;
    
    struct memblock ext_mem_72370;
    
    ext_mem_72370.references = NULL;
    
    struct memblock ext_mem_72371;
    
    ext_mem_72371.references = NULL;
    
    struct memblock ext_mem_72372;
    
    ext_mem_72372.references = NULL;
    
    struct memblock mem_72368;
    
    mem_72368.references = NULL;
    
    struct memblock mem_72366;
    
    mem_72366.references = NULL;
    
    struct memblock mem_72364;
    
    mem_72364.references = NULL;
    
    struct memblock mem_72362;
    
    mem_72362.references = NULL;
    
    struct memblock mem_param_70979;
    
    mem_param_70979.references = NULL;
    
    struct memblock mem_param_70975;
    
    mem_param_70975.references = NULL;
    
    struct memblock mem_param_70971;
    
    mem_param_70971.references = NULL;
    
    struct memblock mem_param_70967;
    
    mem_param_70967.references = NULL;
    
    struct memblock mem_param_70963;
    
    mem_param_70963.references = NULL;
    
    struct memblock mem_param_70959;
    
    mem_param_70959.references = NULL;
    
    struct memblock mem_param_70955;
    
    mem_param_70955.references = NULL;
    
    struct memblock mem_param_70951;
    
    mem_param_70951.references = NULL;
    
    struct memblock mem_param_70947;
    
    mem_param_70947.references = NULL;
    
    struct memblock mem_param_70943;
    
    mem_param_70943.references = NULL;
    
    struct memblock mem_param_70939;
    
    mem_param_70939.references = NULL;
    
    struct memblock mem_param_70935;
    
    mem_param_70935.references = NULL;
    
    struct memblock mem_param_70931;
    
    mem_param_70931.references = NULL;
    
    struct memblock mem_param_70927;
    
    mem_param_70927.references = NULL;
    
    struct memblock mem_param_70923;
    
    mem_param_70923.references = NULL;
    
    struct memblock mem_param_70919;
    
    mem_param_70919.references = NULL;
    
    struct memblock mem_param_70915;
    
    mem_param_70915.references = NULL;
    
    struct memblock mem_param_70911;
    
    mem_param_70911.references = NULL;
    
    struct memblock mem_param_70907;
    
    mem_param_70907.references = NULL;
    
    struct memblock mem_param_70903;
    
    mem_param_70903.references = NULL;
    
    struct memblock mem_param_70899;
    
    mem_param_70899.references = NULL;
    
    struct memblock mem_param_70895;
    
    mem_param_70895.references = NULL;
    
    struct memblock mem_param_70891;
    
    mem_param_70891.references = NULL;
    
    struct memblock mem_param_70887;
    
    mem_param_70887.references = NULL;
    
    struct memblock mem_param_70883;
    
    mem_param_70883.references = NULL;
    
    struct memblock mem_param_70879;
    
    mem_param_70879.references = NULL;
    
    struct memblock mem_param_70875;
    
    mem_param_70875.references = NULL;
    
    struct memblock ext_mem_72542;
    
    ext_mem_72542.references = NULL;
    
    struct memblock ext_mem_72543;
    
    ext_mem_72543.references = NULL;
    
    struct memblock ext_mem_72544;
    
    ext_mem_72544.references = NULL;
    
    struct memblock ext_mem_72545;
    
    ext_mem_72545.references = NULL;
    
    struct memblock ext_mem_72546;
    
    ext_mem_72546.references = NULL;
    
    struct memblock ext_mem_72547;
    
    ext_mem_72547.references = NULL;
    
    struct memblock ext_mem_72548;
    
    ext_mem_72548.references = NULL;
    
    struct memblock ext_mem_72549;
    
    ext_mem_72549.references = NULL;
    
    struct memblock ext_mem_72550;
    
    ext_mem_72550.references = NULL;
    
    struct memblock ext_mem_72551;
    
    ext_mem_72551.references = NULL;
    
    struct memblock ext_mem_72552;
    
    ext_mem_72552.references = NULL;
    
    struct memblock ext_mem_72553;
    
    ext_mem_72553.references = NULL;
    
    struct memblock ext_mem_72554;
    
    ext_mem_72554.references = NULL;
    
    struct memblock ext_mem_72555;
    
    ext_mem_72555.references = NULL;
    
    struct memblock ext_mem_72556;
    
    ext_mem_72556.references = NULL;
    
    struct memblock ext_mem_72557;
    
    ext_mem_72557.references = NULL;
    
    struct memblock ext_mem_72558;
    
    ext_mem_72558.references = NULL;
    
    struct memblock ext_mem_72559;
    
    ext_mem_72559.references = NULL;
    
    struct memblock ext_mem_72560;
    
    ext_mem_72560.references = NULL;
    
    struct memblock ext_mem_72561;
    
    ext_mem_72561.references = NULL;
    
    struct memblock ext_mem_72562;
    
    ext_mem_72562.references = NULL;
    
    struct memblock ext_mem_72563;
    
    ext_mem_72563.references = NULL;
    
    struct memblock ext_mem_72564;
    
    ext_mem_72564.references = NULL;
    
    struct memblock ext_mem_72565;
    
    ext_mem_72565.references = NULL;
    
    struct memblock ext_mem_72566;
    
    ext_mem_72566.references = NULL;
    
    struct memblock ext_mem_72567;
    
    ext_mem_72567.references = NULL;
    
    struct memblock ext_mem_72568;
    
    ext_mem_72568.references = NULL;
    
    struct memblock mem_out_72667;
    
    mem_out_72667.references = NULL;
    
    struct memblock mem_out_72666;
    
    mem_out_72666.references = NULL;
    
    struct memblock mem_out_72665;
    
    mem_out_72665.references = NULL;
    
    struct memblock mem_out_72664;
    
    mem_out_72664.references = NULL;
    
    struct memblock mem_out_72663;
    
    mem_out_72663.references = NULL;
    
    struct memblock mem_out_72662;
    
    mem_out_72662.references = NULL;
    
    struct memblock mem_out_72661;
    
    mem_out_72661.references = NULL;
    
    struct memblock mem_out_72660;
    
    mem_out_72660.references = NULL;
    
    struct memblock mem_out_72659;
    
    mem_out_72659.references = NULL;
    
    struct memblock mem_out_72658;
    
    mem_out_72658.references = NULL;
    
    struct memblock mem_out_72657;
    
    mem_out_72657.references = NULL;
    
    struct memblock mem_out_72656;
    
    mem_out_72656.references = NULL;
    
    struct memblock mem_out_72655;
    
    mem_out_72655.references = NULL;
    
    struct memblock mem_out_72654;
    
    mem_out_72654.references = NULL;
    
    struct memblock mem_out_72653;
    
    mem_out_72653.references = NULL;
    
    struct memblock mem_out_72652;
    
    mem_out_72652.references = NULL;
    
    struct memblock mem_out_72651;
    
    mem_out_72651.references = NULL;
    
    struct memblock mem_out_72650;
    
    mem_out_72650.references = NULL;
    
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock mem_70833 = ctx->constants->mem_70833;
    struct memblock mem_70834 = ctx->constants->mem_70834;
    struct memblock mem_70835 = ctx->constants->mem_70835;
    struct memblock mem_70836 = ctx->constants->mem_70836;
    struct memblock mem_70837 = ctx->constants->mem_70837;
    struct memblock mem_70838 = ctx->constants->mem_70838;
    struct memblock mem_70839 = ctx->constants->mem_70839;
    struct memblock mem_70840 = ctx->constants->mem_70840;
    struct memblock mem_70841 = ctx->constants->mem_70841;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_70980_cached_sizze_73006 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_70980, &mem_70980_cached_sizze_73006, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_70981_cached_sizze_73007 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_70981, &mem_70981_cached_sizze_73007, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_70990_cached_sizze_73008 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_70990, &mem_70990_cached_sizze_73008, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_70997_cached_sizze_73009 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_70997, &mem_70997_cached_sizze_73009, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71012_cached_sizze_73010 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_71012, &mem_71012_cached_sizze_73010, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71013_cached_sizze_73011 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71013, &mem_71013_cached_sizze_73011, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71022_cached_sizze_73012 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71022, &mem_71022_cached_sizze_73012, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71029_cached_sizze_73013 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71029, &mem_71029_cached_sizze_73013, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71044_cached_sizze_73014 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71044, &mem_71044_cached_sizze_73014, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71045_cached_sizze_73015 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71045, &mem_71045_cached_sizze_73015, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71054_cached_sizze_73016 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71054, &mem_71054_cached_sizze_73016, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71055_cached_sizze_73017 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71055, &mem_71055_cached_sizze_73017, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71076_cached_sizze_73018 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71076, &mem_71076_cached_sizze_73018, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71077_cached_sizze_73019 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71077, &mem_71077_cached_sizze_73019, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71078_cached_sizze_73020 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71078, &mem_71078_cached_sizze_73020, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71090_cached_sizze_73021 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71090, &mem_71090_cached_sizze_73021, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71091_cached_sizze_73022 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71091, &mem_71091_cached_sizze_73022, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71115_cached_sizze_73023 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71115, &mem_71115_cached_sizze_73023, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71116_cached_sizze_73024 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71116, &mem_71116_cached_sizze_73024, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71117_cached_sizze_73025 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71117, &mem_71117_cached_sizze_73025, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71118_cached_sizze_73026 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71118, &mem_71118_cached_sizze_73026, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71119_cached_sizze_73027 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71119, &mem_71119_cached_sizze_73027, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71138_cached_sizze_73028 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71138, &mem_71138_cached_sizze_73028, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71139_cached_sizze_73029 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71139, &mem_71139_cached_sizze_73029, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71140_cached_sizze_73030 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71140, &mem_71140_cached_sizze_73030, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71177_cached_sizze_73031 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71177, &mem_71177_cached_sizze_73031, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71178_cached_sizze_73032 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71178, &mem_71178_cached_sizze_73032, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71179_cached_sizze_73033 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71179, &mem_71179_cached_sizze_73033, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71195_cached_sizze_73034 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71195, &mem_71195_cached_sizze_73034, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71196_cached_sizze_73035 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71196, &mem_71196_cached_sizze_73035, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71197_cached_sizze_73036 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71197, &mem_71197_cached_sizze_73036, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71210_cached_sizze_73037 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71210, &mem_71210_cached_sizze_73037, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71211_cached_sizze_73038 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71211, &mem_71211_cached_sizze_73038, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71212_cached_sizze_73039 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71212, &mem_71212_cached_sizze_73039, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71258_cached_sizze_73040 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71258, &mem_71258_cached_sizze_73040, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71259_cached_sizze_73041 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71259, &mem_71259_cached_sizze_73041, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71270_cached_sizze_73042 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71270, &mem_71270_cached_sizze_73042, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71271_cached_sizze_73043 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71271, &mem_71271_cached_sizze_73043, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71280_cached_sizze_73044 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71280, &mem_71280_cached_sizze_73044, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71281_cached_sizze_73045 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71281, &mem_71281_cached_sizze_73045, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71302_cached_sizze_73046 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71302, &mem_71302_cached_sizze_73046, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71307_cached_sizze_73047 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71307, &mem_71307_cached_sizze_73047, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71318_cached_sizze_73048 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71318, &mem_71318_cached_sizze_73048, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71323_cached_sizze_73049 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71323, &mem_71323_cached_sizze_73049, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71330_cached_sizze_73050 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71330, &mem_71330_cached_sizze_73050, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71337_cached_sizze_73051 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71337, &mem_71337_cached_sizze_73051, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71348_cached_sizze_73052 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71348, &mem_71348_cached_sizze_73052, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71353_cached_sizze_73053 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71353, &mem_71353_cached_sizze_73053, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71374_cached_sizze_73054 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71374, &mem_71374_cached_sizze_73054, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71375_cached_sizze_73055 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71375, &mem_71375_cached_sizze_73055, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71383_cached_sizze_73056 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71383, &mem_71383_cached_sizze_73056, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71397_cached_sizze_73057 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71397, &mem_71397_cached_sizze_73057, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71402_cached_sizze_73058 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71402, &mem_71402_cached_sizze_73058, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71413_cached_sizze_73059 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71413, &mem_71413_cached_sizze_73059, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71418_cached_sizze_73060 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71418, &mem_71418_cached_sizze_73060, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71429_cached_sizze_73061 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71429, &mem_71429_cached_sizze_73061, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71430_cached_sizze_73062 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71430, &mem_71430_cached_sizze_73062, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71439_cached_sizze_73063 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71439, &mem_71439_cached_sizze_73063, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71440_cached_sizze_73064 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71440, &mem_71440_cached_sizze_73064, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71461_cached_sizze_73065 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71461, &mem_71461_cached_sizze_73065, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71462_cached_sizze_73066 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71462, &mem_71462_cached_sizze_73066, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71470_cached_sizze_73067 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71470, &mem_71470_cached_sizze_73067, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71484_cached_sizze_73068 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71484, &mem_71484_cached_sizze_73068, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71485_cached_sizze_73069 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71485, &mem_71485_cached_sizze_73069, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71493_cached_sizze_73070 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71493, &mem_71493_cached_sizze_73070, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71507_cached_sizze_73071 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71507, &mem_71507_cached_sizze_73071, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71512_cached_sizze_73072 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71512, &mem_71512_cached_sizze_73072, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71523_cached_sizze_73073 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71523, &mem_71523_cached_sizze_73073, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71528_cached_sizze_73074 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71528, &mem_71528_cached_sizze_73074, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71539_cached_sizze_73075 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_71539, &mem_71539_cached_sizze_73075, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71544_cached_sizze_73076 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71544, &mem_71544_cached_sizze_73076, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71555_cached_sizze_73077 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_71555, &mem_71555_cached_sizze_73077, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71556_cached_sizze_73078 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_71556, &mem_71556_cached_sizze_73078, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71565_cached_sizze_73079 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71565, &mem_71565_cached_sizze_73079, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71566_cached_sizze_73080 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71566, &mem_71566_cached_sizze_73080, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71579_cached_sizze_73081 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71579, &mem_71579_cached_sizze_73081, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71580_cached_sizze_73082 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71580, &mem_71580_cached_sizze_73082, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71593_cached_sizze_73083 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71593, &mem_71593_cached_sizze_73083, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71594_cached_sizze_73084 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71594, &mem_71594_cached_sizze_73084, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71615_cached_sizze_73085 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71615, &mem_71615_cached_sizze_73085, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71622_cached_sizze_73086 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_71622, &mem_71622_cached_sizze_73086, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71627_cached_sizze_73087 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_71627, &mem_71627_cached_sizze_73087, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71638_cached_sizze_73088 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71638, &mem_71638_cached_sizze_73088, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71643_cached_sizze_73089 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71643, &mem_71643_cached_sizze_73089, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71654_cached_sizze_73090 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71654, &mem_71654_cached_sizze_73090, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71655_cached_sizze_73091 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71655, &mem_71655_cached_sizze_73091, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71664_cached_sizze_73092 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71664, &mem_71664_cached_sizze_73092, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71665_cached_sizze_73093 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71665, &mem_71665_cached_sizze_73093, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71686_cached_sizze_73094 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71686, &mem_71686_cached_sizze_73094, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71691_cached_sizze_73095 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71691, &mem_71691_cached_sizze_73095, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71702_cached_sizze_73096 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71702, &mem_71702_cached_sizze_73096, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71707_cached_sizze_73097 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71707, &mem_71707_cached_sizze_73097, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71718_cached_sizze_73098 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71718, &mem_71718_cached_sizze_73098, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71725_cached_sizze_73099 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71725, &mem_71725_cached_sizze_73099, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71732_cached_sizze_73100 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71732, &mem_71732_cached_sizze_73100, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71742_cached_sizze_73101 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71742, &mem_71742_cached_sizze_73101, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71747_cached_sizze_73102 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71747, &mem_71747_cached_sizze_73102, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71758_cached_sizze_73103 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71758, &mem_71758_cached_sizze_73103, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71759_cached_sizze_73104 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71759, &mem_71759_cached_sizze_73104, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71768_cached_sizze_73105 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71768, &mem_71768_cached_sizze_73105, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71769_cached_sizze_73106 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71769, &mem_71769_cached_sizze_73106, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71790_cached_sizze_73107 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71790, &mem_71790_cached_sizze_73107, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71791_cached_sizze_73108 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71791, &mem_71791_cached_sizze_73108, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71802_cached_sizze_73109 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71802, &mem_71802_cached_sizze_73109, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71803_cached_sizze_73110 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71803, &mem_71803_cached_sizze_73110, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71812_cached_sizze_73111 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71812, &mem_71812_cached_sizze_73111, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71819_cached_sizze_73112 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71819, &mem_71819_cached_sizze_73112, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71844_cached_sizze_73113 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71844, &mem_71844_cached_sizze_73113, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71845_cached_sizze_73114 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71845, &mem_71845_cached_sizze_73114, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71856_cached_sizze_73115 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71856, &mem_71856_cached_sizze_73115, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71857_cached_sizze_73116 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71857, &mem_71857_cached_sizze_73116, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71866_cached_sizze_73117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71866, &mem_71866_cached_sizze_73117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71873_cached_sizze_73118 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71873, &mem_71873_cached_sizze_73118, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71880_cached_sizze_73119 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71880, &mem_71880_cached_sizze_73119, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71887_cached_sizze_73120 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71887, &mem_71887_cached_sizze_73120, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71912_cached_sizze_73121 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71912, &mem_71912_cached_sizze_73121, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71913_cached_sizze_73122 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71913, &mem_71913_cached_sizze_73122, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71924_cached_sizze_73123 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71924, &mem_71924_cached_sizze_73123, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71925_cached_sizze_73124 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71925, &mem_71925_cached_sizze_73124, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71934_cached_sizze_73125 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71934, &mem_71934_cached_sizze_73125, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71941_cached_sizze_73126 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_71941, &mem_71941_cached_sizze_73126, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71966_cached_sizze_73127 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_71966, &mem_71966_cached_sizze_73127, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71971_cached_sizze_73128 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71971, &mem_71971_cached_sizze_73128, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71982_cached_sizze_73129 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_71982, &mem_71982_cached_sizze_73129, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71988_cached_sizze_73130 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_71988, &mem_71988_cached_sizze_73130, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_71993_cached_sizze_73131 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_71993, &mem_71993_cached_sizze_73131, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72009_cached_sizze_73132 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_72009, &mem_72009_cached_sizze_73132, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72015_cached_sizze_73133 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72015, &mem_72015_cached_sizze_73133, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72020_cached_sizze_73134 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72020, &mem_72020_cached_sizze_73134, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72036_cached_sizze_73135 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72036, &mem_72036_cached_sizze_73135, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72037_cached_sizze_73136 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72037, &mem_72037_cached_sizze_73136, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72048_cached_sizze_73137 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_72048, &mem_72048_cached_sizze_73137, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72049_cached_sizze_73138 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_72049, &mem_72049_cached_sizze_73138, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72058_cached_sizze_73139 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_72058, &mem_72058_cached_sizze_73139, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72059_cached_sizze_73140 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_72059, &mem_72059_cached_sizze_73140, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72090_cached_sizze_73141 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72090, &mem_72090_cached_sizze_73141, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72091_cached_sizze_73142 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72091, &mem_72091_cached_sizze_73142, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72092_cached_sizze_73143 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72092, &mem_72092_cached_sizze_73143, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72105_cached_sizze_73144 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72105, &mem_72105_cached_sizze_73144, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72106_cached_sizze_73145 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72106, &mem_72106_cached_sizze_73145, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72107_cached_sizze_73146 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72107, &mem_72107_cached_sizze_73146, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72138_cached_sizze_73147 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72138, &mem_72138_cached_sizze_73147, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72139_cached_sizze_73148 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72139, &mem_72139_cached_sizze_73148, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72140_cached_sizze_73149 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72140, &mem_72140_cached_sizze_73149, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72141_cached_sizze_73150 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72141, &mem_72141_cached_sizze_73150, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72158_cached_sizze_73151 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72158, &mem_72158_cached_sizze_73151, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72159_cached_sizze_73152 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72159, &mem_72159_cached_sizze_73152, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72160_cached_sizze_73153 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72160, &mem_72160_cached_sizze_73153, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72161_cached_sizze_73154 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72161, &mem_72161_cached_sizze_73154, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72202_cached_sizze_73155 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72202, &mem_72202_cached_sizze_73155, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72209_cached_sizze_73156 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72209, &mem_72209_cached_sizze_73156, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72216_cached_sizze_73157 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72216, &mem_72216_cached_sizze_73157, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72226_cached_sizze_73158 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72226, &mem_72226_cached_sizze_73158, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72231_cached_sizze_73159 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72231, &mem_72231_cached_sizze_73159, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72242_cached_sizze_73160 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72242, &mem_72242_cached_sizze_73160, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72249_cached_sizze_73161 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72249, &mem_72249_cached_sizze_73161, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72256_cached_sizze_73162 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72256, &mem_72256_cached_sizze_73162, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72266_cached_sizze_73163 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72266, &mem_72266_cached_sizze_73163, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72271_cached_sizze_73164 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72271, &mem_72271_cached_sizze_73164, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72282_cached_sizze_73165 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72282, &mem_72282_cached_sizze_73165, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72283_cached_sizze_73166 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_72283, &mem_72283_cached_sizze_73166, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72292_cached_sizze_73167 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72292, &mem_72292_cached_sizze_73167, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72293_cached_sizze_73168 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72293, &mem_72293_cached_sizze_73168, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72314_cached_sizze_73169 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_72314, &mem_72314_cached_sizze_73169, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72319_cached_sizze_73170 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72319, &mem_72319_cached_sizze_73170, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72330_cached_sizze_73171 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_72330, &mem_72330_cached_sizze_73171, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72331_cached_sizze_73172 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_72331, &mem_72331_cached_sizze_73172, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72340_cached_sizze_73173 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72340, &mem_72340_cached_sizze_73173, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_72341_cached_sizze_73174 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_72341, &mem_72341_cached_sizze_73174, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:498:5-505:24
    if (memblock_set(ctx, &mem_param_70875, &wdown_mem_70842, "wdown_mem_70842") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70879, &wkey_mem_70843, "wkey_mem_70843") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70883, &wout_mem_70844, "wout_mem_70844") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70887, &wpe_mem_70845, "wpe_mem_70845") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70891, &wqry_mem_70846, "wqry_mem_70846") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70895, &wte_mem_70847, "wte_mem_70847") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70899, &wup_mem_70848, "wup_mem_70848") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70903, &wval_mem_70849, "wval_mem_70849") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70907, &wvoc_mem_70850, "wvoc_mem_70850") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70911, &wdown_mem_70851, "wdown_mem_70851") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70915, &wkey_mem_70852, "wkey_mem_70852") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70919, &wout_mem_70853, "wout_mem_70853") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70923, &wpe_mem_70854, "wpe_mem_70854") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70927, &wqry_mem_70855, "wqry_mem_70855") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70931, &wte_mem_70856, "wte_mem_70856") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70935, &wup_mem_70857, "wup_mem_70857") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70939, &wval_mem_70858, "wval_mem_70858") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70943, &wvoc_mem_70859, "wvoc_mem_70859") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70947, &wdown_mem_70860, "wdown_mem_70860") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70951, &wkey_mem_70861, "wkey_mem_70861") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70955, &wout_mem_70862, "wout_mem_70862") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70959, &wpe_mem_70863, "wpe_mem_70863") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70963, &wqry_mem_70864, "wqry_mem_70864") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70967, &wte_mem_70865, "wte_mem_70865") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70971, &wup_mem_70866, "wup_mem_70866") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70975, &wval_mem_70867, "wval_mem_70867") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_70979, &wvoc_mem_70868, "wvoc_mem_70868") != 0)
        return 1;
    for (int64_t step_64196 = 0; step_64196 < (int64_t) 10000; step_64196++) {
        // futhark/microgpt.fut:500:16-25
        
        int64_t dl_64224 = ((int64_t *) dls_mem_70870.mem)[step_64196];
        
        // futhark/microgpt.fut:405:37-40
        
        int64_t zl_rhs_64225 = sub64(dl_64224, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_69995 = 0; i_69995 < (int64_t) 16; i_69995++) {
            // futhark/microgpt.fut:405:25-81
            
            bool cond_66014 = slt64(i_69995, zl_rhs_64225);
            
            // futhark/microgpt.fut:405:56-59
            
            int64_t zeze_lhs_66015 = add64((int64_t) 1, i_69995);
            
            // futhark/microgpt.fut:405:47-60
            
            bool x_66016 = sle64((int64_t) 0, zeze_lhs_66015);
            
            // futhark/microgpt.fut:405:47-60
            
            bool y_66017 = slt64(zeze_lhs_66015, (int64_t) 16);
            
            // futhark/microgpt.fut:405:47-60
            
            bool bounds_check_66018 = x_66016 && y_66017;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_66019 = !cond_66014;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_66020 = bounds_check_66018 || loop_not_taken_66019;
            
            // futhark/microgpt.fut:405:47-60
            
            bool index_certs_66021;
            
            if (!protect_assert_disj_66020) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_66015, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:405:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:405:3-83\n   #6  futhark/microgpt.fut:469:18-38\n   #7  futhark/microgpt.fut:479:26-486:31\n   #8  futhark/microgpt.fut:503:29-68\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_66036 = ((int64_t *) seqs_mem_70871.mem)[step_64196 * (int64_t) 16 + i_69995];
            
            // futhark/microgpt.fut:471:37-51
            
            bool x_66037 = sle64((int64_t) 0, tmp_66036);
            
            // futhark/microgpt.fut:471:37-51
            
            bool y_66038 = slt64(tmp_66036, (int64_t) 27);
            
            // futhark/microgpt.fut:471:37-51
            
            bool bounds_check_66039 = x_66037 && y_66038;
            
            // futhark/microgpt.fut:471:37-51
            
            bool index_certs_66040;
            
            if (!bounds_check_66039) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_66036, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:471:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:471:16-55\n   #6  futhark/microgpt.fut:479:26-486:31\n   #7  futhark/microgpt.fut:503:29-68\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:405:47-60
            
            int64_t zeze_lhs_66022;
            
            if (cond_66014) {
                int64_t x_69804 = ((int64_t *) seqs_mem_70871.mem)[step_64196 * (int64_t) 16 + zeze_lhs_66015];
                
                zeze_lhs_66022 = x_69804;
            } else {
                zeze_lhs_66022 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_69985 = 0; i_69985 < (int64_t) 27; i_69985++) {
                // futhark/microgpt.fut:405:61-65
                
                bool cond_t_res_66026 = zeze_lhs_66022 == i_69985;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_66027 = cond_66014 && cond_t_res_66026;
                
                // futhark/microgpt.fut:405:25-81
                
                double lifted_lambda_res_66028;
                
                if (x_66027) {
                    lifted_lambda_res_66028 = 1.0;
                } else {
                    lifted_lambda_res_66028 = 0.0;
                }
                ((double *) mem_70990)[i_69985] = lifted_lambda_res_66028;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_69989 = 0; i_69989 < (int64_t) 16; i_69989++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_66047 = ((double *) mem_param_70895.mem)[tmp_66036 * (int64_t) 16 + i_69989];
                
                ((double *) mem_70997)[i_69989] = lifted_lambda_res_66047;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_70980, i_69995 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_70997, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_70981, i_69995 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_70990, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70010 = 0; i_70010 < (int64_t) 16; i_70010++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70000 = 0; i_70000 < (int64_t) 16; i_70000++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_66072 = ((double *) mem_param_70887.mem)[i_70010 * (int64_t) 16 + i_70000];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_66073 = ((double *) mem_70980)[i_70010 * (int64_t) 16 + i_70000];
                
                // futhark/microgpt.fut:265:35-63
                
                double zp_res_66074 = zp_lhs_66072 + zp_rhs_66073;
                
                ((double *) mem_71022)[i_70000] = zp_res_66074;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70004 = 0; i_70004 < (int64_t) 27; i_70004++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_66088 = ((double *) mem_70981)[i_70010 * (int64_t) 27 + i_70004];
                
                // futhark/microgpt.fut:297:51-87
                
                double zt_res_66089 = -6.25e-2 * zt_rhs_66088;
                
                ((double *) mem_71029)[i_70004] = zt_res_66089;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71012, i_70010 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71029, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71013, i_70010 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71022, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70024 = 0; i_70024 < (int64_t) 16; i_70024++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_66108;
            double r_66110 = 0.0;
            
            for (int64_t i_66109 = 0; i_66109 < (int64_t) 16; i_66109++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_66111 = ((double *) mem_71013)[i_70024 * (int64_t) 16 + i_66109];
                
                // futhark/microgpt.fut:266:58-83
                
                double zt_res_66112 = zt_lhs_66111 * zt_lhs_66111;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_66113 = r_66110 + zt_res_66112;
                double r_tmp_72732 = zp_res_66113;
                
                r_66110 = r_tmp_72732;
            }
            defunc_0_lifted_lambda_res_66108 = r_66110;
            // futhark/microgpt.fut:266:40-101
            
            double zs_res_66114 = defunc_0_lifted_lambda_res_66108 / 16.0;
            
            // futhark/microgpt.fut:267:23-53
            
            double zp_res_66115 = 1.0e-5 + zs_res_66114;
            
            // futhark/microgpt.fut:267:15-53
            
            double sqrt_res_66116 = futrts_sqrt64(zp_res_66115);
            
            // futhark/microgpt.fut:268:39-49
            
            double zs_res_66117 = 1.0 / sqrt_res_66116;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70017 = 0; i_70017 < (int64_t) 16; i_70017++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_68113 = ((double *) mem_71013)[i_70024 * (int64_t) 16 + i_70017];
                
                // futhark/microgpt.fut:268:23-49
                
                double zt_res_68114 = zs_res_66117 * zt_lhs_68113;
                
                // futhark/microgpt.fut:340:53-86
                
                double zt_res_68122 = zt_lhs_68113 * zt_lhs_68113;
                
                ((double *) mem_71054)[i_70017] = zt_res_68122;
                ((double *) mem_71055)[i_70017] = zt_res_68114;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71044, i_70024 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71054, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71045, i_70024 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71055, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70040 = 0; i_70040 < (int64_t) 16; i_70040++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_66216;
            double r_66218 = 0.0;
            
            for (int64_t i_66217 = 0; i_66217 < (int64_t) 16; i_66217++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_66219 = ((double *) mem_71045)[i_70040 * (int64_t) 16 + i_66217];
                
                // futhark/microgpt.fut:269:61-90
                
                double zt_res_66220 = zt_lhs_66219 * zt_lhs_66219;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_66221 = r_66218 + zt_res_66220;
                double r_tmp_72738 = zp_res_66221;
                
                r_66218 = r_tmp_72738;
            }
            defunc_0_lifted_lambda_res_66216 = r_66218;
            // futhark/microgpt.fut:269:42-108
            
            double zs_res_66222 = defunc_0_lifted_lambda_res_66216 / 16.0;
            
            // futhark/microgpt.fut:270:24-55
            
            double zp_res_66223 = 1.0e-5 + zs_res_66222;
            
            // futhark/microgpt.fut:270:16-55
            
            double sqrt_res_66224 = futrts_sqrt64(zp_res_66223);
            
            // futhark/microgpt.fut:271:42-53
            
            double zs_res_66225 = 1.0 / sqrt_res_66224;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70031 = 0; i_70031 < (int64_t) 16; i_70031++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_68142 = ((double *) mem_71045)[i_70040 * (int64_t) 16 + i_70031];
                
                // futhark/microgpt.fut:271:24-53
                
                double zt_res_68143 = zs_res_66225 * zt_lhs_68142;
                
                // futhark/microgpt.fut:333:53-86
                
                double zt_res_68151 = zt_lhs_68142 * zt_lhs_68142;
                
                ((double *) mem_71090)[i_70031] = zt_res_68151;
                ((double *) mem_71091)[i_70031] = zt_res_68143;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_66259;
            double r_66261 = 0.0;
            
            for (int64_t i_66260 = 0; i_66260 < (int64_t) 16; i_66260++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_66262 = ((double *) mem_71044)[i_70040 * (int64_t) 16 + i_66260];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_66263 = r_66261 + lifted_lambda_res_66262;
                double r_tmp_72741 = zp_res_66263;
                
                r_66261 = r_tmp_72741;
            }
            defunc_0_lifted_lambda_res_66259 = r_66261;
            // futhark/microgpt.fut:341:34-86
            
            double zs_res_66264 = defunc_0_lifted_lambda_res_66259 / 16.0;
            
            ((double *) mem_71076)[i_70040] = zs_res_66264;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71077, i_70040 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71090, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71078, i_70040 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71091, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70064 = 0; i_70064 < (int64_t) 16; i_70064++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70050 = 0; i_70050 < (int64_t) 16; i_70050++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68214;
                double r_68216 = 0.0;
                
                for (int64_t i_68215 = 0; i_68215 < (int64_t) 16; i_68215++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68217 = ((double *) mem_param_70891.mem)[i_70050 * (int64_t) 16 + i_68215];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68218 = ((double *) mem_71078)[i_70064 * (int64_t) 16 + i_68215];
                    
                    // futhark/microgpt.fut:272:69-100
                    
                    double zt_res_68219 = zt_lhs_68217 * zt_rhs_68218;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68220 = r_68216 + zt_res_68219;
                    double r_tmp_72750 = zp_res_68220;
                    
                    r_68216 = r_tmp_72750;
                }
                defunc_0_lifted_lambda_res_68214 = r_68216;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68227;
                double r_68229 = 0.0;
                
                for (int64_t i_68228 = 0; i_68228 < (int64_t) 16; i_68228++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68230 = ((double *) mem_param_70879.mem)[i_70050 * (int64_t) 16 + i_68228];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68231 = ((double *) mem_71078)[i_70064 * (int64_t) 16 + i_68228];
                    
                    // futhark/microgpt.fut:273:69-100
                    
                    double zt_res_68232 = zt_lhs_68230 * zt_rhs_68231;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68233 = r_68229 + zt_res_68232;
                    double r_tmp_72751 = zp_res_68233;
                    
                    r_68229 = r_tmp_72751;
                }
                defunc_0_lifted_lambda_res_68227 = r_68229;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68243;
                double r_68245 = 0.0;
                
                for (int64_t i_68244 = 0; i_68244 < (int64_t) 16; i_68244++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68246 = ((double *) mem_param_70903.mem)[i_70050 * (int64_t) 16 + i_68244];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68247 = ((double *) mem_71078)[i_70064 * (int64_t) 16 + i_68244];
                    
                    // futhark/microgpt.fut:274:69-100
                    
                    double zt_res_68248 = zt_lhs_68246 * zt_rhs_68247;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68249 = r_68245 + zt_res_68248;
                    double r_tmp_72752 = zp_res_68249;
                    
                    r_68245 = r_tmp_72752;
                }
                defunc_0_lifted_lambda_res_68243 = r_68245;
                ((double *) mem_71138)[i_70050] = defunc_0_lifted_lambda_res_68243;
                ((double *) mem_71139)[i_70050] = defunc_0_lifted_lambda_res_68227;
                ((double *) mem_71140)[i_70050] = defunc_0_lifted_lambda_res_68214;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_66606;
            double r_66608 = 0.0;
            
            for (int64_t i_66607 = 0; i_66607 < (int64_t) 16; i_66607++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_66609 = ((double *) mem_71077)[i_70064 * (int64_t) 16 + i_66607];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_66610 = r_66608 + lifted_lambda_res_66609;
                double r_tmp_72753 = zp_res_66610;
                
                r_66608 = r_tmp_72753;
            }
            defunc_0_lifted_lambda_res_66606 = r_66608;
            // futhark/microgpt.fut:334:34-86
            
            double zs_res_66611 = defunc_0_lifted_lambda_res_66606 / 16.0;
            
            // futhark/microgpt.fut:342:41-51
            
            double zp_lhs_66625 = ((double *) mem_71076)[i_70064];
            
            // futhark/microgpt.fut:342:41-79
            
            double zp_res_66626 = 1.0e-5 + zp_lhs_66625;
            
            // futhark/microgpt.fut:342:33-79
            
            double sqrt_res_66627 = futrts_sqrt64(zp_res_66626);
            
            ((double *) mem_71115)[i_70064] = sqrt_res_66627;
            ((double *) mem_71116)[i_70064] = zs_res_66611;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71117, i_70064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71138, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71118, i_70064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71139, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71119, i_70064 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71140, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70096 = 0; i_70096 < (int64_t) 4; i_70096++) {
            // futhark/microgpt.fut:275:81-84
            
            int64_t zp_lhs_66699 = mul64((int64_t) 4, i_70096);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70086 = 0; i_70086 < (int64_t) 16; i_70086++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70076 = 0; i_70076 < (int64_t) 4; i_70076++) {
                    // futhark/microgpt.fut:275:86-91
                    
                    int64_t tmp_68407 = add64(zp_lhs_66699, i_70076);
                    
                    // futhark/microgpt.fut:275:66-93
                    
                    bool x_68408 = sle64((int64_t) 0, tmp_68407);
                    
                    // futhark/microgpt.fut:275:66-93
                    
                    bool y_68409 = slt64(tmp_68407, (int64_t) 16);
                    
                    // futhark/microgpt.fut:275:66-93
                    
                    bool bounds_check_68410 = x_68408 && y_68409;
                    
                    // futhark/microgpt.fut:275:66-93
                    
                    bool index_certs_68411;
                    
                    if (!bounds_check_68410) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_68407, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:275:66-93\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:275:49-94\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:275:30-96\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:275:12-98\n   #10 futhark/microgpt.fut:474:5-76\n   #11 futhark/microgpt.fut:479:26-486:31\n   #12 futhark/microgpt.fut:503:29-68\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_68412 = ((double *) mem_71119)[i_70086 * (int64_t) 16 + tmp_68407];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_68420 = ((double *) mem_71118)[i_70086 * (int64_t) 16 + tmp_68407];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_68431 = ((double *) mem_71117)[i_70086 * (int64_t) 16 + tmp_68407];
                    
                    ((double *) mem_71210)[i_70076] = lifted_lambda_res_68431;
                    ((double *) mem_71211)[i_70076] = lifted_lambda_res_68420;
                    ((double *) mem_71212)[i_70076] = lifted_lambda_res_68412;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71195, i_70086 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71210, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71196, i_70086 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71211, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71197, i_70086 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71212, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71177, i_70096 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71195, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71178, i_70096 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71196, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71179, i_70096 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71197, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70152 = 0; i_70152 < (int64_t) 4; i_70152++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70111 = 0; i_70111 < (int64_t) 16; i_70111++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70104 = 0; i_70104 < (int64_t) 16; i_70104++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_68510;
                    double r_68512 = 0.0;
                    
                    for (int64_t i_68511 = 0; i_68511 < (int64_t) 4; i_68511++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_68513 = ((double *) mem_71179)[i_70152 * (int64_t) 64 + i_70111 * (int64_t) 4 + i_68511];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_68514 = ((double *) mem_71178)[i_70152 * (int64_t) 64 + i_70104 * (int64_t) 4 + i_68511];
                        
                        // futhark/microgpt.fut:278:97-138
                        
                        double zt_res_68515 = zt_lhs_68513 * zt_rhs_68514;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_68516 = r_68512 + zt_res_68515;
                        double r_tmp_72769 = zp_res_68516;
                        
                        r_68512 = r_tmp_72769;
                    }
                    defunc_0_lifted_lambda_res_68510 = r_68512;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_68523;
                    double r_68525 = 0.0;
                    
                    for (int64_t i_68524 = 0; i_68524 < (int64_t) 4; i_68524++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_68526 = ((double *) mem_71179)[i_70152 * (int64_t) 64 + i_70111 * (int64_t) 4 + i_68524];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_68527 = ((double *) mem_71178)[i_70152 * (int64_t) 64 + i_70104 * (int64_t) 4 + i_68524];
                        
                        // futhark/microgpt.fut:317:91-138
                        
                        double zt_res_68528 = zt_lhs_68526 * zt_rhs_68527;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_68529 = r_68525 + zt_res_68528;
                        double r_tmp_72770 = zp_res_68529;
                        
                        r_68525 = r_tmp_72770;
                    }
                    defunc_0_lifted_lambda_res_68523 = r_68525;
                    ((double *) mem_71280)[i_70104] = defunc_0_lifted_lambda_res_68523;
                    ((double *) mem_71281)[i_70104] = defunc_0_lifted_lambda_res_68510;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71270, i_70111 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71280, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71271, i_70111 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71281, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70120 = 0; i_70120 < (int64_t) 16; i_70120++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70116 = 0; i_70116 < (int64_t) 16; i_70116++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_66808 = ((double *) mem_71271)[i_70120 * (int64_t) 16 + i_70116];
                    
                    // futhark/microgpt.fut:279:43-70
                    
                    double zs_res_66809 = zs_lhs_66808 / 2.0;
                    double zp_rhs_66810 = ((double *) masks_mem_70869.mem)[step_64196 * (int64_t) 256 + i_70120 * (int64_t) 16 + i_70116];
                    
                    // futhark/microgpt.fut:279:57-90
                    
                    double zp_res_66811 = zs_res_66809 + zp_rhs_66810;
                    
                    ((double *) mem_71307)[i_70116] = zp_res_66811;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71302, i_70120 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71307, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70138 = 0; i_70138 < (int64_t) 16; i_70138++) {
                // futhark/microgpt.fut:103:13-33
                
                double defunc_0_reduce_res_69825;
                double redout_70122 = -INFINITY;
                
                for (int64_t i_70123 = 0; i_70123 < (int64_t) 16; i_70123++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_68547 = ((double *) mem_71302)[i_70138 * (int64_t) 16 + i_70123];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_66832 = fmax64(lifted_lambda_res_68547, redout_70122);
                    double redout_tmp_72774 = max_res_66832;
                    
                    redout_70122 = redout_tmp_72774;
                }
                defunc_0_reduce_res_69825 = redout_70122;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_66833 = -defunc_0_reduce_res_69825;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70126 = 0; i_70126 < (int64_t) 16; i_70126++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_66840 = ((double *) mem_71302)[i_70138 * (int64_t) 16 + i_70126];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_66841 = neg_res_66833 + lifted_lambda_res_66840;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_66842 = futrts_exp64(zp_res_66841);
                    
                    ((double *) mem_71323)[i_70126] = exp_res_66842;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_66844;
                double r_66846 = 0.0;
                
                for (int64_t i_66845 = 0; i_66845 < (int64_t) 16; i_66845++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_66847 = ((double *) mem_71323)[i_66845];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_66848 = r_66846 + lifted_lambda_res_66847;
                    double r_tmp_72776 = zp_res_66848;
                    
                    r_66846 = r_tmp_72776;
                }
                defunc_0_lifted_lambda_res_66844 = r_66846;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70130 = 0; i_70130 < (int64_t) 16; i_70130++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_66855 = ((double *) mem_71323)[i_70130];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_66856 = zs_lhs_66855 / defunc_0_lifted_lambda_res_66844;
                    
                    ((double *) mem_71330)[i_70130] = zs_res_66856;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70134 = 0; i_70134 < (int64_t) 16; i_70134++) {
                    // futhark/microgpt.fut:281:23-31
                    
                    double lifted_lambda_res_66864 = ((double *) mem_71330)[i_70134];
                    
                    ((double *) mem_71337)[i_70134] = lifted_lambda_res_66864;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71318, i_70138 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71337, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70146 = 0; i_70146 < (int64_t) 16; i_70146++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70142 = 0; i_70142 < (int64_t) 4; i_70142++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_66879;
                    double r_66881 = 0.0;
                    
                    for (int64_t i_66880 = 0; i_66880 < (int64_t) 16; i_66880++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_66882 = ((double *) mem_71318)[i_70146 * (int64_t) 16 + i_66880];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_66883 = ((double *) mem_71177)[i_70152 * (int64_t) 64 + i_66880 * (int64_t) 4 + i_70142];
                        
                        // futhark/microgpt.fut:282:61-97
                        
                        double zt_res_66884 = zt_lhs_66882 * zt_rhs_66883;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_66885 = r_66881 + zt_res_66884;
                        double r_tmp_72781 = zp_res_66885;
                        
                        r_66881 = r_tmp_72781;
                    }
                    defunc_0_lifted_lambda_res_66879 = r_66881;
                    ((double *) mem_71353)[i_70142] = defunc_0_lifted_lambda_res_66879;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71348, i_70146 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71258, i_70152 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71270, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71259, i_70152 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71348, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70163 = 0; i_70163 < (int64_t) 16; i_70163++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70157 = 0; i_70157 < (int64_t) 16; i_70157++) {
                // futhark/microgpt.fut:283:58-61
                
                int64_t tmp_66934 = sdiv64(i_70157, (int64_t) 4);
                
                // futhark/microgpt.fut:283:49-63
                
                bool x_66935 = sle64((int64_t) 0, tmp_66934);
                
                // futhark/microgpt.fut:283:49-63
                
                bool y_66936 = slt64(tmp_66934, (int64_t) 4);
                
                // futhark/microgpt.fut:283:49-63
                
                bool bounds_check_66937 = x_66935 && y_66936;
                
                // futhark/microgpt.fut:283:49-63
                
                bool index_certs_66938;
                
                if (!bounds_check_66937) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_66934, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:283:49-63\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:283:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:283:12-82\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:29-68\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:283:74-77
                
                int64_t tmp_66939 = smod64(i_70157, (int64_t) 4);
                
                // futhark/microgpt.fut:283:49-79
                
                bool x_66940 = sle64((int64_t) 0, tmp_66939);
                
                // futhark/microgpt.fut:283:49-79
                
                bool y_66941 = slt64(tmp_66939, (int64_t) 4);
                
                // futhark/microgpt.fut:283:49-79
                
                bool bounds_check_66942 = x_66940 && y_66941;
                
                // futhark/microgpt.fut:283:49-79
                
                bool index_certs_66943;
                
                if (!bounds_check_66942) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_66939, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:283:49-79\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:283:31-80\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:283:12-82\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:29-68\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_66944 = ((double *) mem_71259)[tmp_66934 * (int64_t) 64 + i_70163 * (int64_t) 4 + tmp_66939];
                
                ((double *) mem_71383)[i_70157] = lifted_lambda_res_66944;
            }
            // futhark/microgpt.fut:335:41-51
            
            double zp_lhs_66952 = ((double *) mem_71116)[i_70163];
            
            // futhark/microgpt.fut:335:41-79
            
            double zp_res_66953 = 1.0e-5 + zp_lhs_66952;
            
            // futhark/microgpt.fut:335:33-79
            
            double sqrt_res_66954 = futrts_sqrt64(zp_res_66953);
            
            ((double *) mem_71374)[i_70163] = sqrt_res_66954;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71375, i_70163 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71383, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70172 = 0; i_70172 < (int64_t) 16; i_70172++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70168 = 0; i_70168 < (int64_t) 16; i_70168++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_64610;
                double r_64612 = 0.0;
                
                for (int64_t i_64611 = 0; i_64611 < (int64_t) 16; i_64611++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_64613 = ((double *) mem_param_70883.mem)[i_70168 * (int64_t) 16 + i_64611];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_64614 = ((double *) mem_71375)[i_70172 * (int64_t) 16 + i_64611];
                    
                    // futhark/microgpt.fut:284:69-101
                    
                    double zt_res_64615 = zt_lhs_64613 * zt_rhs_64614;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_64616 = r_64612 + zt_res_64615;
                    double r_tmp_72787 = zp_res_64616;
                    
                    r_64612 = r_tmp_72787;
                }
                defunc_0_lifted_lambda_res_64610 = r_64612;
                ((double *) mem_71402)[i_70168] = defunc_0_lifted_lambda_res_64610;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71397, i_70172 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71402, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70180 = 0; i_70180 < (int64_t) 16; i_70180++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70176 = 0; i_70176 < (int64_t) 16; i_70176++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_64631 = ((double *) mem_71397)[i_70180 * (int64_t) 16 + i_70176];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_64632 = ((double *) mem_71045)[i_70180 * (int64_t) 16 + i_70176];
                
                // futhark/microgpt.fut:285:38-68
                
                double zp_res_64633 = zp_lhs_64631 + zp_rhs_64632;
                
                ((double *) mem_71418)[i_70176] = zp_res_64633;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71413, i_70180 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71418, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70193 = 0; i_70193 < (int64_t) 16; i_70193++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_66972;
            double r_66974 = 0.0;
            
            for (int64_t i_66973 = 0; i_66973 < (int64_t) 16; i_66973++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_66975 = ((double *) mem_71413)[i_70193 * (int64_t) 16 + i_66973];
                
                // futhark/microgpt.fut:286:62-93
                
                double zt_res_66976 = zt_lhs_66975 * zt_lhs_66975;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_66977 = r_66974 + zt_res_66976;
                double r_tmp_72792 = zp_res_66977;
                
                r_66974 = r_tmp_72792;
            }
            defunc_0_lifted_lambda_res_66972 = r_66974;
            // futhark/microgpt.fut:286:43-111
            
            double zs_res_66978 = defunc_0_lifted_lambda_res_66972 / 16.0;
            
            // futhark/microgpt.fut:287:24-55
            
            double zp_res_66979 = 1.0e-5 + zs_res_66978;
            
            // futhark/microgpt.fut:287:16-55
            
            double sqrt_res_66980 = futrts_sqrt64(zp_res_66979);
            
            // futhark/microgpt.fut:288:43-54
            
            double zs_res_66981 = 1.0 / sqrt_res_66980;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70186 = 0; i_70186 < (int64_t) 16; i_70186++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_68588 = ((double *) mem_71413)[i_70193 * (int64_t) 16 + i_70186];
                
                // futhark/microgpt.fut:288:24-54
                
                double zt_res_68589 = zs_res_66981 * zt_lhs_68588;
                
                // futhark/microgpt.fut:308:53-88
                
                double zt_res_68597 = zt_lhs_68588 * zt_lhs_68588;
                
                ((double *) mem_71439)[i_70186] = zt_res_68597;
                ((double *) mem_71440)[i_70186] = zt_res_68589;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71429, i_70193 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71439, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71430, i_70193 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71440, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70204 = 0; i_70204 < (int64_t) 16; i_70204++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70198 = 0; i_70198 < (int64_t) 64; i_70198++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_67029;
                double r_67031 = 0.0;
                
                for (int64_t i_67030 = 0; i_67030 < (int64_t) 16; i_67030++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_67032 = ((double *) mem_param_70899.mem)[i_70198 * (int64_t) 16 + i_67030];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_67033 = ((double *) mem_71430)[i_70204 * (int64_t) 16 + i_67030];
                    
                    // futhark/microgpt.fut:289:69-100
                    
                    double zt_res_67034 = zt_lhs_67032 * zt_rhs_67033;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_67035 = r_67031 + zt_res_67034;
                    double r_tmp_72798 = zp_res_67035;
                    
                    r_67031 = r_tmp_72798;
                }
                defunc_0_lifted_lambda_res_67029 = r_67031;
                ((double *) mem_71470)[i_70198] = defunc_0_lifted_lambda_res_67029;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_67043;
            double r_67045 = 0.0;
            
            for (int64_t i_67044 = 0; i_67044 < (int64_t) 16; i_67044++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_67046 = ((double *) mem_71429)[i_70204 * (int64_t) 16 + i_67044];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_67047 = r_67045 + lifted_lambda_res_67046;
                double r_tmp_72799 = zp_res_67047;
                
                r_67045 = r_tmp_72799;
            }
            defunc_0_lifted_lambda_res_67043 = r_67045;
            // futhark/microgpt.fut:309:34-86
            
            double zs_res_67048 = defunc_0_lifted_lambda_res_67043 / 16.0;
            
            ((double *) mem_71461)[i_70204] = zs_res_67048;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71462, i_70204 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71470, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70215 = 0; i_70215 < (int64_t) 16; i_70215++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70209 = 0; i_70209 < (int64_t) 64; i_70209++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_67072 = ((double *) mem_71462)[i_70215 * (int64_t) 64 + i_70209];
                
                // futhark/microgpt.fut:290:38-62
                
                double max_res_67073 = fmax64(0.0, max_arg0_67072);
                
                ((double *) mem_71493)[i_70209] = max_res_67073;
            }
            // futhark/microgpt.fut:310:41-51
            
            double zp_lhs_67081 = ((double *) mem_71461)[i_70215];
            
            // futhark/microgpt.fut:310:41-79
            
            double zp_res_67082 = 1.0e-5 + zp_lhs_67081;
            
            // futhark/microgpt.fut:310:33-79
            
            double sqrt_res_67083 = futrts_sqrt64(zp_res_67082);
            
            ((double *) mem_71484)[i_70215] = sqrt_res_67083;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71485, i_70215 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71493, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70224 = 0; i_70224 < (int64_t) 16; i_70224++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70220 = 0; i_70220 < (int64_t) 16; i_70220++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_64711;
                double r_64713 = 0.0;
                
                for (int64_t i_64712 = 0; i_64712 < (int64_t) 64; i_64712++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_64714 = ((double *) mem_param_70875.mem)[i_70220 * (int64_t) 64 + i_64712];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_64715 = ((double *) mem_71485)[i_70224 * (int64_t) 64 + i_64712];
                    
                    // futhark/microgpt.fut:291:69-102
                    
                    double zt_res_64716 = zt_lhs_64714 * zt_rhs_64715;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_64717 = r_64713 + zt_res_64716;
                    double r_tmp_72805 = zp_res_64717;
                    
                    r_64713 = r_tmp_72805;
                }
                defunc_0_lifted_lambda_res_64711 = r_64713;
                ((double *) mem_71512)[i_70220] = defunc_0_lifted_lambda_res_64711;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71507, i_70224 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71512, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70232 = 0; i_70232 < (int64_t) 16; i_70232++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70228 = 0; i_70228 < (int64_t) 16; i_70228++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_64732 = ((double *) mem_71507)[i_70232 * (int64_t) 16 + i_70228];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_64733 = ((double *) mem_71413)[i_70232 * (int64_t) 16 + i_70228];
                
                // futhark/microgpt.fut:292:38-69
                
                double zp_res_64734 = zp_lhs_64732 + zp_rhs_64733;
                
                ((double *) mem_71528)[i_70228] = zp_res_64734;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71523, i_70232 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71528, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70240 = 0; i_70240 < (int64_t) 16; i_70240++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70236 = 0; i_70236 < (int64_t) 27; i_70236++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_64749;
                double r_64751 = 0.0;
                
                for (int64_t i_64750 = 0; i_64750 < (int64_t) 16; i_64750++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_64752 = ((double *) mem_param_70907.mem)[i_70236 * (int64_t) 16 + i_64750];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_64753 = ((double *) mem_71523)[i_70240 * (int64_t) 16 + i_64750];
                    
                    // futhark/microgpt.fut:293:69-101
                    
                    double zt_res_64754 = zt_lhs_64752 * zt_rhs_64753;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_64755 = r_64751 + zt_res_64754;
                    double r_tmp_72810 = zp_res_64755;
                    
                    r_64751 = r_tmp_72810;
                }
                defunc_0_lifted_lambda_res_64749 = r_64751;
                ((double *) mem_71544)[i_70236] = defunc_0_lifted_lambda_res_64749;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71539, i_70240 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71544, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70270 = 0; i_70270 < (int64_t) 16; i_70270++) {
            // futhark/microgpt.fut:103:13-33
            
            double defunc_0_reduce_res_69845;
            double defunc_0_reduce_res_69846;
            double redout_70242;
            double redout_70243;
            
            redout_70242 = -INFINITY;
            redout_70243 = -INFINITY;
            for (int64_t i_70244 = 0; i_70244 < (int64_t) 27; i_70244++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_68665 = ((double *) mem_71539)[i_70270 * (int64_t) 27 + i_70244];
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_67113 = fmax64(lifted_lambda_res_68665, redout_70242);
                
                // futhark/microgpt.fut:103:13-33
                
                double max_res_67165 = fmax64(lifted_lambda_res_68665, redout_70243);
                double redout_tmp_72813 = max_res_67113;
                double redout_tmp_72814 = max_res_67165;
                
                redout_70242 = redout_tmp_72813;
                redout_70243 = redout_tmp_72814;
            }
            defunc_0_reduce_res_69845 = redout_70242;
            defunc_0_reduce_res_69846 = redout_70243;
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_67114 = -defunc_0_reduce_res_69845;
            
            // futhark/microgpt.fut:113:47-56
            
            double neg_res_67166 = -defunc_0_reduce_res_69846;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70249 = 0; i_70249 < (int64_t) 27; i_70249++) {
                // futhark/microgpt.fut:113:38-41
                
                double lifted_lambda_res_68704 = ((double *) mem_71539)[i_70270 * (int64_t) 27 + i_70249];
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_68705 = neg_res_67114 + lifted_lambda_res_68704;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_68706 = futrts_exp64(zp_res_68705);
                
                // futhark/microgpt.fut:113:38-56
                
                double zp_res_68714 = neg_res_67166 + lifted_lambda_res_68704;
                
                // futhark/microgpt.fut:113:31-56
                
                double exp_res_68715 = futrts_exp64(zp_res_68714);
                
                ((double *) mem_71565)[i_70249] = exp_res_68715;
                ((double *) mem_71566)[i_70249] = exp_res_68706;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_67125;
            double r_67127 = 0.0;
            
            for (int64_t i_67126 = 0; i_67126 < (int64_t) 27; i_67126++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_67128 = ((double *) mem_71566)[i_67126];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_67129 = r_67127 + lifted_lambda_res_67128;
                double r_tmp_72817 = zp_res_67129;
                
                r_67127 = r_tmp_72817;
            }
            defunc_0_lifted_lambda_res_67125 = r_67127;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_67177;
            double r_67179 = 0.0;
            
            for (int64_t i_67178 = 0; i_67178 < (int64_t) 27; i_67178++) {
                // futhark/microgpt.fut:114:32-39
                
                double lifted_lambda_res_67180 = ((double *) mem_71565)[i_67178];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_67181 = r_67179 + lifted_lambda_res_67180;
                double r_tmp_72818 = zp_res_67181;
                
                r_67179 = r_tmp_72818;
            }
            defunc_0_lifted_lambda_res_67177 = r_67179;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70256 = 0; i_70256 < (int64_t) 27; i_70256++) {
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_68733 = ((double *) mem_71566)[i_70256];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_68734 = zs_lhs_68733 / defunc_0_lifted_lambda_res_67125;
                
                // futhark/microgpt.fut:115:23-30
                
                double zs_lhs_68741 = ((double *) mem_71565)[i_70256];
                
                // futhark/microgpt.fut:115:23-40
                
                double zs_res_68742 = zs_lhs_68741 / defunc_0_lifted_lambda_res_67177;
                
                ((double *) mem_71579)[i_70256] = zs_res_68742;
                ((double *) mem_71580)[i_70256] = zs_res_68734;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70263 = 0; i_70263 < (int64_t) 27; i_70263++) {
                // futhark/microgpt.fut:299:24-34
                
                double lifted_lambda_res_68760 = ((double *) mem_71580)[i_70263];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_68767 = ((double *) mem_71012)[i_70270 * (int64_t) 27 + i_70263];
                
                // futhark/microgpt.fut:301:4-14
                
                double zs_rhs_68768 = ((double *) mem_71579)[i_70263];
                
                // futhark/microgpt.fut:300:74-301:14
                
                double zs_res_68769 = 1.0 / zs_rhs_68768;
                
                // futhark/microgpt.fut:300:53-301:14
                
                double zt_res_68770 = zt_lhs_68767 * zs_res_68769;
                
                ((double *) mem_71593)[i_70263] = zt_res_68770;
                ((double *) mem_71594)[i_70263] = lifted_lambda_res_68760;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71555, i_70270 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71593, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71556, i_70270 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71594, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70275 = 0; i_70275 < (int64_t) 16; i_70275++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_64889;
            double r_64891 = 0.0;
            
            for (int64_t i_64890 = 0; i_64890 < (int64_t) 27; i_64890++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_64892 = ((double *) mem_71555)[i_70275 * (int64_t) 27 + i_64890];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_64893 = ((double *) mem_71556)[i_70275 * (int64_t) 27 + i_64890];
                
                // futhark/microgpt.fut:302:53-90
                
                double zt_res_64894 = zt_lhs_64892 * zt_rhs_64893;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_64895 = r_64891 + zt_res_64894;
                double r_tmp_72824 = zp_res_64895;
                
                r_64891 = r_tmp_72824;
            }
            defunc_0_lifted_lambda_res_64889 = r_64891;
            ((double *) mem_71615)[i_70275] = defunc_0_lifted_lambda_res_64889;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70283 = 0; i_70283 < (int64_t) 16; i_70283++) {
            // futhark/microgpt.fut:303:103-113
            
            double neg_arg0_64903 = ((double *) mem_71615)[i_70283];
            
            // futhark/microgpt.fut:303:97-113
            
            double neg_res_64904 = -neg_arg0_64903;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70279 = 0; i_70279 < (int64_t) 27; i_70279++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_64911 = ((double *) mem_71556)[i_70283 * (int64_t) 27 + i_70279];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_64912 = ((double *) mem_71555)[i_70283 * (int64_t) 27 + i_70279];
                
                // futhark/microgpt.fut:303:75-113
                
                double zp_res_64913 = neg_res_64904 + zp_lhs_64912;
                
                // futhark/microgpt.fut:303:53-113
                
                double zt_res_64914 = zt_lhs_64911 * zp_res_64913;
                
                ((double *) mem_71627)[i_70279] = zt_res_64914;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71622, i_70283 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71627, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70291 = 0; i_70291 < (int64_t) 16; i_70291++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70287 = 0; i_70287 < (int64_t) 16; i_70287++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_64929;
                double r_64931 = 0.0;
                
                for (int64_t i_64930 = 0; i_64930 < (int64_t) 27; i_64930++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_64932 = ((double *) mem_param_70907.mem)[i_64930 * (int64_t) 16 + i_70287];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_64933 = ((double *) mem_71622)[i_70291 * (int64_t) 27 + i_64930];
                    
                    // futhark/microgpt.fut:304:73-110
                    
                    double zt_res_64934 = zt_lhs_64932 * zt_rhs_64933;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_64935 = r_64931 + zt_res_64934;
                    double r_tmp_72829 = zp_res_64935;
                    
                    r_64931 = r_tmp_72829;
                }
                defunc_0_lifted_lambda_res_64929 = r_64931;
                ((double *) mem_71643)[i_70287] = defunc_0_lifted_lambda_res_64929;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71638, i_70291 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70304 = 0; i_70304 < (int64_t) 16; i_70304++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70297 = 0; i_70297 < (int64_t) 64; i_70297++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68798;
                double r_68800 = 0.0;
                
                for (int64_t i_68799 = 0; i_68799 < (int64_t) 16; i_68799++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68801 = ((double *) mem_param_70875.mem)[i_68799 * (int64_t) 64 + i_70297];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68802 = ((double *) mem_71638)[i_70304 * (int64_t) 16 + i_68799];
                    
                    // futhark/microgpt.fut:305:73-111
                    
                    double zt_res_68803 = zt_lhs_68801 * zt_rhs_68802;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68804 = r_68800 + zt_res_68803;
                    double r_tmp_72834 = zp_res_68804;
                    
                    r_68800 = r_tmp_72834;
                }
                defunc_0_lifted_lambda_res_68798 = r_68800;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68811;
                double r_68813 = 0.0;
                
                for (int64_t i_68812 = 0; i_68812 < (int64_t) 16; i_68812++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68814 = ((double *) mem_71638)[i_68812 * (int64_t) 16 + i_70304];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68815 = ((double *) mem_71485)[i_68812 * (int64_t) 64 + i_70297];
                    
                    // futhark/microgpt.fut:355:75-111
                    
                    double zt_res_68816 = zt_lhs_68814 * zt_rhs_68815;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68817 = r_68813 + zt_res_68816;
                    double r_tmp_72835 = zp_res_68817;
                    
                    r_68813 = r_tmp_72835;
                }
                defunc_0_lifted_lambda_res_68811 = r_68813;
                ((double *) mem_71664)[i_70297] = defunc_0_lifted_lambda_res_68811;
                ((double *) mem_71665)[i_70297] = defunc_0_lifted_lambda_res_68798;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71654, i_70304 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71664, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71655, i_70304 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71665, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70313 = 0; i_70313 < (int64_t) 16; i_70313++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70309 = 0; i_70309 < (int64_t) 64; i_70309++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_64971 = ((double *) mem_71462)[i_70313 * (int64_t) 64 + i_70309];
                
                // futhark/microgpt.fut:125:42-54
                
                double max_res_64972 = fmax64(0.0, indicatorp_arg0_64971);
                
                // futhark/microgpt.fut:125:35-54
                
                double sgn_res_64973 = fsignum64(max_res_64972);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_64974 = ((double *) mem_71655)[i_70313 * (int64_t) 64 + i_70309];
                
                // futhark/microgpt.fut:306:42-90
                
                double zt_res_64975 = sgn_res_64973 * zt_rhs_64974;
                
                ((double *) mem_71691)[i_70309] = zt_res_64975;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71686, i_70313 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71691, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70321 = 0; i_70321 < (int64_t) 16; i_70321++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70317 = 0; i_70317 < (int64_t) 16; i_70317++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_64990;
                double r_64992 = 0.0;
                
                for (int64_t i_64991 = 0; i_64991 < (int64_t) 64; i_64991++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_64993 = ((double *) mem_param_70899.mem)[i_64991 * (int64_t) 16 + i_70317];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_64994 = ((double *) mem_71686)[i_70321 * (int64_t) 64 + i_64991];
                    
                    // futhark/microgpt.fut:307:73-109
                    
                    double zt_res_64995 = zt_lhs_64993 * zt_rhs_64994;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_64996 = r_64992 + zt_res_64995;
                    double r_tmp_72840 = zp_res_64996;
                    
                    r_64992 = r_tmp_72840;
                }
                defunc_0_lifted_lambda_res_64990 = r_64992;
                ((double *) mem_71707)[i_70317] = defunc_0_lifted_lambda_res_64990;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71702, i_70321 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71707, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70325 = 0; i_70325 < (int64_t) 16; i_70325++) {
            // futhark/microgpt.fut:311:49-59
            
            double zs_rhs_65044 = ((double *) mem_71484)[i_70325];
            
            // futhark/microgpt.fut:311:41-59
            
            double zs_res_65045 = 1.0 / zs_rhs_65044;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_65046;
            double r_65048 = 0.0;
            
            for (int64_t i_65047 = 0; i_65047 < (int64_t) 16; i_65047++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_65049 = ((double *) mem_71413)[i_70325 * (int64_t) 16 + i_65047];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_65050 = ((double *) mem_71702)[i_70325 * (int64_t) 16 + i_65047];
                
                // futhark/microgpt.fut:311:87-123
                
                double zt_res_65051 = zt_lhs_65049 * zt_rhs_65050;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_65052 = r_65048 + zt_res_65051;
                double r_tmp_72842 = zp_res_65052;
                
                r_65048 = r_tmp_72842;
            }
            defunc_0_lifted_lambda_res_65046 = r_65048;
            // futhark/microgpt.fut:311:67-150
            
            double zt_res_65053 = zs_res_65045 * defunc_0_lifted_lambda_res_65046;
            
            // futhark/microgpt.fut:311:45-150
            
            double zt_res_65054 = zs_res_65045 * zt_res_65053;
            
            // futhark/microgpt.fut:311:33-150
            
            double neg_res_65055 = -zt_res_65054;
            
            ((double *) mem_71718)[i_70325] = neg_res_65055;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70329 = 0; i_70329 < (int64_t) 16; i_70329++) {
            // futhark/microgpt.fut:312:33-43
            
            double zt_lhs_65063 = ((double *) mem_71718)[i_70329];
            
            // futhark/microgpt.fut:312:85-95
            
            double zp_lhs_65064 = ((double *) mem_71461)[i_70329];
            
            // futhark/microgpt.fut:312:85-123
            
            double zp_res_65065 = 1.0e-5 + zp_lhs_65064;
            
            // futhark/microgpt.fut:312:77-123
            
            double sqrt_res_65066 = futrts_sqrt64(zp_res_65065);
            
            // futhark/microgpt.fut:312:63-125
            
            double zt_res_65067 = 2.0 * sqrt_res_65066;
            
            // futhark/microgpt.fut:312:49-125
            
            double zs_res_65068 = 1.0 / zt_res_65067;
            
            // futhark/microgpt.fut:312:33-125
            
            double zt_res_65069 = zt_lhs_65063 * zs_res_65068;
            
            ((double *) mem_71725)[i_70329] = zt_res_65069;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70333 = 0; i_70333 < (int64_t) 16; i_70333++) {
            // futhark/microgpt.fut:313:53-63
            
            double zs_lhs_65077 = ((double *) mem_71725)[i_70333];
            
            // futhark/microgpt.fut:313:53-78
            
            double zs_res_65078 = zs_lhs_65077 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_72845 = 0; nest_i_72845 < (int64_t) 16; nest_i_72845++) {
                ((double *) mem_71732)[i_70333 * (int64_t) 16 + nest_i_72845] = zs_res_65078;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70341 = 0; i_70341 < (int64_t) 16; i_70341++) {
            // futhark/microgpt.fut:314:107-117
            
            double zs_rhs_65087 = ((double *) mem_71484)[i_70341];
            
            // futhark/microgpt.fut:314:99-117
            
            double zs_res_65088 = 1.0 / zs_rhs_65087;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70337 = 0; i_70337 < (int64_t) 16; i_70337++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_65095 = ((double *) mem_71638)[i_70341 * (int64_t) 16 + i_70337];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65096 = ((double *) mem_71702)[i_70341 * (int64_t) 16 + i_70337];
                
                // futhark/microgpt.fut:314:77-117
                
                double zt_res_65097 = zs_res_65088 * zt_lhs_65096;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65098 = ((double *) mem_71413)[i_70341 * (int64_t) 16 + i_70337];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_65099 = ((double *) mem_71732)[i_70341 * (int64_t) 16 + i_70337];
                
                // futhark/microgpt.fut:314:125-161
                
                double zt_res_65100 = zt_lhs_65098 * zt_rhs_65099;
                
                // futhark/microgpt.fut:314:94-161
                
                double zp_res_65101 = zt_res_65097 + zt_res_65100;
                
                // futhark/microgpt.fut:314:120-205
                
                double zp_res_65102 = zt_res_65100 + zp_res_65101;
                
                // futhark/microgpt.fut:314:53-205
                
                double zp_res_65103 = zp_lhs_65095 + zp_res_65102;
                
                ((double *) mem_71747)[i_70337] = zp_res_65103;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71742, i_70341 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71747, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70354 = 0; i_70354 < (int64_t) 16; i_70354++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70347 = 0; i_70347 < (int64_t) 16; i_70347++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68840;
                double r_68842 = 0.0;
                
                for (int64_t i_68841 = 0; i_68841 < (int64_t) 16; i_68841++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68843 = ((double *) mem_param_70883.mem)[i_68841 * (int64_t) 16 + i_70347];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68844 = ((double *) mem_71742)[i_70354 * (int64_t) 16 + i_68841];
                    
                    // futhark/microgpt.fut:315:73-110
                    
                    double zt_res_68845 = zt_lhs_68843 * zt_rhs_68844;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68846 = r_68842 + zt_res_68845;
                    double r_tmp_72852 = zp_res_68846;
                    
                    r_68842 = r_tmp_72852;
                }
                defunc_0_lifted_lambda_res_68840 = r_68842;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68853;
                double r_68855 = 0.0;
                
                for (int64_t i_68854 = 0; i_68854 < (int64_t) 16; i_68854++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_68856 = ((double *) mem_71742)[i_68854 * (int64_t) 16 + i_70354];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_68857 = ((double *) mem_71375)[i_68854 * (int64_t) 16 + i_70347];
                    
                    // futhark/microgpt.fut:353:74-110
                    
                    double zt_res_68858 = zt_lhs_68856 * zt_rhs_68857;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68859 = r_68855 + zt_res_68858;
                    double r_tmp_72853 = zp_res_68859;
                    
                    r_68855 = r_tmp_72853;
                }
                defunc_0_lifted_lambda_res_68853 = r_68855;
                ((double *) mem_71768)[i_70347] = defunc_0_lifted_lambda_res_68853;
                ((double *) mem_71769)[i_70347] = defunc_0_lifted_lambda_res_68840;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71758, i_70354 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71768, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71759, i_70354 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71769, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70376 = 0; i_70376 < (int64_t) 4; i_70376++) {
            // futhark/microgpt.fut:316:88-91
            
            int64_t zp_lhs_67317 = mul64((int64_t) 4, i_70376);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70369 = 0; i_70369 < (int64_t) 16; i_70369++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70359 = 0; i_70359 < (int64_t) 4; i_70359++) {
                    // futhark/microgpt.fut:316:93-99
                    
                    int64_t tmp_68881 = add64(zp_lhs_67317, i_70359);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool x_68882 = sle64((int64_t) 0, tmp_68881);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool y_68883 = slt64(tmp_68881, (int64_t) 16);
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool bounds_check_68884 = x_68882 && y_68883;
                    
                    // futhark/microgpt.fut:316:70-101
                    
                    bool index_certs_68885;
                    
                    if (!bounds_check_68884) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_68881, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:316:70-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:316:52-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:316:32-104\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:6:13-17\n   #9  futhark/microgpt.fut:316:13-106\n   #10 futhark/microgpt.fut:474:5-76\n   #11 futhark/microgpt.fut:479:26-486:31\n   #12 futhark/microgpt.fut:503:29-68\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_68886 = ((double *) mem_71759)[i_70369 * (int64_t) 16 + tmp_68881];
                    
                    ((double *) mem_71812)[i_70359] = lifted_lambda_res_68886;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70363 = 0; i_70363 < (int64_t) 16; i_70363++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_68900 = ((double *) mem_71258)[i_70376 * (int64_t) 256 + i_70369 * (int64_t) 16 + i_70363];
                    
                    // futhark/microgpt.fut:318:61-97
                    
                    double zs_res_68901 = zs_lhs_68900 / 2.0;
                    double zp_rhs_68902 = ((double *) masks_mem_70869.mem)[step_64196 * (int64_t) 256 + i_70369 * (int64_t) 16 + i_70363];
                    
                    // futhark/microgpt.fut:318:84-119
                    
                    double zp_res_68903 = zs_res_68901 + zp_rhs_68902;
                    
                    ((double *) mem_71819)[i_70363] = zp_res_68903;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71802, i_70369 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71819, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71803, i_70369 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71812, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71790, i_70376 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71802, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71791, i_70376 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71803, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70407 = 0; i_70407 < (int64_t) 4; i_70407++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70400 = 0; i_70400 < (int64_t) 16; i_70400++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_69866;
                double redout_70380 = -INFINITY;
                
                for (int64_t i_70382 = 0; i_70382 < (int64_t) 16; i_70382++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_69029 = ((double *) mem_71790)[i_70407 * (int64_t) 256 + i_70400 * (int64_t) 16 + i_70382];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_69040;
                    double r_69042 = 0.0;
                    
                    for (int64_t i_69041 = 0; i_69041 < (int64_t) 4; i_69041++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_69043 = ((double *) mem_71791)[i_70407 * (int64_t) 64 + i_70400 * (int64_t) 4 + i_69041];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_69044 = ((double *) mem_71177)[i_70407 * (int64_t) 64 + i_70382 * (int64_t) 4 + i_69041];
                        
                        // futhark/microgpt.fut:321:91-139
                        
                        double zt_res_69045 = zt_lhs_69043 * zt_rhs_69044;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_69046 = r_69042 + zt_res_69045;
                        double r_tmp_72866 = zp_res_69046;
                        
                        r_69042 = r_tmp_72866;
                    }
                    defunc_0_lifted_lambda_res_69040 = r_69042;
                    // futhark/microgpt.fut:103:13-33
                    
                    double max_res_68940 = fmax64(lifted_lambda_res_69029, redout_70380);
                    
                    ((double *) mem_71866)[i_70382] = defunc_0_lifted_lambda_res_69040;
                    
                    double redout_tmp_72864 = max_res_68940;
                    
                    redout_70380 = redout_tmp_72864;
                }
                defunc_0_reduce_res_69866 = redout_70380;
                // futhark/microgpt.fut:113:47-56
                
                double neg_res_68941 = -defunc_0_reduce_res_69866;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70386 = 0; i_70386 < (int64_t) 16; i_70386++) {
                    // futhark/microgpt.fut:113:38-41
                    
                    double lifted_lambda_res_68948 = ((double *) mem_71790)[i_70407 * (int64_t) 256 + i_70400 * (int64_t) 16 + i_70386];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    double zp_res_68949 = neg_res_68941 + lifted_lambda_res_68948;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    double exp_res_68950 = futrts_exp64(zp_res_68949);
                    
                    ((double *) mem_71873)[i_70386] = exp_res_68950;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_68952;
                double r_68954 = 0.0;
                
                for (int64_t i_68953 = 0; i_68953 < (int64_t) 16; i_68953++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    double lifted_lambda_res_68955 = ((double *) mem_71873)[i_68953];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_68956 = r_68954 + lifted_lambda_res_68955;
                    double r_tmp_72868 = zp_res_68956;
                    
                    r_68954 = r_tmp_72868;
                }
                defunc_0_lifted_lambda_res_68952 = r_68954;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70390 = 0; i_70390 < (int64_t) 16; i_70390++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    double zs_lhs_68963 = ((double *) mem_71873)[i_70390];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    double zs_res_68964 = zs_lhs_68963 / defunc_0_lifted_lambda_res_68952;
                    
                    ((double *) mem_71880)[i_70390] = zs_res_68964;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70394 = 0; i_70394 < (int64_t) 16; i_70394++) {
                    // futhark/microgpt.fut:320:24-34
                    
                    double lifted_lambda_res_68972 = ((double *) mem_71880)[i_70394];
                    
                    ((double *) mem_71887)[i_70394] = lifted_lambda_res_68972;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71856, i_70400 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71866, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71857, i_70400 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71887, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71844, i_70407 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71856, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71845, i_70407 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71857, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70429 = 0; i_70429 < (int64_t) 4; i_70429++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70422 = 0; i_70422 < (int64_t) 16; i_70422++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70412 = 0; i_70412 < (int64_t) 16; i_70412++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_69082 = ((double *) mem_71844)[i_70429 * (int64_t) 256 + i_70422 * (int64_t) 16 + i_70412];
                    
                    ((double *) mem_71934)[i_70412] = lifted_lambda_res_69082;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70416 = 0; i_70416 < (int64_t) 4; i_70416++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_69096;
                    double r_69098 = 0.0;
                    
                    for (int64_t i_69097 = 0; i_69097 < (int64_t) 16; i_69097++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_69099 = ((double *) mem_71845)[i_70429 * (int64_t) 256 + i_69097 * (int64_t) 16 + i_70422];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_69100 = ((double *) mem_71791)[i_70429 * (int64_t) 64 + i_69097 * (int64_t) 4 + i_70416];
                        
                        // futhark/microgpt.fut:326:91-140
                        
                        double zt_res_69101 = zt_lhs_69099 * zt_rhs_69100;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_69102 = r_69098 + zt_res_69101;
                        double r_tmp_72877 = zp_res_69102;
                        
                        r_69098 = r_tmp_72877;
                    }
                    defunc_0_lifted_lambda_res_69096 = r_69098;
                    ((double *) mem_71941)[i_70416] = defunc_0_lifted_lambda_res_69096;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71924, i_70422 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71941, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71925, i_70422 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71934, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71912, i_70429 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_71924, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71913, i_70429 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71925, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70438 = 0; i_70438 < (int64_t) 4; i_70438++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70434 = 0; i_70434 < (int64_t) 16; i_70434++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_65322;
                double r_65324 = 0.0;
                
                for (int64_t i_65323 = 0; i_65323 < (int64_t) 16; i_65323++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_65325 = ((double *) mem_71913)[i_70438 * (int64_t) 256 + i_70434 * (int64_t) 16 + i_65323];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_65326 = ((double *) mem_71845)[i_70438 * (int64_t) 256 + i_70434 * (int64_t) 16 + i_65323];
                    
                    // futhark/microgpt.fut:323:72-121
                    
                    double zt_res_65327 = zt_lhs_65325 * zt_rhs_65326;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_65328 = r_65324 + zt_res_65327;
                    double r_tmp_72880 = zp_res_65328;
                    
                    r_65324 = r_tmp_72880;
                }
                defunc_0_lifted_lambda_res_65322 = r_65324;
                ((double *) mem_71971)[i_70434] = defunc_0_lifted_lambda_res_65322;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_71966, i_70438 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71971, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70450 = 0; i_70450 < (int64_t) 4; i_70450++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70446 = 0; i_70446 < (int64_t) 16; i_70446++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_65343 = ((double *) mem_71966)[i_70450 * (int64_t) 16 + i_70446];
                
                // futhark/microgpt.fut:324:128-150
                
                double neg_res_65344 = -neg_arg0_65343;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70442 = 0; i_70442 < (int64_t) 16; i_70442++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_65351 = ((double *) mem_71845)[i_70450 * (int64_t) 256 + i_70446 * (int64_t) 16 + i_70442];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_65352 = ((double *) mem_71913)[i_70450 * (int64_t) 256 + i_70446 * (int64_t) 16 + i_70442];
                    
                    // futhark/microgpt.fut:324:100-150
                    
                    double zp_res_65353 = neg_res_65344 + zp_lhs_65352;
                    
                    // futhark/microgpt.fut:324:72-150
                    
                    double zt_res_65354 = zt_lhs_65351 * zp_res_65353;
                    
                    ((double *) mem_71993)[i_70442] = zt_res_65354;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_71988, i_70446 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_71993, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_71982, i_70450 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71988, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70462 = 0; i_70462 < (int64_t) 4; i_70462++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70458 = 0; i_70458 < (int64_t) 16; i_70458++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70454 = 0; i_70454 < (int64_t) 16; i_70454++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_65376 = ((double *) mem_71982)[i_70462 * (int64_t) 256 + i_70458 * (int64_t) 16 + i_70454];
                    
                    // futhark/microgpt.fut:325:60-96
                    
                    double zs_res_65377 = zs_lhs_65376 / 2.0;
                    
                    ((double *) mem_72020)[i_70454] = zs_res_65377;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_72015, i_70458 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72020, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_72009, i_70462 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72015, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70482 = 0; i_70482 < (int64_t) 4; i_70482++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70475 = 0; i_70475 < (int64_t) 16; i_70475++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_70468 = 0; i_70468 < (int64_t) 4; i_70468++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_69183;
                    double r_69185 = 0.0;
                    
                    for (int64_t i_69184 = 0; i_69184 < (int64_t) 16; i_69184++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_69186 = ((double *) mem_71179)[i_70482 * (int64_t) 64 + i_69184 * (int64_t) 4 + i_70468];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_69187 = ((double *) mem_72009)[i_70482 * (int64_t) 256 + i_69184 * (int64_t) 16 + i_70475];
                        
                        // futhark/microgpt.fut:327:91-139
                        
                        double zt_res_69188 = zt_lhs_69186 * zt_rhs_69187;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_69189 = r_69185 + zt_res_69188;
                        double r_tmp_72893 = zp_res_69189;
                        
                        r_69185 = r_tmp_72893;
                    }
                    defunc_0_lifted_lambda_res_69183 = r_69185;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_69196;
                    double r_69198 = 0.0;
                    
                    for (int64_t i_69197 = 0; i_69197 < (int64_t) 16; i_69197++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_69199 = ((double *) mem_72009)[i_70482 * (int64_t) 256 + i_70475 * (int64_t) 16 + i_69197];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_69200 = ((double *) mem_71178)[i_70482 * (int64_t) 64 + i_69197 * (int64_t) 4 + i_70468];
                        
                        // futhark/microgpt.fut:328:91-139
                        
                        double zt_res_69201 = zt_lhs_69199 * zt_rhs_69200;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_69202 = r_69198 + zt_res_69201;
                        double r_tmp_72894 = zp_res_69202;
                        
                        r_69198 = r_tmp_72894;
                    }
                    defunc_0_lifted_lambda_res_69196 = r_69198;
                    ((double *) mem_72058)[i_70468] = defunc_0_lifted_lambda_res_69196;
                    ((double *) mem_72059)[i_70468] = defunc_0_lifted_lambda_res_69183;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_72048, i_70475 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72058, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_72049, i_70475 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72059, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_72036, i_70482 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_72048, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_72037, i_70482 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_72049, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70501 = 0; i_70501 < (int64_t) 16; i_70501++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70491 = 0; i_70491 < (int64_t) 16; i_70491++) {
                // futhark/microgpt.fut:329:63-66
                
                int64_t tmp_69265 = sdiv64(i_70491, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-68
                
                bool x_69266 = sle64((int64_t) 0, tmp_69265);
                
                // futhark/microgpt.fut:329:52-68
                
                bool y_69267 = slt64(tmp_69265, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-68
                
                bool bounds_check_69268 = x_69266 && y_69267;
                
                // futhark/microgpt.fut:329:52-68
                
                bool index_certs_69269;
                
                if (!bounds_check_69268) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_69265, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:329:52-68\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:329:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:329:13-89\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:29-68\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:329:81-84
                
                int64_t tmp_69270 = smod64(i_70491, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-86
                
                bool x_69271 = sle64((int64_t) 0, tmp_69270);
                
                // futhark/microgpt.fut:329:52-86
                
                bool y_69272 = slt64(tmp_69270, (int64_t) 4);
                
                // futhark/microgpt.fut:329:52-86
                
                bool bounds_check_69273 = x_69271 && y_69272;
                
                // futhark/microgpt.fut:329:52-86
                
                bool index_certs_69274;
                
                if (!bounds_check_69273) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_69270, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:329:52-86\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:329:33-87\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:329:13-89\n   #7  futhark/microgpt.fut:474:5-76\n   #8  futhark/microgpt.fut:479:26-486:31\n   #9  futhark/microgpt.fut:503:29-68\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_69275 = ((double *) mem_71912)[tmp_69265 * (int64_t) 64 + i_70501 * (int64_t) 4 + tmp_69270];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_69288 = ((double *) mem_72037)[tmp_69265 * (int64_t) 64 + i_70501 * (int64_t) 4 + tmp_69270];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_69304 = ((double *) mem_72036)[tmp_69265 * (int64_t) 64 + i_70501 * (int64_t) 4 + tmp_69270];
                
                ((double *) mem_72105)[i_70491] = lifted_lambda_res_69304;
                ((double *) mem_72106)[i_70491] = lifted_lambda_res_69288;
                ((double *) mem_72107)[i_70491] = lifted_lambda_res_69275;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72090, i_70501 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72105, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72091, i_70501 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72106, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72092, i_70501 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72107, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70526 = 0; i_70526 < (int64_t) 16; i_70526++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70513 = 0; i_70513 < (int64_t) 16; i_70513++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69467;
                double r_69469 = 0.0;
                
                for (int64_t i_69468 = 0; i_69468 < (int64_t) 16; i_69468++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69470 = ((double *) mem_param_70903.mem)[i_69468 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69471 = ((double *) mem_72092)[i_70526 * (int64_t) 16 + i_69468];
                    
                    // futhark/microgpt.fut:332:75-112
                    
                    double zt_res_69472 = zt_lhs_69470 * zt_rhs_69471;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69473 = r_69469 + zt_res_69472;
                    double r_tmp_72909 = zp_res_69473;
                    
                    r_69469 = r_tmp_72909;
                }
                defunc_0_lifted_lambda_res_69467 = r_69469;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69474;
                double r_69476 = 0.0;
                
                for (int64_t i_69475 = 0; i_69475 < (int64_t) 16; i_69475++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69477 = ((double *) mem_param_70879.mem)[i_69475 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69478 = ((double *) mem_72091)[i_70526 * (int64_t) 16 + i_69475];
                    
                    // futhark/microgpt.fut:332:141-178
                    
                    double zt_res_69479 = zt_lhs_69477 * zt_rhs_69478;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69480 = r_69476 + zt_res_69479;
                    double r_tmp_72910 = zp_res_69480;
                    
                    r_69476 = r_tmp_72910;
                }
                defunc_0_lifted_lambda_res_69474 = r_69476;
                // futhark/microgpt.fut:332:55-180
                
                double zp_res_69481 = defunc_0_lifted_lambda_res_69467 + defunc_0_lifted_lambda_res_69474;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69482;
                double r_69484 = 0.0;
                
                for (int64_t i_69483 = 0; i_69483 < (int64_t) 16; i_69483++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69485 = ((double *) mem_param_70891.mem)[i_69483 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69486 = ((double *) mem_72090)[i_70526 * (int64_t) 16 + i_69483];
                    
                    // futhark/microgpt.fut:332:208-245
                    
                    double zt_res_69487 = zt_lhs_69485 * zt_rhs_69486;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69488 = r_69484 + zt_res_69487;
                    double r_tmp_72911 = zp_res_69488;
                    
                    r_69484 = r_tmp_72911;
                }
                defunc_0_lifted_lambda_res_69482 = r_69484;
                // futhark/microgpt.fut:332:116-247
                
                double zp_res_69489 = zp_res_69481 + defunc_0_lifted_lambda_res_69482;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69496;
                double r_69498 = 0.0;
                
                for (int64_t i_69497 = 0; i_69497 < (int64_t) 16; i_69497++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69499 = ((double *) mem_72090)[i_69497 * (int64_t) 16 + i_70526];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69500 = ((double *) mem_71078)[i_69497 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:350:74-109
                    
                    double zt_res_69501 = zt_lhs_69499 * zt_rhs_69500;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69502 = r_69498 + zt_res_69501;
                    double r_tmp_72912 = zp_res_69502;
                    
                    r_69498 = r_tmp_72912;
                }
                defunc_0_lifted_lambda_res_69496 = r_69498;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69512;
                double r_69514 = 0.0;
                
                for (int64_t i_69513 = 0; i_69513 < (int64_t) 16; i_69513++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69515 = ((double *) mem_72091)[i_69513 * (int64_t) 16 + i_70526];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69516 = ((double *) mem_71078)[i_69513 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:351:74-109
                    
                    double zt_res_69517 = zt_lhs_69515 * zt_rhs_69516;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69518 = r_69514 + zt_res_69517;
                    double r_tmp_72913 = zp_res_69518;
                    
                    r_69514 = r_tmp_72913;
                }
                defunc_0_lifted_lambda_res_69512 = r_69514;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69530;
                double r_69532 = 0.0;
                
                for (int64_t i_69531 = 0; i_69531 < (int64_t) 16; i_69531++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69533 = ((double *) mem_72092)[i_69531 * (int64_t) 16 + i_70526];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69534 = ((double *) mem_71078)[i_69531 * (int64_t) 16 + i_70513];
                    
                    // futhark/microgpt.fut:352:74-109
                    
                    double zt_res_69535 = zt_lhs_69533 * zt_rhs_69534;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69536 = r_69532 + zt_res_69535;
                    double r_tmp_72914 = zp_res_69536;
                    
                    r_69532 = r_tmp_72914;
                }
                defunc_0_lifted_lambda_res_69530 = r_69532;
                ((double *) mem_72158)[i_70513] = defunc_0_lifted_lambda_res_69530;
                ((double *) mem_72159)[i_70513] = defunc_0_lifted_lambda_res_69512;
                ((double *) mem_72160)[i_70513] = defunc_0_lifted_lambda_res_69496;
                ((double *) mem_72161)[i_70513] = zp_res_69489;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72138, i_70526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72158, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72139, i_70526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72159, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72140, i_70526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72160, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72141, i_70526 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72161, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70533 = 0; i_70533 < (int64_t) 16; i_70533++) {
            // futhark/microgpt.fut:336:49-59
            
            double zs_rhs_65610 = ((double *) mem_71374)[i_70533];
            
            // futhark/microgpt.fut:336:41-59
            
            double zs_res_65611 = 1.0 / zs_rhs_65610;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_65612;
            double r_65614 = 0.0;
            
            for (int64_t i_65613 = 0; i_65613 < (int64_t) 16; i_65613++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_65615 = ((double *) mem_71045)[i_70533 * (int64_t) 16 + i_65613];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_65616 = ((double *) mem_72141)[i_70533 * (int64_t) 16 + i_65613];
                
                // futhark/microgpt.fut:336:87-122
                
                double zt_res_65617 = zt_lhs_65615 * zt_rhs_65616;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_65618 = r_65614 + zt_res_65617;
                double r_tmp_72916 = zp_res_65618;
                
                r_65614 = r_tmp_72916;
            }
            defunc_0_lifted_lambda_res_65612 = r_65614;
            // futhark/microgpt.fut:336:67-149
            
            double zt_res_65619 = zs_res_65611 * defunc_0_lifted_lambda_res_65612;
            
            // futhark/microgpt.fut:336:45-149
            
            double zt_res_65620 = zs_res_65611 * zt_res_65619;
            
            // futhark/microgpt.fut:336:33-149
            
            double neg_res_65621 = -zt_res_65620;
            
            ((double *) mem_72202)[i_70533] = neg_res_65621;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70537 = 0; i_70537 < (int64_t) 16; i_70537++) {
            // futhark/microgpt.fut:337:33-43
            
            double zt_lhs_65629 = ((double *) mem_72202)[i_70537];
            
            // futhark/microgpt.fut:337:85-95
            
            double zp_lhs_65630 = ((double *) mem_71116)[i_70537];
            
            // futhark/microgpt.fut:337:85-123
            
            double zp_res_65631 = 1.0e-5 + zp_lhs_65630;
            
            // futhark/microgpt.fut:337:77-123
            
            double sqrt_res_65632 = futrts_sqrt64(zp_res_65631);
            
            // futhark/microgpt.fut:337:63-125
            
            double zt_res_65633 = 2.0 * sqrt_res_65632;
            
            // futhark/microgpt.fut:337:49-125
            
            double zs_res_65634 = 1.0 / zt_res_65633;
            
            // futhark/microgpt.fut:337:33-125
            
            double zt_res_65635 = zt_lhs_65629 * zs_res_65634;
            
            ((double *) mem_72209)[i_70537] = zt_res_65635;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70541 = 0; i_70541 < (int64_t) 16; i_70541++) {
            // futhark/microgpt.fut:338:53-63
            
            double zs_lhs_65643 = ((double *) mem_72209)[i_70541];
            
            // futhark/microgpt.fut:338:53-78
            
            double zs_res_65644 = zs_lhs_65643 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_72919 = 0; nest_i_72919 < (int64_t) 16; nest_i_72919++) {
                ((double *) mem_72216)[i_70541 * (int64_t) 16 + nest_i_72919] = zs_res_65644;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70549 = 0; i_70549 < (int64_t) 16; i_70549++) {
            // futhark/microgpt.fut:339:107-117
            
            double zs_rhs_65653 = ((double *) mem_71374)[i_70549];
            
            // futhark/microgpt.fut:339:99-117
            
            double zs_res_65654 = 1.0 / zs_rhs_65653;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70545 = 0; i_70545 < (int64_t) 16; i_70545++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_65661 = ((double *) mem_71742)[i_70549 * (int64_t) 16 + i_70545];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65662 = ((double *) mem_72141)[i_70549 * (int64_t) 16 + i_70545];
                
                // futhark/microgpt.fut:339:77-117
                
                double zt_res_65663 = zs_res_65654 * zt_lhs_65662;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65664 = ((double *) mem_71045)[i_70549 * (int64_t) 16 + i_70545];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_65665 = ((double *) mem_72216)[i_70549 * (int64_t) 16 + i_70545];
                
                // futhark/microgpt.fut:339:125-160
                
                double zt_res_65666 = zt_lhs_65664 * zt_rhs_65665;
                
                // futhark/microgpt.fut:339:94-160
                
                double zp_res_65667 = zt_res_65663 + zt_res_65666;
                
                // futhark/microgpt.fut:339:120-203
                
                double zp_res_65668 = zt_res_65666 + zp_res_65667;
                
                // futhark/microgpt.fut:339:53-203
                
                double zp_res_65669 = zp_lhs_65661 + zp_res_65668;
                
                ((double *) mem_72231)[i_70545] = zp_res_65669;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72226, i_70549 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72231, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70553 = 0; i_70553 < (int64_t) 16; i_70553++) {
            // futhark/microgpt.fut:343:49-59
            
            double zs_rhs_65717 = ((double *) mem_71115)[i_70553];
            
            // futhark/microgpt.fut:343:41-59
            
            double zs_res_65718 = 1.0 / zs_rhs_65717;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_65719;
            double r_65721 = 0.0;
            
            for (int64_t i_65720 = 0; i_65720 < (int64_t) 16; i_65720++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_65722 = ((double *) mem_71013)[i_70553 * (int64_t) 16 + i_65720];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_65723 = ((double *) mem_72226)[i_70553 * (int64_t) 16 + i_65720];
                
                // futhark/microgpt.fut:343:87-122
                
                double zt_res_65724 = zt_lhs_65722 * zt_rhs_65723;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_65725 = r_65721 + zt_res_65724;
                double r_tmp_72923 = zp_res_65725;
                
                r_65721 = r_tmp_72923;
            }
            defunc_0_lifted_lambda_res_65719 = r_65721;
            // futhark/microgpt.fut:343:67-149
            
            double zt_res_65726 = zs_res_65718 * defunc_0_lifted_lambda_res_65719;
            
            // futhark/microgpt.fut:343:45-149
            
            double zt_res_65727 = zs_res_65718 * zt_res_65726;
            
            // futhark/microgpt.fut:343:33-149
            
            double neg_res_65728 = -zt_res_65727;
            
            ((double *) mem_72242)[i_70553] = neg_res_65728;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70557 = 0; i_70557 < (int64_t) 16; i_70557++) {
            // futhark/microgpt.fut:344:33-43
            
            double zt_lhs_65736 = ((double *) mem_72242)[i_70557];
            
            // futhark/microgpt.fut:344:85-95
            
            double zp_lhs_65737 = ((double *) mem_71076)[i_70557];
            
            // futhark/microgpt.fut:344:85-123
            
            double zp_res_65738 = 1.0e-5 + zp_lhs_65737;
            
            // futhark/microgpt.fut:344:77-123
            
            double sqrt_res_65739 = futrts_sqrt64(zp_res_65738);
            
            // futhark/microgpt.fut:344:63-125
            
            double zt_res_65740 = 2.0 * sqrt_res_65739;
            
            // futhark/microgpt.fut:344:49-125
            
            double zs_res_65741 = 1.0 / zt_res_65740;
            
            // futhark/microgpt.fut:344:33-125
            
            double zt_res_65742 = zt_lhs_65736 * zs_res_65741;
            
            ((double *) mem_72249)[i_70557] = zt_res_65742;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70561 = 0; i_70561 < (int64_t) 16; i_70561++) {
            // futhark/microgpt.fut:345:53-63
            
            double zs_lhs_65750 = ((double *) mem_72249)[i_70561];
            
            // futhark/microgpt.fut:345:53-78
            
            double zs_res_65751 = zs_lhs_65750 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_72926 = 0; nest_i_72926 < (int64_t) 16; nest_i_72926++) {
                ((double *) mem_72256)[i_70561 * (int64_t) 16 + nest_i_72926] = zs_res_65751;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70569 = 0; i_70569 < (int64_t) 16; i_70569++) {
            // futhark/microgpt.fut:346:85-95
            
            double zs_rhs_65760 = ((double *) mem_71115)[i_70569];
            
            // futhark/microgpt.fut:346:77-95
            
            double zs_res_65761 = 1.0 / zs_rhs_65760;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70565 = 0; i_70565 < (int64_t) 16; i_70565++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65768 = ((double *) mem_72226)[i_70569 * (int64_t) 16 + i_70565];
                
                // futhark/microgpt.fut:346:55-95
                
                double zt_res_65769 = zs_res_65761 * zt_lhs_65768;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_65770 = ((double *) mem_71013)[i_70569 * (int64_t) 16 + i_70565];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_65771 = ((double *) mem_72256)[i_70569 * (int64_t) 16 + i_70565];
                
                // futhark/microgpt.fut:346:103-138
                
                double zt_res_65772 = zt_lhs_65770 * zt_rhs_65771;
                
                // futhark/microgpt.fut:346:72-138
                
                double zp_res_65773 = zt_res_65769 + zt_res_65772;
                
                // futhark/microgpt.fut:346:98-181
                
                double zp_res_65774 = zt_res_65772 + zp_res_65773;
                
                ((double *) mem_72271)[i_70565] = zp_res_65774;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72266, i_70569 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72271, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70582 = 0; i_70582 < (int64_t) 16; i_70582++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70575 = 0; i_70575 < (int64_t) 16; i_70575++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_69562 = ((double *) mem_72266)[i_70582 * (int64_t) 16 + i_70575];
                
                ((double *) mem_72292)[i_70575] = lifted_lambda_res_69562;
                ((double *) mem_72293)[i_70575] = lifted_lambda_res_69562;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72282, i_70582 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72292, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72283, i_70582 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72293, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70591 = 0; i_70591 < (int64_t) 64; i_70591++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70587 = 0; i_70587 < (int64_t) 16; i_70587++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_65888;
                double r_65890 = 0.0;
                
                for (int64_t i_65889 = 0; i_65889 < (int64_t) 16; i_65889++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_65891 = ((double *) mem_71686)[i_65889 * (int64_t) 64 + i_70591];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_65892 = ((double *) mem_71430)[i_65889 * (int64_t) 16 + i_70587];
                    
                    // futhark/microgpt.fut:354:73-109
                    
                    double zt_res_65893 = zt_lhs_65891 * zt_rhs_65892;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_65894 = r_65890 + zt_res_65893;
                    double r_tmp_72935 = zp_res_65894;
                    
                    r_65890 = r_tmp_72935;
                }
                defunc_0_lifted_lambda_res_65888 = r_65890;
                ((double *) mem_72319)[i_70587] = defunc_0_lifted_lambda_res_65888;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72314, i_70591 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_70604 = 0; i_70604 < (int64_t) 27; i_70604++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_70597 = 0; i_70597 < (int64_t) 16; i_70597++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69590;
                double r_69592 = 0.0;
                
                for (int64_t i_69591 = 0; i_69591 < (int64_t) 16; i_69591++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_69593 = ((double *) mem_71622)[i_69591 * (int64_t) 27 + i_70604];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_69594 = ((double *) mem_71523)[i_69591 * (int64_t) 16 + i_70597];
                    
                    // futhark/microgpt.fut:356:74-110
                    
                    double zt_res_69595 = zt_lhs_69593 * zt_rhs_69594;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69596 = r_69592 + zt_res_69595;
                    double r_tmp_72940 = zp_res_69596;
                    
                    r_69592 = r_tmp_72940;
                }
                defunc_0_lifted_lambda_res_69590 = r_69592;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_69599;
                double r_69601 = 0.0;
                
                for (int64_t i_69600 = 0; i_69600 < (int64_t) 16; i_69600++) {
                    int64_t zeze_lhs_69602 = ((int64_t *) seqs_mem_70871.mem)[step_64196 * (int64_t) 16 + i_69600];
                    
                    // futhark/microgpt.fut:475:58-109
                    
                    bool cond_69603 = zeze_lhs_69602 == i_70604;
                    
                    // futhark/microgpt.fut:475:58-109
                    
                    double lifted_lambda_res_69604;
                    
                    if (cond_69603) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_69902 = ((double *) mem_72282)[i_69600 * (int64_t) 16 + i_70597];
                        
                        lifted_lambda_res_69604 = lifted_lambda_res_t_res_69902;
                    } else {
                        lifted_lambda_res_69604 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_69610 = r_69601 + lifted_lambda_res_69604;
                    double r_tmp_72941 = zp_res_69610;
                    
                    r_69601 = r_tmp_72941;
                }
                defunc_0_lifted_lambda_res_69599 = r_69601;
                ((double *) mem_72340)[i_70597] = defunc_0_lifted_lambda_res_69599;
                ((double *) mem_72341)[i_70597] = defunc_0_lifted_lambda_res_69590;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72330, i_70604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72340, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_72331, i_70604 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_72341, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_65972 = sitofp_i64_f64(step_64196);
        
        // futhark/microgpt.fut:431:46-67
        
        double zm_rhs_65973 = i64_res_65972 / 10000.0;
        
        // futhark/microgpt.fut:431:24-67
        
        double zt_rhs_65974 = 1.0 - zm_rhs_65973;
        
        // futhark/microgpt.fut:431:19-67
        
        double lt_r_65975 = 1.0e-2 * zt_rhs_65974;
        
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_72362, (int64_t) 3456, "mem_72362")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72362.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70895.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_72364, (int64_t) 3456, "mem_72364")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72364.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70931.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_72366, (int64_t) 3456, "mem_72366")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72366.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70967.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (memblock_alloc(ctx, &mem_72368, (int64_t) 3456, "mem_72368")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:433:5-52
        // futhark/microgpt.fut:433:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72368.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72330, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:433:5-52
        if (futrts_adam_opt_w_9496(ctx, &ext_mem_72372, &ext_mem_72371, &ext_mem_72370, mem_72362, mem_72364, mem_72366, mem_72368, (int64_t) 27, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72362, "mem_72362") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72364, "mem_72364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72366, "mem_72366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72368, "mem_72368") != 0)
            return 1;
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_72373, (int64_t) 2048, "mem_72373")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72373.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70887.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_72375, (int64_t) 2048, "mem_72375")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72375.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70923.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_72377, (int64_t) 2048, "mem_72377")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72377.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70959.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (memblock_alloc(ctx, &mem_72379, (int64_t) 2048, "mem_72379")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:435:5-52
        // futhark/microgpt.fut:435:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72379.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72283, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:435:5-52
        if (futrts_adam_opt_w_9497(ctx, &ext_mem_72383, &ext_mem_72382, &ext_mem_72381, mem_72373, mem_72375, mem_72377, mem_72379, (int64_t) 16, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72373, "mem_72373") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72375, "mem_72375") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72377, "mem_72377") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72379, "mem_72379") != 0)
            return 1;
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_72384, (int64_t) 2048, "mem_72384")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72384.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70891.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_72386, (int64_t) 2048, "mem_72386")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72386.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70927.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_72388, (int64_t) 2048, "mem_72388")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72388.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70963.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (memblock_alloc(ctx, &mem_72390, (int64_t) 2048, "mem_72390")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:437:5-56
        // futhark/microgpt.fut:437:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72390.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72140, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:437:5-56
        if (futrts_adam_opt_w_9497(ctx, &ext_mem_72394, &ext_mem_72393, &ext_mem_72392, mem_72384, mem_72386, mem_72388, mem_72390, (int64_t) 16, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72384, "mem_72384") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72386, "mem_72386") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72388, "mem_72388") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72390, "mem_72390") != 0)
            return 1;
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_72395, (int64_t) 2048, "mem_72395")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72395.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70879.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_72397, (int64_t) 2048, "mem_72397")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72397.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70915.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_72399, (int64_t) 2048, "mem_72399")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72399.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70951.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (memblock_alloc(ctx, &mem_72401, (int64_t) 2048, "mem_72401")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:439:5-56
        // futhark/microgpt.fut:439:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72401.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72139, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:439:5-56
        if (futrts_adam_opt_w_9497(ctx, &ext_mem_72405, &ext_mem_72404, &ext_mem_72403, mem_72395, mem_72397, mem_72399, mem_72401, (int64_t) 16, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72395, "mem_72395") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72397, "mem_72397") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72399, "mem_72399") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72401, "mem_72401") != 0)
            return 1;
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_72406, (int64_t) 2048, "mem_72406")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72406.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70903.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_72408, (int64_t) 2048, "mem_72408")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72408.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70939.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_72410, (int64_t) 2048, "mem_72410")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72410.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70975.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (memblock_alloc(ctx, &mem_72412, (int64_t) 2048, "mem_72412")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:441:5-56
        // futhark/microgpt.fut:441:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72412.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72138, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:441:5-56
        if (futrts_adam_opt_w_9497(ctx, &ext_mem_72416, &ext_mem_72415, &ext_mem_72414, mem_72406, mem_72408, mem_72410, mem_72412, (int64_t) 16, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72406, "mem_72406") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72408, "mem_72408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72410, "mem_72410") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72412, "mem_72412") != 0)
            return 1;
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_72417, (int64_t) 2048, "mem_72417")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72417.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70883.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_72419, (int64_t) 2048, "mem_72419")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72419.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70919.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_72421, (int64_t) 2048, "mem_72421")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72421.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70955.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (memblock_alloc(ctx, &mem_72423, (int64_t) 2048, "mem_72423")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:443:5-56
        // futhark/microgpt.fut:443:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72423.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_71758, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:443:5-56
        if (futrts_adam_opt_w_9497(ctx, &ext_mem_72427, &ext_mem_72426, &ext_mem_72425, mem_72417, mem_72419, mem_72421, mem_72423, (int64_t) 16, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72417, "mem_72417") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72419, "mem_72419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72421, "mem_72421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72423, "mem_72423") != 0)
            return 1;
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_72428, (int64_t) 8192, "mem_72428")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72428.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70899.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_72430, (int64_t) 8192, "mem_72430")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72430.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70935.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_72432, (int64_t) 8192, "mem_72432")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72432.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70971.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (memblock_alloc(ctx, &mem_72434, (int64_t) 8192, "mem_72434")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:445:5-52
        // futhark/microgpt.fut:445:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72434.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72314, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:445:5-52
        if (futrts_adam_opt_w_9496(ctx, &ext_mem_72438, &ext_mem_72437, &ext_mem_72436, mem_72428, mem_72430, mem_72432, mem_72434, (int64_t) 64, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72428, "mem_72428") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72430, "mem_72430") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72432, "mem_72432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72434, "mem_72434") != 0)
            return 1;
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_72439, (int64_t) 8192, "mem_72439")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72439.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_70875.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_72441, (int64_t) 8192, "mem_72441")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72441.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_70911.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_72443, (int64_t) 8192, "mem_72443")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72443.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_70947.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (memblock_alloc(ctx, &mem_72445, (int64_t) 8192, "mem_72445")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:447:5-60
        // futhark/microgpt.fut:447:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72445.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_71654, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:447:5-60
        if (futrts_adam_opt_w_9496(ctx, &ext_mem_72449, &ext_mem_72448, &ext_mem_72447, mem_72439, mem_72441, mem_72443, mem_72445, (int64_t) 16, (int64_t) 64, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72439, "mem_72439") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72441, "mem_72441") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72443, "mem_72443") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72445, "mem_72445") != 0)
            return 1;
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_72450, (int64_t) 3456, "mem_72450")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72450.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70907.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_72452, (int64_t) 3456, "mem_72452")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72452.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70943.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_72454, (int64_t) 3456, "mem_72454")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72454.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_70979.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (memblock_alloc(ctx, &mem_72456, (int64_t) 3456, "mem_72456")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:449:5-56
        // futhark/microgpt.fut:449:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_72456.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_72331, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:449:5-56
        if (futrts_adam_opt_w_9496(ctx, &ext_mem_72460, &ext_mem_72459, &ext_mem_72458, mem_72450, mem_72452, mem_72454, mem_72456, (int64_t) 27, (int64_t) 16, step_64196, lt_r_65975) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_72450, "mem_72450") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72452, "mem_72452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72454, "mem_72454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72456, "mem_72456") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72668, &ext_mem_72449, "ext_mem_72449") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72669, &ext_mem_72405, "ext_mem_72405") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72670, &ext_mem_72427, "ext_mem_72427") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72671, &ext_mem_72383, "ext_mem_72383") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72672, &ext_mem_72394, "ext_mem_72394") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72673, &ext_mem_72372, "ext_mem_72372") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72674, &ext_mem_72438, "ext_mem_72438") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72675, &ext_mem_72416, "ext_mem_72416") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72676, &ext_mem_72460, "ext_mem_72460") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72677, &ext_mem_72448, "ext_mem_72448") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72678, &ext_mem_72404, "ext_mem_72404") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72679, &ext_mem_72426, "ext_mem_72426") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72680, &ext_mem_72382, "ext_mem_72382") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72681, &ext_mem_72393, "ext_mem_72393") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72682, &ext_mem_72371, "ext_mem_72371") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72683, &ext_mem_72437, "ext_mem_72437") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72684, &ext_mem_72415, "ext_mem_72415") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72685, &ext_mem_72459, "ext_mem_72459") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72686, &ext_mem_72447, "ext_mem_72447") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72687, &ext_mem_72403, "ext_mem_72403") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72688, &ext_mem_72425, "ext_mem_72425") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72689, &ext_mem_72381, "ext_mem_72381") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72690, &ext_mem_72392, "ext_mem_72392") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72691, &ext_mem_72370, "ext_mem_72370") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72692, &ext_mem_72436, "ext_mem_72436") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72693, &ext_mem_72414, "ext_mem_72414") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_72694, &ext_mem_72458, "ext_mem_72458") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70875, &mem_param_tmp_72668, "mem_param_tmp_72668") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70879, &mem_param_tmp_72669, "mem_param_tmp_72669") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70883, &mem_param_tmp_72670, "mem_param_tmp_72670") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70887, &mem_param_tmp_72671, "mem_param_tmp_72671") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70891, &mem_param_tmp_72672, "mem_param_tmp_72672") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70895, &mem_param_tmp_72673, "mem_param_tmp_72673") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70899, &mem_param_tmp_72674, "mem_param_tmp_72674") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70903, &mem_param_tmp_72675, "mem_param_tmp_72675") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70907, &mem_param_tmp_72676, "mem_param_tmp_72676") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70911, &mem_param_tmp_72677, "mem_param_tmp_72677") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70915, &mem_param_tmp_72678, "mem_param_tmp_72678") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70919, &mem_param_tmp_72679, "mem_param_tmp_72679") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70923, &mem_param_tmp_72680, "mem_param_tmp_72680") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70927, &mem_param_tmp_72681, "mem_param_tmp_72681") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70931, &mem_param_tmp_72682, "mem_param_tmp_72682") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70935, &mem_param_tmp_72683, "mem_param_tmp_72683") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70939, &mem_param_tmp_72684, "mem_param_tmp_72684") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70943, &mem_param_tmp_72685, "mem_param_tmp_72685") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70947, &mem_param_tmp_72686, "mem_param_tmp_72686") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70951, &mem_param_tmp_72687, "mem_param_tmp_72687") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70955, &mem_param_tmp_72688, "mem_param_tmp_72688") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70959, &mem_param_tmp_72689, "mem_param_tmp_72689") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70963, &mem_param_tmp_72690, "mem_param_tmp_72690") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70967, &mem_param_tmp_72691, "mem_param_tmp_72691") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70971, &mem_param_tmp_72692, "mem_param_tmp_72692") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70975, &mem_param_tmp_72693, "mem_param_tmp_72693") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_70979, &mem_param_tmp_72694, "mem_param_tmp_72694") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_72568, &mem_param_70875, "mem_param_70875") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72567, &mem_param_70879, "mem_param_70879") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72566, &mem_param_70883, "mem_param_70883") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72565, &mem_param_70887, "mem_param_70887") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72564, &mem_param_70891, "mem_param_70891") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72563, &mem_param_70895, "mem_param_70895") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72562, &mem_param_70899, "mem_param_70899") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72561, &mem_param_70903, "mem_param_70903") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72560, &mem_param_70907, "mem_param_70907") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72559, &mem_param_70911, "mem_param_70911") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72558, &mem_param_70915, "mem_param_70915") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72557, &mem_param_70919, "mem_param_70919") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72556, &mem_param_70923, "mem_param_70923") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72555, &mem_param_70927, "mem_param_70927") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72554, &mem_param_70931, "mem_param_70931") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72553, &mem_param_70935, "mem_param_70935") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72552, &mem_param_70939, "mem_param_70939") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72551, &mem_param_70943, "mem_param_70943") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72550, &mem_param_70947, "mem_param_70947") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72549, &mem_param_70951, "mem_param_70951") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72548, &mem_param_70955, "mem_param_70955") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72547, &mem_param_70959, "mem_param_70959") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72546, &mem_param_70963, "mem_param_70963") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72545, &mem_param_70967, "mem_param_70967") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72544, &mem_param_70971, "mem_param_70971") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72543, &mem_param_70975, "mem_param_70975") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_72542, &mem_param_70979, "mem_param_70979") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72641, &ext_mem_72563, "ext_mem_72563") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72642, &ext_mem_72565, "ext_mem_72565") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72643, &ext_mem_72564, "ext_mem_72564") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72644, &ext_mem_72567, "ext_mem_72567") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72645, &ext_mem_72561, "ext_mem_72561") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72646, &ext_mem_72566, "ext_mem_72566") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72647, &ext_mem_72562, "ext_mem_72562") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72648, &ext_mem_72568, "ext_mem_72568") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72649, &ext_mem_72560, "ext_mem_72560") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72650, &ext_mem_72554, "ext_mem_72554") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72651, &ext_mem_72556, "ext_mem_72556") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72652, &ext_mem_72555, "ext_mem_72555") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72653, &ext_mem_72558, "ext_mem_72558") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72654, &ext_mem_72552, "ext_mem_72552") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72655, &ext_mem_72557, "ext_mem_72557") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72656, &ext_mem_72553, "ext_mem_72553") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72657, &ext_mem_72559, "ext_mem_72559") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72658, &ext_mem_72551, "ext_mem_72551") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72659, &ext_mem_72545, "ext_mem_72545") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72660, &ext_mem_72547, "ext_mem_72547") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72661, &ext_mem_72546, "ext_mem_72546") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72662, &ext_mem_72549, "ext_mem_72549") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72663, &ext_mem_72543, "ext_mem_72543") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72664, &ext_mem_72548, "ext_mem_72548") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72665, &ext_mem_72544, "ext_mem_72544") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72666, &ext_mem_72550, "ext_mem_72550") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72667, &ext_mem_72542, "ext_mem_72542") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72979, &mem_out_72641, "mem_out_72641") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72980, &mem_out_72642, "mem_out_72642") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72981, &mem_out_72643, "mem_out_72643") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72982, &mem_out_72644, "mem_out_72644") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72983, &mem_out_72645, "mem_out_72645") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72984, &mem_out_72646, "mem_out_72646") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72985, &mem_out_72647, "mem_out_72647") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72986, &mem_out_72648, "mem_out_72648") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72987, &mem_out_72649, "mem_out_72649") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72988, &mem_out_72650, "mem_out_72650") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72989, &mem_out_72651, "mem_out_72651") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72990, &mem_out_72652, "mem_out_72652") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72991, &mem_out_72653, "mem_out_72653") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72992, &mem_out_72654, "mem_out_72654") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72993, &mem_out_72655, "mem_out_72655") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72994, &mem_out_72656, "mem_out_72656") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72995, &mem_out_72657, "mem_out_72657") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72996, &mem_out_72658, "mem_out_72658") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72997, &mem_out_72659, "mem_out_72659") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72998, &mem_out_72660, "mem_out_72660") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_72999, &mem_out_72661, "mem_out_72661") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73000, &mem_out_72662, "mem_out_72662") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73001, &mem_out_72663, "mem_out_72663") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73002, &mem_out_72664, "mem_out_72664") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73003, &mem_out_72665, "mem_out_72665") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73004, &mem_out_72666, "mem_out_72666") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73005, &mem_out_72667, "mem_out_72667") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_70980);
        free(mem_70981);
        free(mem_70990);
        free(mem_70997);
        free(mem_71012);
        free(mem_71013);
        free(mem_71022);
        free(mem_71029);
        free(mem_71044);
        free(mem_71045);
        free(mem_71054);
        free(mem_71055);
        free(mem_71076);
        free(mem_71077);
        free(mem_71078);
        free(mem_71090);
        free(mem_71091);
        free(mem_71115);
        free(mem_71116);
        free(mem_71117);
        free(mem_71118);
        free(mem_71119);
        free(mem_71138);
        free(mem_71139);
        free(mem_71140);
        free(mem_71177);
        free(mem_71178);
        free(mem_71179);
        free(mem_71195);
        free(mem_71196);
        free(mem_71197);
        free(mem_71210);
        free(mem_71211);
        free(mem_71212);
        free(mem_71258);
        free(mem_71259);
        free(mem_71270);
        free(mem_71271);
        free(mem_71280);
        free(mem_71281);
        free(mem_71302);
        free(mem_71307);
        free(mem_71318);
        free(mem_71323);
        free(mem_71330);
        free(mem_71337);
        free(mem_71348);
        free(mem_71353);
        free(mem_71374);
        free(mem_71375);
        free(mem_71383);
        free(mem_71397);
        free(mem_71402);
        free(mem_71413);
        free(mem_71418);
        free(mem_71429);
        free(mem_71430);
        free(mem_71439);
        free(mem_71440);
        free(mem_71461);
        free(mem_71462);
        free(mem_71470);
        free(mem_71484);
        free(mem_71485);
        free(mem_71493);
        free(mem_71507);
        free(mem_71512);
        free(mem_71523);
        free(mem_71528);
        free(mem_71539);
        free(mem_71544);
        free(mem_71555);
        free(mem_71556);
        free(mem_71565);
        free(mem_71566);
        free(mem_71579);
        free(mem_71580);
        free(mem_71593);
        free(mem_71594);
        free(mem_71615);
        free(mem_71622);
        free(mem_71627);
        free(mem_71638);
        free(mem_71643);
        free(mem_71654);
        free(mem_71655);
        free(mem_71664);
        free(mem_71665);
        free(mem_71686);
        free(mem_71691);
        free(mem_71702);
        free(mem_71707);
        free(mem_71718);
        free(mem_71725);
        free(mem_71732);
        free(mem_71742);
        free(mem_71747);
        free(mem_71758);
        free(mem_71759);
        free(mem_71768);
        free(mem_71769);
        free(mem_71790);
        free(mem_71791);
        free(mem_71802);
        free(mem_71803);
        free(mem_71812);
        free(mem_71819);
        free(mem_71844);
        free(mem_71845);
        free(mem_71856);
        free(mem_71857);
        free(mem_71866);
        free(mem_71873);
        free(mem_71880);
        free(mem_71887);
        free(mem_71912);
        free(mem_71913);
        free(mem_71924);
        free(mem_71925);
        free(mem_71934);
        free(mem_71941);
        free(mem_71966);
        free(mem_71971);
        free(mem_71982);
        free(mem_71988);
        free(mem_71993);
        free(mem_72009);
        free(mem_72015);
        free(mem_72020);
        free(mem_72036);
        free(mem_72037);
        free(mem_72048);
        free(mem_72049);
        free(mem_72058);
        free(mem_72059);
        free(mem_72090);
        free(mem_72091);
        free(mem_72092);
        free(mem_72105);
        free(mem_72106);
        free(mem_72107);
        free(mem_72138);
        free(mem_72139);
        free(mem_72140);
        free(mem_72141);
        free(mem_72158);
        free(mem_72159);
        free(mem_72160);
        free(mem_72161);
        free(mem_72202);
        free(mem_72209);
        free(mem_72216);
        free(mem_72226);
        free(mem_72231);
        free(mem_72242);
        free(mem_72249);
        free(mem_72256);
        free(mem_72266);
        free(mem_72271);
        free(mem_72282);
        free(mem_72283);
        free(mem_72292);
        free(mem_72293);
        free(mem_72314);
        free(mem_72319);
        free(mem_72330);
        free(mem_72331);
        free(mem_72340);
        free(mem_72341);
        if (memblock_unref(ctx, &mem_param_tmp_72694, "mem_param_tmp_72694") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72693, "mem_param_tmp_72693") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72692, "mem_param_tmp_72692") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72691, "mem_param_tmp_72691") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72690, "mem_param_tmp_72690") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72689, "mem_param_tmp_72689") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72688, "mem_param_tmp_72688") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72687, "mem_param_tmp_72687") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72686, "mem_param_tmp_72686") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72685, "mem_param_tmp_72685") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72684, "mem_param_tmp_72684") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72683, "mem_param_tmp_72683") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72682, "mem_param_tmp_72682") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72681, "mem_param_tmp_72681") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72680, "mem_param_tmp_72680") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72679, "mem_param_tmp_72679") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72678, "mem_param_tmp_72678") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72677, "mem_param_tmp_72677") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72676, "mem_param_tmp_72676") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72675, "mem_param_tmp_72675") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72674, "mem_param_tmp_72674") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72673, "mem_param_tmp_72673") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72672, "mem_param_tmp_72672") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72671, "mem_param_tmp_72671") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72670, "mem_param_tmp_72670") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72669, "mem_param_tmp_72669") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_72668, "mem_param_tmp_72668") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72458, "ext_mem_72458") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72459, "ext_mem_72459") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72460, "ext_mem_72460") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72456, "mem_72456") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72454, "mem_72454") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72452, "mem_72452") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72450, "mem_72450") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72447, "ext_mem_72447") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72448, "ext_mem_72448") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72449, "ext_mem_72449") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72445, "mem_72445") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72443, "mem_72443") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72441, "mem_72441") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72439, "mem_72439") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72436, "ext_mem_72436") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72437, "ext_mem_72437") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72438, "ext_mem_72438") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72434, "mem_72434") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72432, "mem_72432") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72430, "mem_72430") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72428, "mem_72428") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72425, "ext_mem_72425") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72426, "ext_mem_72426") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72427, "ext_mem_72427") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72423, "mem_72423") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72421, "mem_72421") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72419, "mem_72419") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72417, "mem_72417") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72414, "ext_mem_72414") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72415, "ext_mem_72415") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72416, "ext_mem_72416") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72412, "mem_72412") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72410, "mem_72410") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72408, "mem_72408") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72406, "mem_72406") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72403, "ext_mem_72403") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72404, "ext_mem_72404") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72405, "ext_mem_72405") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72401, "mem_72401") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72399, "mem_72399") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72397, "mem_72397") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72395, "mem_72395") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72392, "ext_mem_72392") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72393, "ext_mem_72393") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72394, "ext_mem_72394") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72390, "mem_72390") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72388, "mem_72388") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72386, "mem_72386") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72384, "mem_72384") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72381, "ext_mem_72381") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72382, "ext_mem_72382") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72383, "ext_mem_72383") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72379, "mem_72379") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72377, "mem_72377") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72375, "mem_72375") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72373, "mem_72373") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72370, "ext_mem_72370") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72371, "ext_mem_72371") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72372, "ext_mem_72372") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72368, "mem_72368") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72366, "mem_72366") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72364, "mem_72364") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_72362, "mem_72362") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70979, "mem_param_70979") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70975, "mem_param_70975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70971, "mem_param_70971") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70967, "mem_param_70967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70963, "mem_param_70963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70959, "mem_param_70959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70955, "mem_param_70955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70951, "mem_param_70951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70947, "mem_param_70947") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70943, "mem_param_70943") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70939, "mem_param_70939") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70935, "mem_param_70935") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70931, "mem_param_70931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70927, "mem_param_70927") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70923, "mem_param_70923") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70919, "mem_param_70919") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70915, "mem_param_70915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70911, "mem_param_70911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70907, "mem_param_70907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70903, "mem_param_70903") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70899, "mem_param_70899") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70895, "mem_param_70895") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70891, "mem_param_70891") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70887, "mem_param_70887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70883, "mem_param_70883") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70879, "mem_param_70879") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_70875, "mem_param_70875") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72542, "ext_mem_72542") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72543, "ext_mem_72543") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72544, "ext_mem_72544") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72545, "ext_mem_72545") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72546, "ext_mem_72546") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72547, "ext_mem_72547") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72548, "ext_mem_72548") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72549, "ext_mem_72549") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72550, "ext_mem_72550") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72551, "ext_mem_72551") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72552, "ext_mem_72552") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72553, "ext_mem_72553") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72554, "ext_mem_72554") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72555, "ext_mem_72555") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72556, "ext_mem_72556") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72557, "ext_mem_72557") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72558, "ext_mem_72558") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72559, "ext_mem_72559") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72560, "ext_mem_72560") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72561, "ext_mem_72561") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72562, "ext_mem_72562") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72563, "ext_mem_72563") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72564, "ext_mem_72564") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72565, "ext_mem_72565") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72566, "ext_mem_72566") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72567, "ext_mem_72567") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_72568, "ext_mem_72568") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72667, "mem_out_72667") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72666, "mem_out_72666") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72665, "mem_out_72665") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72664, "mem_out_72664") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72663, "mem_out_72663") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72662, "mem_out_72662") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72661, "mem_out_72661") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72660, "mem_out_72660") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72659, "mem_out_72659") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72658, "mem_out_72658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72657, "mem_out_72657") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72656, "mem_out_72656") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72655, "mem_out_72655") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72654, "mem_out_72654") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72653, "mem_out_72653") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72652, "mem_out_72652") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72651, "mem_out_72651") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72650, "mem_out_72650") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72649, "mem_out_72649") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72648, "mem_out_72648") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72647, "mem_out_72647") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72646, "mem_out_72646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72645, "mem_out_72645") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72644, "mem_out_72644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72643, "mem_out_72643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72642, "mem_out_72642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72641, "mem_out_72641") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_73175, struct memblock *mem_out_p_73176, struct memblock *mem_out_p_73177, struct memblock *mem_out_p_73178, struct memblock *mem_out_p_73179, struct memblock *mem_out_p_73180, struct memblock *mem_out_p_73181, struct memblock *mem_out_p_73182, struct memblock *mem_out_p_73183)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock mem_70833 = ctx->constants->mem_70833;
    struct memblock mem_70834 = ctx->constants->mem_70834;
    struct memblock mem_70835 = ctx->constants->mem_70835;
    struct memblock mem_70836 = ctx->constants->mem_70836;
    struct memblock mem_70837 = ctx->constants->mem_70837;
    struct memblock mem_70838 = ctx->constants->mem_70838;
    struct memblock mem_70839 = ctx->constants->mem_70839;
    struct memblock mem_70840 = ctx->constants->mem_70840;
    struct memblock mem_70841 = ctx->constants->mem_70841;
    
    if (memblock_set(ctx, &mem_out_72641, &mem_70840, "mem_70840") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72642, &mem_70836, "mem_70836") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72643, &mem_70838, "mem_70838") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72644, &mem_70834, "mem_70834") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72645, &mem_70835, "mem_70835") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72646, &mem_70833, "mem_70833") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72647, &mem_70839, "mem_70839") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72648, &mem_70837, "mem_70837") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_72649, &mem_70841, "mem_70841") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73175, &mem_out_72641, "mem_out_72641") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73176, &mem_out_72642, "mem_out_72642") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73177, &mem_out_72643, "mem_out_72643") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73178, &mem_out_72644, "mem_out_72644") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73179, &mem_out_72645, "mem_out_72645") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73180, &mem_out_72646, "mem_out_72646") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73181, &mem_out_72647, "mem_out_72647") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73182, &mem_out_72648, "mem_out_72648") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_73183, &mem_out_72649, "mem_out_72649") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_72649, "mem_out_72649") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72648, "mem_out_72648") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72647, "mem_out_72647") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72646, "mem_out_72646") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72645, "mem_out_72645") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72644, "mem_out_72644") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72643, "mem_out_72643") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72642, "mem_out_72642") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_72641, "mem_out_72641") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f64_2d *in0, const struct futhark_f64_2d *in1, const struct futhark_f64_2d *in2, const struct futhark_f64_2d *in3, const struct futhark_f64_2d *in4, const struct futhark_f64_2d *in5, const struct futhark_f64_2d *in6, const struct futhark_f64_2d *in7, const struct futhark_f64_2d *in8)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock wvoc_mem_70850;
    
    wvoc_mem_70850.references = NULL;
    
    struct memblock wdown_mem_70849;
    
    wdown_mem_70849.references = NULL;
    
    struct memblock wup_mem_70848;
    
    wup_mem_70848.references = NULL;
    
    struct memblock wout_mem_70847;
    
    wout_mem_70847.references = NULL;
    
    struct memblock wval_mem_70846;
    
    wval_mem_70846.references = NULL;
    
    struct memblock wkey_mem_70845;
    
    wkey_mem_70845.references = NULL;
    
    struct memblock wqry_mem_70844;
    
    wqry_mem_70844.references = NULL;
    
    struct memblock wpe_mem_70843;
    
    wpe_mem_70843.references = NULL;
    
    struct memblock wte_mem_70842;
    
    wte_mem_70842.references = NULL;
    wte_mem_70842 = in0->mem;
    wpe_mem_70843 = in1->mem;
    wqry_mem_70844 = in2->mem;
    wkey_mem_70845 = in3->mem;
    wval_mem_70846 = in4->mem;
    wout_mem_70847 = in5->mem;
    wup_mem_70848 = in6->mem;
    wdown_mem_70849 = in7->mem;
    wvoc_mem_70850 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_72641, &mem_out_72642, &mem_out_72643, &mem_out_72644, &mem_out_72645, &mem_out_72646, &mem_out_72647, &mem_out_72648, &mem_out_72649, wte_mem_70842, wpe_mem_70843, wqry_mem_70844, wkey_mem_70845, wval_mem_70846, wout_mem_70847, wup_mem_70848, wdown_mem_70849, wvoc_mem_70850);
        if (ret == 0) {
            struct memblock mem_70833 = ctx->constants->mem_70833;
            struct memblock mem_70834 = ctx->constants->mem_70834;
            struct memblock mem_70835 = ctx->constants->mem_70835;
            struct memblock mem_70836 = ctx->constants->mem_70836;
            struct memblock mem_70837 = ctx->constants->mem_70837;
            struct memblock mem_70838 = ctx->constants->mem_70838;
            struct memblock mem_70839 = ctx->constants->mem_70839;
            struct memblock mem_70840 = ctx->constants->mem_70840;
            struct memblock mem_70841 = ctx->constants->mem_70841;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_72641;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_72642;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_72643;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_72644;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_72645;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_72646;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_72647;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_72648;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_72649;
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
    
    struct memblock mem_out_72667;
    
    mem_out_72667.references = NULL;
    
    struct memblock mem_out_72666;
    
    mem_out_72666.references = NULL;
    
    struct memblock mem_out_72665;
    
    mem_out_72665.references = NULL;
    
    struct memblock mem_out_72664;
    
    mem_out_72664.references = NULL;
    
    struct memblock mem_out_72663;
    
    mem_out_72663.references = NULL;
    
    struct memblock mem_out_72662;
    
    mem_out_72662.references = NULL;
    
    struct memblock mem_out_72661;
    
    mem_out_72661.references = NULL;
    
    struct memblock mem_out_72660;
    
    mem_out_72660.references = NULL;
    
    struct memblock mem_out_72659;
    
    mem_out_72659.references = NULL;
    
    struct memblock mem_out_72658;
    
    mem_out_72658.references = NULL;
    
    struct memblock mem_out_72657;
    
    mem_out_72657.references = NULL;
    
    struct memblock mem_out_72656;
    
    mem_out_72656.references = NULL;
    
    struct memblock mem_out_72655;
    
    mem_out_72655.references = NULL;
    
    struct memblock mem_out_72654;
    
    mem_out_72654.references = NULL;
    
    struct memblock mem_out_72653;
    
    mem_out_72653.references = NULL;
    
    struct memblock mem_out_72652;
    
    mem_out_72652.references = NULL;
    
    struct memblock mem_out_72651;
    
    mem_out_72651.references = NULL;
    
    struct memblock mem_out_72650;
    
    mem_out_72650.references = NULL;
    
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    
    struct memblock seqs_mem_70871;
    
    seqs_mem_70871.references = NULL;
    
    struct memblock dls_mem_70870;
    
    dls_mem_70870.references = NULL;
    
    struct memblock masks_mem_70869;
    
    masks_mem_70869.references = NULL;
    
    struct memblock wvoc_mem_70868;
    
    wvoc_mem_70868.references = NULL;
    
    struct memblock wval_mem_70867;
    
    wval_mem_70867.references = NULL;
    
    struct memblock wup_mem_70866;
    
    wup_mem_70866.references = NULL;
    
    struct memblock wte_mem_70865;
    
    wte_mem_70865.references = NULL;
    
    struct memblock wqry_mem_70864;
    
    wqry_mem_70864.references = NULL;
    
    struct memblock wpe_mem_70863;
    
    wpe_mem_70863.references = NULL;
    
    struct memblock wout_mem_70862;
    
    wout_mem_70862.references = NULL;
    
    struct memblock wkey_mem_70861;
    
    wkey_mem_70861.references = NULL;
    
    struct memblock wdown_mem_70860;
    
    wdown_mem_70860.references = NULL;
    
    struct memblock wvoc_mem_70859;
    
    wvoc_mem_70859.references = NULL;
    
    struct memblock wval_mem_70858;
    
    wval_mem_70858.references = NULL;
    
    struct memblock wup_mem_70857;
    
    wup_mem_70857.references = NULL;
    
    struct memblock wte_mem_70856;
    
    wte_mem_70856.references = NULL;
    
    struct memblock wqry_mem_70855;
    
    wqry_mem_70855.references = NULL;
    
    struct memblock wpe_mem_70854;
    
    wpe_mem_70854.references = NULL;
    
    struct memblock wout_mem_70853;
    
    wout_mem_70853.references = NULL;
    
    struct memblock wkey_mem_70852;
    
    wkey_mem_70852.references = NULL;
    
    struct memblock wdown_mem_70851;
    
    wdown_mem_70851.references = NULL;
    
    struct memblock wvoc_mem_70850;
    
    wvoc_mem_70850.references = NULL;
    
    struct memblock wval_mem_70849;
    
    wval_mem_70849.references = NULL;
    
    struct memblock wup_mem_70848;
    
    wup_mem_70848.references = NULL;
    
    struct memblock wte_mem_70847;
    
    wte_mem_70847.references = NULL;
    
    struct memblock wqry_mem_70846;
    
    wqry_mem_70846.references = NULL;
    
    struct memblock wpe_mem_70845;
    
    wpe_mem_70845.references = NULL;
    
    struct memblock wout_mem_70844;
    
    wout_mem_70844.references = NULL;
    
    struct memblock wkey_mem_70843;
    
    wkey_mem_70843.references = NULL;
    
    struct memblock wdown_mem_70842;
    
    wdown_mem_70842.references = NULL;
    wdown_mem_70842 = in0->v0->mem;
    wkey_mem_70843 = in0->v1->mem;
    wout_mem_70844 = in0->v2->mem;
    wpe_mem_70845 = in0->v3->mem;
    wqry_mem_70846 = in0->v4->mem;
    wte_mem_70847 = in0->v5->mem;
    wup_mem_70848 = in0->v6->mem;
    wval_mem_70849 = in0->v7->mem;
    wvoc_mem_70850 = in0->v8->mem;
    wdown_mem_70851 = in1->v0->mem;
    wkey_mem_70852 = in1->v1->mem;
    wout_mem_70853 = in1->v2->mem;
    wpe_mem_70854 = in1->v3->mem;
    wqry_mem_70855 = in1->v4->mem;
    wte_mem_70856 = in1->v5->mem;
    wup_mem_70857 = in1->v6->mem;
    wval_mem_70858 = in1->v7->mem;
    wvoc_mem_70859 = in1->v8->mem;
    wdown_mem_70860 = in2->v0->mem;
    wkey_mem_70861 = in2->v1->mem;
    wout_mem_70862 = in2->v2->mem;
    wpe_mem_70863 = in2->v3->mem;
    wqry_mem_70864 = in2->v4->mem;
    wte_mem_70865 = in2->v5->mem;
    wup_mem_70866 = in2->v6->mem;
    wval_mem_70867 = in2->v7->mem;
    wvoc_mem_70868 = in2->v8->mem;
    masks_mem_70869 = in3->mem;
    dls_mem_70870 = in4->mem;
    seqs_mem_70871 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 10000 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 10000 == in4->shape[0] && ((int64_t) 10000 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_72641, &mem_out_72642, &mem_out_72643, &mem_out_72644, &mem_out_72645, &mem_out_72646, &mem_out_72647, &mem_out_72648, &mem_out_72649, &mem_out_72650, &mem_out_72651, &mem_out_72652, &mem_out_72653, &mem_out_72654, &mem_out_72655, &mem_out_72656, &mem_out_72657, &mem_out_72658, &mem_out_72659, &mem_out_72660, &mem_out_72661, &mem_out_72662, &mem_out_72663, &mem_out_72664, &mem_out_72665, &mem_out_72666, &mem_out_72667, wdown_mem_70842, wkey_mem_70843, wout_mem_70844, wpe_mem_70845, wqry_mem_70846, wte_mem_70847, wup_mem_70848, wval_mem_70849, wvoc_mem_70850, wdown_mem_70851, wkey_mem_70852, wout_mem_70853, wpe_mem_70854, wqry_mem_70855, wte_mem_70856, wup_mem_70857, wval_mem_70858, wvoc_mem_70859, wdown_mem_70860, wkey_mem_70861, wout_mem_70862, wpe_mem_70863, wqry_mem_70864, wte_mem_70865, wup_mem_70866, wval_mem_70867, wvoc_mem_70868, masks_mem_70869, dls_mem_70870, seqs_mem_70871);
        if (ret == 0) {
            struct memblock mem_70833 = ctx->constants->mem_70833;
            struct memblock mem_70834 = ctx->constants->mem_70834;
            struct memblock mem_70835 = ctx->constants->mem_70835;
            struct memblock mem_70836 = ctx->constants->mem_70836;
            struct memblock mem_70837 = ctx->constants->mem_70837;
            struct memblock mem_70838 = ctx->constants->mem_70838;
            struct memblock mem_70839 = ctx->constants->mem_70839;
            struct memblock mem_70840 = ctx->constants->mem_70840;
            struct memblock mem_70841 = ctx->constants->mem_70841;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_72641;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_72642;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_72643;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_72644;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_72645;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_72646;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_72647;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_72648;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_72649;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_72650;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_72651;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_72652;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_72653;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_72654;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_72655;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_72656;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_72657;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_72658;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_72659;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_72660;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_72661;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_72662;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_72663;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_72664;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_72665;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_72666;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_72667;
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
    
    struct memblock mem_out_72649;
    
    mem_out_72649.references = NULL;
    
    struct memblock mem_out_72648;
    
    mem_out_72648.references = NULL;
    
    struct memblock mem_out_72647;
    
    mem_out_72647.references = NULL;
    
    struct memblock mem_out_72646;
    
    mem_out_72646.references = NULL;
    
    struct memblock mem_out_72645;
    
    mem_out_72645.references = NULL;
    
    struct memblock mem_out_72644;
    
    mem_out_72644.references = NULL;
    
    struct memblock mem_out_72643;
    
    mem_out_72643.references = NULL;
    
    struct memblock mem_out_72642;
    
    mem_out_72642.references = NULL;
    
    struct memblock mem_out_72641;
    
    mem_out_72641.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_72641, &mem_out_72642, &mem_out_72643, &mem_out_72644, &mem_out_72645, &mem_out_72646, &mem_out_72647, &mem_out_72648, &mem_out_72649);
        if (ret == 0) {
            struct memblock mem_70833 = ctx->constants->mem_70833;
            struct memblock mem_70834 = ctx->constants->mem_70834;
            struct memblock mem_70835 = ctx->constants->mem_70835;
            struct memblock mem_70836 = ctx->constants->mem_70836;
            struct memblock mem_70837 = ctx->constants->mem_70837;
            struct memblock mem_70838 = ctx->constants->mem_70838;
            struct memblock mem_70839 = ctx->constants->mem_70839;
            struct memblock mem_70840 = ctx->constants->mem_70840;
            struct memblock mem_70841 = ctx->constants->mem_70841;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_72641;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_72642;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_72643;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_72644;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_72645;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_72646;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_72647;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_72648;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_72649;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
