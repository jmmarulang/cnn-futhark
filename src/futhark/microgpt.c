
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
    struct memblock mem_87791;
    struct memblock mem_87792;
    struct memblock mem_87793;
    struct memblock mem_87794;
    struct memblock mem_87795;
    struct memblock mem_87796;
    struct memblock mem_87797;
    struct memblock mem_87798;
    struct memblock mem_87799;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10264(struct futhark_context *ctx, struct memblock *mem_out_p_89860, struct memblock *mem_out_p_89861, struct memblock *mem_out_p_89862, struct memblock w_mem_87800, struct memblock mw_mem_87801, struct memblock vw_mem_87802, struct memblock dw_mem_87803, int64_t n_62936, int64_t m_62937, int64_t step_62942, double lt_r_62943);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10265(struct futhark_context *ctx, struct memblock *mem_out_p_89865, struct memblock *mem_out_p_89866, struct memblock *mem_out_p_89867, struct memblock w_mem_87800, struct memblock mw_mem_87801, struct memblock vw_mem_87802, struct memblock dw_mem_87803, int64_t n_63969, int64_t m_63970, int64_t step_63975, double lt_r_63976);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_89870, struct memblock wdown_mem_87800, struct memblock wkey_mem_87801, struct memblock wout_mem_87802, struct memblock wpe_mem_87803, struct memblock wqry_mem_87804, struct memblock wte_mem_87805, struct memblock wup_mem_87806, struct memblock wval_mem_87807, struct memblock wvoc_mem_87808, struct memblock tokens_mem_87809, struct memblock mask_mem_87810);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_89924, struct memblock *mem_out_p_89925, struct memblock *mem_out_p_89926, struct memblock *mem_out_p_89927, struct memblock *mem_out_p_89928, struct memblock *mem_out_p_89929, struct memblock *mem_out_p_89930, struct memblock *mem_out_p_89931, struct memblock *mem_out_p_89932, struct memblock wte_mem_87800, struct memblock wpe_mem_87801, struct memblock wqry_mem_87802, struct memblock wkey_mem_87803, struct memblock wval_mem_87804, struct memblock wout_mem_87805, struct memblock wup_mem_87806, struct memblock wdown_mem_87807, struct memblock wvoc_mem_87808);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_89933, struct memblock *mem_out_p_89934, struct memblock *mem_out_p_89935, struct memblock *mem_out_p_89936, struct memblock *mem_out_p_89937, struct memblock *mem_out_p_89938, struct memblock *mem_out_p_89939, struct memblock *mem_out_p_89940, struct memblock *mem_out_p_89941, struct memblock *mem_out_p_89942, struct memblock *mem_out_p_89943, struct memblock *mem_out_p_89944, struct memblock *mem_out_p_89945, struct memblock *mem_out_p_89946, struct memblock *mem_out_p_89947, struct memblock *mem_out_p_89948, struct memblock *mem_out_p_89949, struct memblock *mem_out_p_89950, struct memblock *mem_out_p_89951, struct memblock *mem_out_p_89952, struct memblock *mem_out_p_89953, struct memblock *mem_out_p_89954, struct memblock *mem_out_p_89955, struct memblock *mem_out_p_89956, struct memblock *mem_out_p_89957, struct memblock *mem_out_p_89958, struct memblock *mem_out_p_89959, struct memblock wdown_mem_87800, struct memblock wkey_mem_87801, struct memblock wout_mem_87802, struct memblock wpe_mem_87803, struct memblock wqry_mem_87804, struct memblock wte_mem_87805, struct memblock wup_mem_87806, struct memblock wval_mem_87807, struct memblock wvoc_mem_87808, struct memblock wdown_mem_87809, struct memblock wkey_mem_87810, struct memblock wout_mem_87811, struct memblock wpe_mem_87812, struct memblock wqry_mem_87813, struct memblock wte_mem_87814, struct memblock wup_mem_87815, struct memblock wval_mem_87816, struct memblock wvoc_mem_87817, struct memblock wdown_mem_87818, struct memblock wkey_mem_87819, struct memblock wout_mem_87820, struct memblock wpe_mem_87821, struct memblock wqry_mem_87822, struct memblock wte_mem_87823, struct memblock wup_mem_87824, struct memblock wval_mem_87825, struct memblock wvoc_mem_87826, struct memblock masks_mem_87827, struct memblock dls_mem_87828, struct memblock seqs_mem_87829);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_90122, struct memblock *mem_out_p_90123, struct memblock *mem_out_p_90124, struct memblock *mem_out_p_90125, struct memblock *mem_out_p_90126, struct memblock *mem_out_p_90127, struct memblock *mem_out_p_90128, struct memblock *mem_out_p_90129, struct memblock *mem_out_p_90130);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_87791 (ctx->constants->mem_87791)
    #define mem_87792 (ctx->constants->mem_87792)
    #define mem_87793 (ctx->constants->mem_87793)
    #define mem_87794 (ctx->constants->mem_87794)
    #define mem_87795 (ctx->constants->mem_87795)
    #define mem_87796 (ctx->constants->mem_87796)
    #define mem_87797 (ctx->constants->mem_87797)
    #define mem_87798 (ctx->constants->mem_87798)
    #define mem_87799 (ctx->constants->mem_87799)
    mem_87791.references = NULL;
    mem_87792.references = NULL;
    mem_87793.references = NULL;
    mem_87794.references = NULL;
    mem_87795.references = NULL;
    mem_87796.references = NULL;
    mem_87797.references = NULL;
    mem_87798.references = NULL;
    mem_87799.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87791, (int64_t) 3456, "mem_87791")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89842 = 0; nest_i_89842 < (int64_t) 27; nest_i_89842++) {
        for (int64_t nest_i_89843 = 0; nest_i_89843 < (int64_t) 16; nest_i_89843++) {
            ((double *) mem_87791.mem)[nest_i_89842 * (int64_t) 16 + nest_i_89843] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87792, (int64_t) 2048, "mem_87792")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89844 = 0; nest_i_89844 < (int64_t) 16; nest_i_89844++) {
        for (int64_t nest_i_89845 = 0; nest_i_89845 < (int64_t) 16; nest_i_89845++) {
            ((double *) mem_87792.mem)[nest_i_89844 * (int64_t) 16 + nest_i_89845] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87793, (int64_t) 2048, "mem_87793")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89846 = 0; nest_i_89846 < (int64_t) 16; nest_i_89846++) {
        for (int64_t nest_i_89847 = 0; nest_i_89847 < (int64_t) 16; nest_i_89847++) {
            ((double *) mem_87793.mem)[nest_i_89846 * (int64_t) 16 + nest_i_89847] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87794, (int64_t) 2048, "mem_87794")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89848 = 0; nest_i_89848 < (int64_t) 16; nest_i_89848++) {
        for (int64_t nest_i_89849 = 0; nest_i_89849 < (int64_t) 16; nest_i_89849++) {
            ((double *) mem_87794.mem)[nest_i_89848 * (int64_t) 16 + nest_i_89849] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87795, (int64_t) 2048, "mem_87795")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89850 = 0; nest_i_89850 < (int64_t) 16; nest_i_89850++) {
        for (int64_t nest_i_89851 = 0; nest_i_89851 < (int64_t) 16; nest_i_89851++) {
            ((double *) mem_87795.mem)[nest_i_89850 * (int64_t) 16 + nest_i_89851] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87796, (int64_t) 2048, "mem_87796")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89852 = 0; nest_i_89852 < (int64_t) 16; nest_i_89852++) {
        for (int64_t nest_i_89853 = 0; nest_i_89853 < (int64_t) 16; nest_i_89853++) {
            ((double *) mem_87796.mem)[nest_i_89852 * (int64_t) 16 + nest_i_89853] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87797, (int64_t) 8192, "mem_87797")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89854 = 0; nest_i_89854 < (int64_t) 64; nest_i_89854++) {
        for (int64_t nest_i_89855 = 0; nest_i_89855 < (int64_t) 16; nest_i_89855++) {
            ((double *) mem_87797.mem)[nest_i_89854 * (int64_t) 16 + nest_i_89855] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87798, (int64_t) 8192, "mem_87798")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89856 = 0; nest_i_89856 < (int64_t) 16; nest_i_89856++) {
        for (int64_t nest_i_89857 = 0; nest_i_89857 < (int64_t) 64; nest_i_89857++) {
            ((double *) mem_87798.mem)[nest_i_89856 * (int64_t) 64 + nest_i_89857] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87799, (int64_t) 3456, "mem_87799")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_89858 = 0; nest_i_89858 < (int64_t) 27; nest_i_89858++) {
        for (int64_t nest_i_89859 = 0; nest_i_89859 < (int64_t) 16; nest_i_89859++) {
            ((double *) mem_87799.mem)[nest_i_89858 * (int64_t) 16 + nest_i_89859] = 0.0;
        }
    }
    #undef mem_87791
    #undef mem_87792
    #undef mem_87793
    #undef mem_87794
    #undef mem_87795
    #undef mem_87796
    #undef mem_87797
    #undef mem_87798
    #undef mem_87799
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_87791, "ctx->constants->mem_87791") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87792, "ctx->constants->mem_87792") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87793, "ctx->constants->mem_87793") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87794, "ctx->constants->mem_87794") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87795, "ctx->constants->mem_87795") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87796, "ctx->constants->mem_87796") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87797, "ctx->constants->mem_87797") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87798, "ctx->constants->mem_87798") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_87799, "ctx->constants->mem_87799") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10264(struct futhark_context *ctx, struct memblock *mem_out_p_89860, struct memblock *mem_out_p_89861, struct memblock *mem_out_p_89862, struct memblock w_mem_87800, struct memblock mw_mem_87801, struct memblock vw_mem_87802, struct memblock dw_mem_87803, int64_t n_62936, int64_t m_62937, int64_t step_62942, double lt_r_62943)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_87844_cached_sizze_89863 = 0;
    unsigned char *mem_87844 = NULL;
    int64_t mem_87847_cached_sizze_89864 = 0;
    unsigned char *mem_87847 = NULL;
    struct memblock mem_87882;
    
    mem_87882.references = NULL;
    
    struct memblock mem_87809;
    
    mem_87809.references = NULL;
    
    struct memblock mem_87806;
    
    mem_87806.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_87804 = (int64_t) 8 * n_62936;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_87805 = m_62937 * binop_x_87804;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87806, bytes_87805, "mem_87806")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87809, bytes_87805, "mem_87809")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86975 = 0; i_86975 < n_62936; i_86975++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86968 = 0; i_86968 < m_62937; i_86968++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83072 = ((double *) mw_mem_87801.mem)[i_86975 * m_62937 + i_86968];
            
            // futhark/microgpt.fut:395:10-20
            
            double zp_lhs_83073 = 0.85 * zt_rhs_83072;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83074 = ((double *) dw_mem_87803.mem)[i_86975 * m_62937 + i_86968];
            
            // futhark/microgpt.fut:395:35-45
            
            double zp_rhs_83075 = 0.15000000000000002 * zt_rhs_83074;
            
            // futhark/microgpt.fut:395:21-45
            
            double lifted_lambda_res_83076 = zp_lhs_83073 + zp_rhs_83075;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83083 = ((double *) vw_mem_87802.mem)[i_86975 * m_62937 + i_86968];
            
            // futhark/microgpt.fut:397:10-20
            
            double zp_lhs_83084 = 0.99 * zt_rhs_83083;
            
            // futhark/microgpt.fut:397:35-45
            
            double zt_lhs_83086 = 1.0000000000000009e-2 * zt_rhs_83074;
            
            // futhark/microgpt.fut:397:46-56
            
            double zp_rhs_83087 = zt_rhs_83074 * zt_lhs_83086;
            
            // futhark/microgpt.fut:397:21-56
            
            double lifted_lambda_res_83088 = zp_lhs_83084 + zp_rhs_83087;
            
            ((double *) mem_87806.mem)[i_86975 * m_62937 + i_86968] = lifted_lambda_res_83088;
            ((double *) mem_87809.mem)[i_86975 * m_62937 + i_86968] = lifted_lambda_res_83076;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67995 = sitofp_i64_f64(step_62942);
    
    // futhark/microgpt.fut:399:54-57
    
    double ztzt_rhs_67996 = 1.0 + i64_res_67995;
    
    // futhark/microgpt.fut:399:30-57
    
    double zm_rhs_67997 = fpow64(0.85, ztzt_rhs_67996);
    
    // futhark/microgpt.fut:399:23-57
    
    double zs_rhs_67998 = 1.0 - zm_rhs_67997;
    
    // futhark/microgpt.fut:401:31-58
    
    double zm_rhs_68036 = fpow64(0.99, ztzt_rhs_67996);
    
    // futhark/microgpt.fut:401:23-58
    
    double zs_rhs_68037 = 1.0 - zm_rhs_68036;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_87844_cached_sizze_89863 < bytes_87805) {
        err = lexical_realloc(ctx, &mem_87844, &mem_87844_cached_sizze_89863, bytes_87805);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87847_cached_sizze_89864 < bytes_87805) {
        err = lexical_realloc(ctx, &mem_87847, &mem_87847_cached_sizze_89864, bytes_87805);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86989 = 0; i_86989 < n_62936; i_86989++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86982 = 0; i_86982 < m_62937; i_86982++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83108 = ((double *) mem_87809.mem)[i_86989 * m_62937 + i_86982];
            
            // futhark/microgpt.fut:399:18-57
            
            double lifted_lambda_res_83109 = zs_lhs_83108 / zs_rhs_67998;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83116 = ((double *) mem_87806.mem)[i_86989 * m_62937 + i_86982];
            
            // futhark/microgpt.fut:401:18-58
            
            double lifted_lambda_res_83117 = zs_lhs_83116 / zs_rhs_68037;
            
            ((double *) mem_87844)[i_86989 * m_62937 + i_86982] = lifted_lambda_res_83117;
            ((double *) mem_87847)[i_86989 * m_62937 + i_86982] = lifted_lambda_res_83109;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87882, bytes_87805, "mem_87882")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86998 = 0; i_86998 < n_62936; i_86998++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86994 = 0; i_86994 < m_62937; i_86994++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_67114 = ((double *) w_mem_87800.mem)[i_86998 * m_62937 + i_86994];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_67115 = ((double *) mem_87847)[i_86998 * m_62937 + i_86994];
            
            // futhark/microgpt.fut:403:21-34
            
            double zs_lhs_67116 = lt_r_62943 * zt_rhs_67115;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_67117 = ((double *) mem_87844)[i_86998 * m_62937 + i_86994];
            
            // futhark/microgpt.fut:403:51-57
            
            double zp_lhs_67118 = fpow64(ztzt_lhs_67117, 0.5);
            
            // futhark/microgpt.fut:403:59-71
            
            double zs_rhs_67119 = 1.0e-8 + zp_lhs_67118;
            
            // futhark/microgpt.fut:403:35-71
            
            double zm_rhs_67120 = zs_lhs_67116 / zs_rhs_67119;
            
            // futhark/microgpt.fut:403:13-71
            
            double lifted_lambda_res_67121 = zm_lhs_67114 - zm_rhs_67120;
            
            ((double *) mem_87882.mem)[i_86998 * m_62937 + i_86994] = lifted_lambda_res_67121;
        }
    }
    if (memblock_set(ctx, &mem_out_89540, &mem_87882, "mem_87882") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89541, &mem_87809, "mem_87809") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89542, &mem_87806, "mem_87806") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89860, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89861, &mem_out_89541, "mem_out_89541") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89862, &mem_out_89542, "mem_out_89542") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_87844);
        free(mem_87847);
        if (memblock_unref(ctx, &mem_87882, "mem_87882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_87809, "mem_87809") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_87806, "mem_87806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89542, "mem_out_89542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89541, "mem_out_89541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10265(struct futhark_context *ctx, struct memblock *mem_out_p_89865, struct memblock *mem_out_p_89866, struct memblock *mem_out_p_89867, struct memblock w_mem_87800, struct memblock mw_mem_87801, struct memblock vw_mem_87802, struct memblock dw_mem_87803, int64_t n_63969, int64_t m_63970, int64_t step_63975, double lt_r_63976)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_87844_cached_sizze_89868 = 0;
    unsigned char *mem_87844 = NULL;
    int64_t mem_87847_cached_sizze_89869 = 0;
    unsigned char *mem_87847 = NULL;
    struct memblock mem_87882;
    
    mem_87882.references = NULL;
    
    struct memblock mem_87809;
    
    mem_87809.references = NULL;
    
    struct memblock mem_87806;
    
    mem_87806.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_87804 = (int64_t) 8 * n_63969;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_87805 = m_63970 * binop_x_87804;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87806, bytes_87805, "mem_87806")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87809, bytes_87805, "mem_87809")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86975 = 0; i_86975 < n_63969; i_86975++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86968 = 0; i_86968 < m_63970; i_86968++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83072 = ((double *) mw_mem_87801.mem)[i_86975 * m_63970 + i_86968];
            
            // futhark/microgpt.fut:395:10-20
            
            double zp_lhs_83073 = 0.85 * zt_rhs_83072;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83074 = ((double *) dw_mem_87803.mem)[i_86975 * m_63970 + i_86968];
            
            // futhark/microgpt.fut:395:35-45
            
            double zp_rhs_83075 = 0.15000000000000002 * zt_rhs_83074;
            
            // futhark/microgpt.fut:395:21-45
            
            double lifted_lambda_res_83076 = zp_lhs_83073 + zp_rhs_83075;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_83083 = ((double *) vw_mem_87802.mem)[i_86975 * m_63970 + i_86968];
            
            // futhark/microgpt.fut:397:10-20
            
            double zp_lhs_83084 = 0.99 * zt_rhs_83083;
            
            // futhark/microgpt.fut:397:35-45
            
            double zt_lhs_83086 = 1.0000000000000009e-2 * zt_rhs_83074;
            
            // futhark/microgpt.fut:397:46-56
            
            double zp_rhs_83087 = zt_rhs_83074 * zt_lhs_83086;
            
            // futhark/microgpt.fut:397:21-56
            
            double lifted_lambda_res_83088 = zp_lhs_83084 + zp_rhs_83087;
            
            ((double *) mem_87806.mem)[i_86975 * m_63970 + i_86968] = lifted_lambda_res_83088;
            ((double *) mem_87809.mem)[i_86975 * m_63970 + i_86968] = lifted_lambda_res_83076;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_67995 = sitofp_i64_f64(step_63975);
    
    // futhark/microgpt.fut:399:54-57
    
    double ztzt_rhs_67996 = 1.0 + i64_res_67995;
    
    // futhark/microgpt.fut:399:30-57
    
    double zm_rhs_67997 = fpow64(0.85, ztzt_rhs_67996);
    
    // futhark/microgpt.fut:399:23-57
    
    double zs_rhs_67998 = 1.0 - zm_rhs_67997;
    
    // futhark/microgpt.fut:401:31-58
    
    double zm_rhs_68036 = fpow64(0.99, ztzt_rhs_67996);
    
    // futhark/microgpt.fut:401:23-58
    
    double zs_rhs_68037 = 1.0 - zm_rhs_68036;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_87844_cached_sizze_89868 < bytes_87805) {
        err = lexical_realloc(ctx, &mem_87844, &mem_87844_cached_sizze_89868, bytes_87805);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87847_cached_sizze_89869 < bytes_87805) {
        err = lexical_realloc(ctx, &mem_87847, &mem_87847_cached_sizze_89869, bytes_87805);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86989 = 0; i_86989 < n_63969; i_86989++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86982 = 0; i_86982 < m_63970; i_86982++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83108 = ((double *) mem_87809.mem)[i_86989 * m_63970 + i_86982];
            
            // futhark/microgpt.fut:399:18-57
            
            double lifted_lambda_res_83109 = zs_lhs_83108 / zs_rhs_67998;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_83116 = ((double *) mem_87806.mem)[i_86989 * m_63970 + i_86982];
            
            // futhark/microgpt.fut:401:18-58
            
            double lifted_lambda_res_83117 = zs_lhs_83116 / zs_rhs_68037;
            
            ((double *) mem_87844)[i_86989 * m_63970 + i_86982] = lifted_lambda_res_83117;
            ((double *) mem_87847)[i_86989 * m_63970 + i_86982] = lifted_lambda_res_83109;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_87882, bytes_87805, "mem_87882")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86998 = 0; i_86998 < n_63969; i_86998++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86994 = 0; i_86994 < m_63970; i_86994++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_67114 = ((double *) w_mem_87800.mem)[i_86998 * m_63970 + i_86994];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_67115 = ((double *) mem_87847)[i_86998 * m_63970 + i_86994];
            
            // futhark/microgpt.fut:403:21-34
            
            double zs_lhs_67116 = lt_r_63976 * zt_rhs_67115;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_67117 = ((double *) mem_87844)[i_86998 * m_63970 + i_86994];
            
            // futhark/microgpt.fut:403:51-57
            
            double zp_lhs_67118 = fpow64(ztzt_lhs_67117, 0.5);
            
            // futhark/microgpt.fut:403:59-71
            
            double zs_rhs_67119 = 1.0e-8 + zp_lhs_67118;
            
            // futhark/microgpt.fut:403:35-71
            
            double zm_rhs_67120 = zs_lhs_67116 / zs_rhs_67119;
            
            // futhark/microgpt.fut:403:13-71
            
            double lifted_lambda_res_67121 = zm_lhs_67114 - zm_rhs_67120;
            
            ((double *) mem_87882.mem)[i_86998 * m_63970 + i_86994] = lifted_lambda_res_67121;
        }
    }
    if (memblock_set(ctx, &mem_out_89540, &mem_87882, "mem_87882") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89541, &mem_87809, "mem_87809") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89542, &mem_87806, "mem_87806") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89865, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89866, &mem_out_89541, "mem_out_89541") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89867, &mem_out_89542, "mem_out_89542") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_87844);
        free(mem_87847);
        if (memblock_unref(ctx, &mem_87882, "mem_87882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_87809, "mem_87809") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_87806, "mem_87806") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89542, "mem_out_89542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89541, "mem_out_89541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_89870, struct memblock wdown_mem_87800, struct memblock wkey_mem_87801, struct memblock wout_mem_87802, struct memblock wpe_mem_87803, struct memblock wqry_mem_87804, struct memblock wte_mem_87805, struct memblock wup_mem_87806, struct memblock wval_mem_87807, struct memblock wvoc_mem_87808, struct memblock tokens_mem_87809, struct memblock mask_mem_87810)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_87811_cached_sizze_89871 = 0;
    unsigned char *mem_87811 = NULL;
    int64_t mem_87816_cached_sizze_89872 = 0;
    unsigned char *mem_87816 = NULL;
    int64_t mem_87827_cached_sizze_89873 = 0;
    unsigned char *mem_87827 = NULL;
    int64_t mem_87832_cached_sizze_89874 = 0;
    unsigned char *mem_87832 = NULL;
    int64_t mem_87843_cached_sizze_89875 = 0;
    unsigned char *mem_87843 = NULL;
    int64_t mem_87848_cached_sizze_89876 = 0;
    unsigned char *mem_87848 = NULL;
    int64_t mem_87855_cached_sizze_89877 = 0;
    unsigned char *mem_87855 = NULL;
    int64_t mem_87866_cached_sizze_89878 = 0;
    unsigned char *mem_87866 = NULL;
    int64_t mem_87871_cached_sizze_89879 = 0;
    unsigned char *mem_87871 = NULL;
    int64_t mem_87878_cached_sizze_89880 = 0;
    unsigned char *mem_87878 = NULL;
    int64_t mem_87889_cached_sizze_89881 = 0;
    unsigned char *mem_87889 = NULL;
    int64_t mem_87890_cached_sizze_89882 = 0;
    unsigned char *mem_87890 = NULL;
    int64_t mem_87891_cached_sizze_89883 = 0;
    unsigned char *mem_87891 = NULL;
    int64_t mem_87904_cached_sizze_89884 = 0;
    unsigned char *mem_87904 = NULL;
    int64_t mem_87905_cached_sizze_89885 = 0;
    unsigned char *mem_87905 = NULL;
    int64_t mem_87906_cached_sizze_89886 = 0;
    unsigned char *mem_87906 = NULL;
    int64_t mem_87937_cached_sizze_89887 = 0;
    unsigned char *mem_87937 = NULL;
    int64_t mem_87938_cached_sizze_89888 = 0;
    unsigned char *mem_87938 = NULL;
    int64_t mem_87939_cached_sizze_89889 = 0;
    unsigned char *mem_87939 = NULL;
    int64_t mem_87955_cached_sizze_89890 = 0;
    unsigned char *mem_87955 = NULL;
    int64_t mem_87956_cached_sizze_89891 = 0;
    unsigned char *mem_87956 = NULL;
    int64_t mem_87957_cached_sizze_89892 = 0;
    unsigned char *mem_87957 = NULL;
    int64_t mem_87970_cached_sizze_89893 = 0;
    unsigned char *mem_87970 = NULL;
    int64_t mem_87971_cached_sizze_89894 = 0;
    unsigned char *mem_87971 = NULL;
    int64_t mem_87972_cached_sizze_89895 = 0;
    unsigned char *mem_87972 = NULL;
    int64_t mem_88018_cached_sizze_89896 = 0;
    unsigned char *mem_88018 = NULL;
    int64_t mem_88024_cached_sizze_89897 = 0;
    unsigned char *mem_88024 = NULL;
    int64_t mem_88029_cached_sizze_89898 = 0;
    unsigned char *mem_88029 = NULL;
    int64_t mem_88040_cached_sizze_89899 = 0;
    unsigned char *mem_88040 = NULL;
    int64_t mem_88045_cached_sizze_89900 = 0;
    unsigned char *mem_88045 = NULL;
    int64_t mem_88056_cached_sizze_89901 = 0;
    unsigned char *mem_88056 = NULL;
    int64_t mem_88061_cached_sizze_89902 = 0;
    unsigned char *mem_88061 = NULL;
    int64_t mem_88068_cached_sizze_89903 = 0;
    unsigned char *mem_88068 = NULL;
    int64_t mem_88079_cached_sizze_89904 = 0;
    unsigned char *mem_88079 = NULL;
    int64_t mem_88084_cached_sizze_89905 = 0;
    unsigned char *mem_88084 = NULL;
    int64_t mem_88100_cached_sizze_89906 = 0;
    unsigned char *mem_88100 = NULL;
    int64_t mem_88105_cached_sizze_89907 = 0;
    unsigned char *mem_88105 = NULL;
    int64_t mem_88116_cached_sizze_89908 = 0;
    unsigned char *mem_88116 = NULL;
    int64_t mem_88121_cached_sizze_89909 = 0;
    unsigned char *mem_88121 = NULL;
    int64_t mem_88132_cached_sizze_89910 = 0;
    unsigned char *mem_88132 = NULL;
    int64_t mem_88137_cached_sizze_89911 = 0;
    unsigned char *mem_88137 = NULL;
    int64_t mem_88148_cached_sizze_89912 = 0;
    unsigned char *mem_88148 = NULL;
    int64_t mem_88153_cached_sizze_89913 = 0;
    unsigned char *mem_88153 = NULL;
    int64_t mem_88160_cached_sizze_89914 = 0;
    unsigned char *mem_88160 = NULL;
    int64_t mem_88171_cached_sizze_89915 = 0;
    unsigned char *mem_88171 = NULL;
    int64_t mem_88176_cached_sizze_89916 = 0;
    unsigned char *mem_88176 = NULL;
    int64_t mem_88187_cached_sizze_89917 = 0;
    unsigned char *mem_88187 = NULL;
    int64_t mem_88192_cached_sizze_89918 = 0;
    unsigned char *mem_88192 = NULL;
    int64_t mem_88203_cached_sizze_89919 = 0;
    unsigned char *mem_88203 = NULL;
    int64_t mem_88208_cached_sizze_89920 = 0;
    unsigned char *mem_88208 = NULL;
    int64_t mem_88219_cached_sizze_89921 = 0;
    unsigned char *mem_88219 = NULL;
    int64_t mem_88224_cached_sizze_89922 = 0;
    unsigned char *mem_88224 = NULL;
    int64_t mem_88240_cached_sizze_89923 = 0;
    unsigned char *mem_88240 = NULL;
    struct memblock mem_88235;
    
    mem_88235.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_87811_cached_sizze_89871 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87811, &mem_87811_cached_sizze_89871, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87816_cached_sizze_89872 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87816, &mem_87816_cached_sizze_89872, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86970 = 0; i_86970 < (int64_t) 16; i_86970++) {
        // futhark/microgpt.fut:380:41-50
        
        int64_t tmp_77409 = ((int64_t *) tokens_mem_87809.mem)[i_86970];
        
        // futhark/microgpt.fut:380:37-51
        
        bool x_77410 = sle64((int64_t) 0, tmp_77409);
        
        // futhark/microgpt.fut:380:37-51
        
        bool y_77411 = slt64(tmp_77409, (int64_t) 27);
        
        // futhark/microgpt.fut:380:37-51
        
        bool bounds_check_77412 = x_77410 && y_77411;
        
        // futhark/microgpt.fut:380:37-51
        
        bool index_certs_77413;
        
        if (!bounds_check_77412) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_77409, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:380:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:380:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86966 = 0; i_86966 < (int64_t) 16; i_86966++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_77420 = ((double *) wte_mem_87805.mem)[tmp_77409 * (int64_t) 16 + i_86966];
            
            ((double *) mem_87816)[i_86966] = lifted_lambda_res_77420;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87811, i_86970 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87816, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87827_cached_sizze_89873 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87827, &mem_87827_cached_sizze_89873, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87832_cached_sizze_89874 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87832, &mem_87832_cached_sizze_89874, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86978 = 0; i_86978 < (int64_t) 16; i_86978++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86974 = 0; i_86974 < (int64_t) 16; i_86974++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77452 = ((double *) wpe_mem_87803.mem)[i_86978 * (int64_t) 16 + i_86974];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77453 = ((double *) mem_87811)[i_86978 * (int64_t) 16 + i_86974];
            
            // futhark/microgpt.fut:142:42-82
            
            double zp_res_77454 = zp_lhs_77452 + zp_rhs_77453;
            
            ((double *) mem_87832)[i_86974] = zp_res_77454;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87827, i_86978 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87832, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87843_cached_sizze_89875 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87843, &mem_87843_cached_sizze_89875, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87848_cached_sizze_89876 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87848, &mem_87848_cached_sizze_89876, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87855_cached_sizze_89877 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87855, &mem_87855_cached_sizze_89877, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_86990 = 0; i_86990 < (int64_t) 16; i_86990++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86982 = 0; i_86982 < (int64_t) 16; i_86982++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77469 = ((double *) mem_87827)[i_86990 * (int64_t) 16 + i_86982];
            
            // futhark/microgpt.fut:143:77-114
            
            double zt_res_77470 = zt_lhs_77469 * zt_lhs_77469;
            
            ((double *) mem_87848)[i_86982] = zt_res_77470;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_77472;
        double r_77474 = 0.0;
        
        for (int64_t i_77473 = 0; i_77473 < (int64_t) 16; i_77473++) {
            // futhark/microgpt.fut:144:37-47
            
            double lifted_lambda_res_77475 = ((double *) mem_87848)[i_77473];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_77476 = r_77474 + lifted_lambda_res_77475;
            double r_tmp_89547 = zp_res_77476;
            
            r_77474 = r_tmp_89547;
        }
        defunc_0_lifted_lambda_res_77472 = r_77474;
        // futhark/microgpt.fut:144:17-64
        
        double zs_res_77477 = defunc_0_lifted_lambda_res_77472 / 16.0;
        
        // futhark/microgpt.fut:145:24-55
        
        double zp_res_77478 = 1.0e-5 + zs_res_77477;
        
        // futhark/microgpt.fut:145:16-55
        
        double sqrt_res_77479 = futrts_sqrt64(zp_res_77478);
        
        // futhark/microgpt.fut:146:27-38
        
        double zs_res_77480 = 1.0 / sqrt_res_77479;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86986 = 0; i_86986 < (int64_t) 16; i_86986++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77487 = ((double *) mem_87827)[i_86990 * (int64_t) 16 + i_86986];
            
            // futhark/microgpt.fut:146:5-38
            
            double zt_res_77488 = zs_res_77480 * zt_lhs_77487;
            
            ((double *) mem_87855)[i_86986] = zt_res_77488;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87843, i_86990 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87855, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87866_cached_sizze_89878 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87866, &mem_87866_cached_sizze_89878, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87871_cached_sizze_89879 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87871, &mem_87871_cached_sizze_89879, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87878_cached_sizze_89880 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87878, &mem_87878_cached_sizze_89880, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87002 = 0; i_87002 < (int64_t) 16; i_87002++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86994 = 0; i_86994 < (int64_t) 16; i_86994++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77503 = ((double *) mem_87843)[i_87002 * (int64_t) 16 + i_86994];
            
            // futhark/microgpt.fut:147:77-114
            
            double zt_res_77504 = zt_lhs_77503 * zt_lhs_77503;
            
            ((double *) mem_87871)[i_86994] = zt_res_77504;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_77506;
        double r_77508 = 0.0;
        
        for (int64_t i_77507 = 0; i_77507 < (int64_t) 16; i_77507++) {
            // futhark/microgpt.fut:148:37-47
            
            double lifted_lambda_res_77509 = ((double *) mem_87871)[i_77507];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_77510 = r_77508 + lifted_lambda_res_77509;
            double r_tmp_89551 = zp_res_77510;
            
            r_77508 = r_tmp_89551;
        }
        defunc_0_lifted_lambda_res_77506 = r_77508;
        // futhark/microgpt.fut:148:17-64
        
        double zs_res_77511 = defunc_0_lifted_lambda_res_77506 / 16.0;
        
        // futhark/microgpt.fut:149:24-55
        
        double zp_res_77512 = 1.0e-5 + zs_res_77511;
        
        // futhark/microgpt.fut:149:16-55
        
        double sqrt_res_77513 = futrts_sqrt64(zp_res_77512);
        
        // futhark/microgpt.fut:150:27-38
        
        double zs_res_77514 = 1.0 / sqrt_res_77513;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86998 = 0; i_86998 < (int64_t) 16; i_86998++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77521 = ((double *) mem_87843)[i_87002 * (int64_t) 16 + i_86998];
            
            // futhark/microgpt.fut:150:5-38
            
            double zt_res_77522 = zs_res_77514 * zt_lhs_77521;
            
            ((double *) mem_87878)[i_86998] = zt_res_77522;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87866, i_87002 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87878, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87889_cached_sizze_89881 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87889, &mem_87889_cached_sizze_89881, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87890_cached_sizze_89882 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87890, &mem_87890_cached_sizze_89882, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87891_cached_sizze_89883 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87891, &mem_87891_cached_sizze_89883, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87904_cached_sizze_89884 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87904, &mem_87904_cached_sizze_89884, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87905_cached_sizze_89885 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87905, &mem_87905_cached_sizze_89885, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87906_cached_sizze_89886 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87906, &mem_87906_cached_sizze_89886, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87020 = 0; i_87020 < (int64_t) 16; i_87020++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87010 = 0; i_87010 < (int64_t) 16; i_87010++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83291;
            double r_83293 = 0.0;
            
            for (int64_t i_83292 = 0; i_83292 < (int64_t) 16; i_83292++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83294 = ((double *) wqry_mem_87804.mem)[i_87010 * (int64_t) 16 + i_83292];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_83295 = ((double *) mem_87866)[i_87020 * (int64_t) 16 + i_83292];
                
                // futhark/microgpt.fut:151:66-105
                
                double zt_res_83296 = zt_lhs_83294 * zt_rhs_83295;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83297 = r_83293 + zt_res_83296;
                double r_tmp_89559 = zp_res_83297;
                
                r_83293 = r_tmp_89559;
            }
            defunc_0_lifted_lambda_res_83291 = r_83293;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83304;
            double r_83306 = 0.0;
            
            for (int64_t i_83305 = 0; i_83305 < (int64_t) 16; i_83305++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83307 = ((double *) wkey_mem_87801.mem)[i_87010 * (int64_t) 16 + i_83305];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_83308 = ((double *) mem_87866)[i_87020 * (int64_t) 16 + i_83305];
                
                // futhark/microgpt.fut:152:66-105
                
                double zt_res_83309 = zt_lhs_83307 * zt_rhs_83308;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83310 = r_83306 + zt_res_83309;
                double r_tmp_89560 = zp_res_83310;
                
                r_83306 = r_tmp_89560;
            }
            defunc_0_lifted_lambda_res_83304 = r_83306;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83320;
            double r_83322 = 0.0;
            
            for (int64_t i_83321 = 0; i_83321 < (int64_t) 16; i_83321++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83323 = ((double *) wval_mem_87807.mem)[i_87010 * (int64_t) 16 + i_83321];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_83324 = ((double *) mem_87866)[i_87020 * (int64_t) 16 + i_83321];
                
                // futhark/microgpt.fut:153:66-105
                
                double zt_res_83325 = zt_lhs_83323 * zt_rhs_83324;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83326 = r_83322 + zt_res_83325;
                double r_tmp_89561 = zp_res_83326;
                
                r_83322 = r_tmp_89561;
            }
            defunc_0_lifted_lambda_res_83320 = r_83322;
            ((double *) mem_87904)[i_87010] = defunc_0_lifted_lambda_res_83320;
            ((double *) mem_87905)[i_87010] = defunc_0_lifted_lambda_res_83304;
            ((double *) mem_87906)[i_87010] = defunc_0_lifted_lambda_res_83291;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87889, i_87020 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87904, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87890, i_87020 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87905, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_87891, i_87020 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87906, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87937_cached_sizze_89887 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87937, &mem_87937_cached_sizze_89887, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87938_cached_sizze_89888 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87938, &mem_87938_cached_sizze_89888, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87939_cached_sizze_89889 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87939, &mem_87939_cached_sizze_89889, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87955_cached_sizze_89890 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87955, &mem_87955_cached_sizze_89890, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87956_cached_sizze_89891 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87956, &mem_87956_cached_sizze_89891, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87957_cached_sizze_89892 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_87957, &mem_87957_cached_sizze_89892, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87970_cached_sizze_89893 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87970, &mem_87970_cached_sizze_89893, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87971_cached_sizze_89894 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87971, &mem_87971_cached_sizze_89894, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87972_cached_sizze_89895 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_87972, &mem_87972_cached_sizze_89895, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87050 = 0; i_87050 < (int64_t) 4; i_87050++) {
        // futhark/microgpt.fut:154:69-72
        
        int64_t zp_lhs_83166 = mul64((int64_t) 4, i_87050);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87040 = 0; i_87040 < (int64_t) 16; i_87040++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87030 = 0; i_87030 < (int64_t) 4; i_87030++) {
                // futhark/microgpt.fut:154:74-81
                
                int64_t tmp_83484 = add64(zp_lhs_83166, i_87030);
                
                // futhark/microgpt.fut:154:51-83
                
                bool x_83485 = sle64((int64_t) 0, tmp_83484);
                
                // futhark/microgpt.fut:154:51-83
                
                bool y_83486 = slt64(tmp_83484, (int64_t) 16);
                
                // futhark/microgpt.fut:154:51-83
                
                bool bounds_check_83487 = x_83485 && y_83486;
                
                // futhark/microgpt.fut:154:51-83
                
                bool index_certs_83488;
                
                if (!bounds_check_83487) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_83484, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:154:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:154:15-84\n   #9  futhark/microgpt.fut:381:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83489 = ((double *) mem_87891)[i_87040 * (int64_t) 16 + tmp_83484];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83497 = ((double *) mem_87890)[i_87040 * (int64_t) 16 + tmp_83484];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83508 = ((double *) mem_87889)[i_87040 * (int64_t) 16 + tmp_83484];
                
                ((double *) mem_87970)[i_87030] = lifted_lambda_res_83508;
                ((double *) mem_87971)[i_87030] = lifted_lambda_res_83497;
                ((double *) mem_87972)[i_87030] = lifted_lambda_res_83489;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87955, i_87040 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87970, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87956, i_87040 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87971, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87957, i_87040 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_87937, i_87050 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87955, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_87938, i_87050 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87956, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_87939, i_87050 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_87957, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88018_cached_sizze_89896 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88018, &mem_88018_cached_sizze_89896, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88024_cached_sizze_89897 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88024, &mem_88024_cached_sizze_89897, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88029_cached_sizze_89898 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88029, &mem_88029_cached_sizze_89898, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88040_cached_sizze_89899 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88040, &mem_88040_cached_sizze_89899, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88045_cached_sizze_89900 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88045, &mem_88045_cached_sizze_89900, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88056_cached_sizze_89901 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88056, &mem_88056_cached_sizze_89901, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88061_cached_sizze_89902 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88061, &mem_88061_cached_sizze_89902, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88068_cached_sizze_89903 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88068, &mem_88068_cached_sizze_89903, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88079_cached_sizze_89904 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88079, &mem_88079_cached_sizze_89904, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88084_cached_sizze_89905 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88084, &mem_88084_cached_sizze_89905, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87094 = 0; i_87094 < (int64_t) 4; i_87094++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87060 = 0; i_87060 < (int64_t) 16; i_87060++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87056 = 0; i_87056 < (int64_t) 16; i_87056++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77667;
                double r_77669 = 0.0;
                
                for (int64_t i_77668 = 0; i_77668 < (int64_t) 4; i_77668++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77670 = ((double *) mem_87939)[i_87094 * (int64_t) 64 + i_87060 * (int64_t) 4 + i_77668];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77671 = ((double *) mem_87938)[i_87094 * (int64_t) 64 + i_87056 * (int64_t) 4 + i_77668];
                    
                    // futhark/microgpt.fut:157:113-164
                    
                    double zt_res_77672 = zt_lhs_77670 * zt_rhs_77671;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77673 = r_77669 + zt_res_77672;
                    double r_tmp_89574 = zp_res_77673;
                    
                    r_77669 = r_tmp_89574;
                }
                defunc_0_lifted_lambda_res_77667 = r_77669;
                ((double *) mem_88029)[i_87056] = defunc_0_lifted_lambda_res_77667;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88024, i_87060 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88029, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87068 = 0; i_87068 < (int64_t) 16; i_87068++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87064 = 0; i_87064 < (int64_t) 16; i_87064++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_77688 = ((double *) mem_88024)[i_87068 * (int64_t) 16 + i_87064];
                
                // futhark/microgpt.fut:158:47-78
                
                double zs_res_77689 = zs_lhs_77688 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_77690 = ((double *) mask_mem_87810.mem)[i_87068 * (int64_t) 16 + i_87064];
                
                // futhark/microgpt.fut:158:65-102
                
                double zp_res_77691 = zs_res_77689 + zp_rhs_77690;
                
                ((double *) mem_88045)[i_87064] = zp_res_77691;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88040, i_87068 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88045, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87082 = 0; i_87082 < (int64_t) 16; i_87082++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_83581;
            double redout_87070 = -INFINITY;
            
            for (int64_t i_87071 = 0; i_87071 < (int64_t) 16; i_87071++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83535 = ((double *) mem_88040)[i_87082 * (int64_t) 16 + i_87071];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_77712 = fmax64(lifted_lambda_res_83535, redout_87070);
                double redout_tmp_89578 = max_res_77712;
                
                redout_87070 = redout_tmp_89578;
            }
            defunc_0_reduce_res_83581 = redout_87070;
            // futhark/microgpt.fut:160:65-74
            
            double neg_res_77713 = -defunc_0_reduce_res_83581;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87074 = 0; i_87074 < (int64_t) 16; i_87074++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_77720 = ((double *) mem_88040)[i_87082 * (int64_t) 16 + i_87074];
                
                // futhark/microgpt.fut:160:43-74
                
                double zp_res_77721 = neg_res_77713 + zp_lhs_77720;
                
                // futhark/microgpt.fut:160:36-74
                
                double exp_res_77722 = futrts_exp64(zp_res_77721);
                
                ((double *) mem_88061)[i_87074] = exp_res_77722;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77724;
            double r_77726 = 0.0;
            
            for (int64_t i_77725 = 0; i_77725 < (int64_t) 16; i_77725++) {
                // futhark/microgpt.fut:161:36-46
                
                double lifted_lambda_res_77727 = ((double *) mem_88061)[i_77725];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77728 = r_77726 + lifted_lambda_res_77727;
                double r_tmp_89580 = zp_res_77728;
                
                r_77726 = r_tmp_89580;
            }
            defunc_0_lifted_lambda_res_77724 = r_77726;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87078 = 0; i_87078 < (int64_t) 16; i_87078++) {
                // futhark/microgpt.fut:162:5-15
                
                double zs_lhs_77735 = ((double *) mem_88061)[i_87078];
                
                // futhark/microgpt.fut:162:5-23
                
                double zs_res_77736 = zs_lhs_77735 / defunc_0_lifted_lambda_res_77724;
                
                ((double *) mem_88068)[i_87078] = zs_res_77736;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88056, i_87082 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88068, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87090 = 0; i_87090 < (int64_t) 16; i_87090++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87086 = 0; i_87086 < (int64_t) 4; i_87086++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_77751;
                double r_77753 = 0.0;
                
                for (int64_t i_77752 = 0; i_77752 < (int64_t) 16; i_77752++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_77754 = ((double *) mem_88056)[i_87090 * (int64_t) 16 + i_77752];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_77755 = ((double *) mem_87937)[i_87094 * (int64_t) 64 + i_77752 * (int64_t) 4 + i_87086];
                    
                    // futhark/microgpt.fut:163:26-71
                    
                    double zt_res_77756 = zt_lhs_77754 * zt_rhs_77755;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_77757 = r_77753 + zt_res_77756;
                    double r_tmp_89584 = zp_res_77757;
                    
                    r_77753 = r_tmp_89584;
                }
                defunc_0_lifted_lambda_res_77751 = r_77753;
                ((double *) mem_88084)[i_87086] = defunc_0_lifted_lambda_res_77751;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88079, i_87090 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88084, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_88018, i_87094 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88079, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88100_cached_sizze_89906 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88100, &mem_88100_cached_sizze_89906, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88105_cached_sizze_89907 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88105, &mem_88105_cached_sizze_89907, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87102 = 0; i_87102 < (int64_t) 16; i_87102++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87098 = 0; i_87098 < (int64_t) 16; i_87098++) {
            // futhark/microgpt.fut:164:55-58
            
            int64_t tmp_77769 = sdiv64(i_87098, (int64_t) 4);
            
            // futhark/microgpt.fut:164:45-60
            
            bool x_77770 = sle64((int64_t) 0, tmp_77769);
            
            // futhark/microgpt.fut:164:45-60
            
            bool y_77771 = slt64(tmp_77769, (int64_t) 4);
            
            // futhark/microgpt.fut:164:45-60
            
            bool bounds_check_77772 = x_77770 && y_77771;
            
            // futhark/microgpt.fut:164:45-60
            
            bool index_certs_77773;
            
            if (!bounds_check_77772) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_77769, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:164:45-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:164:16-81\n   #6  futhark/microgpt.fut:381:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:164:75-78
            
            int64_t tmp_77774 = smod64(i_87098, (int64_t) 4);
            
            // futhark/microgpt.fut:164:45-80
            
            bool x_77775 = sle64((int64_t) 0, tmp_77774);
            
            // futhark/microgpt.fut:164:45-80
            
            bool y_77776 = slt64(tmp_77774, (int64_t) 4);
            
            // futhark/microgpt.fut:164:45-80
            
            bool bounds_check_77777 = x_77775 && y_77776;
            
            // futhark/microgpt.fut:164:45-80
            
            bool index_certs_77778;
            
            if (!bounds_check_77777) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_77774, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:164:45-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:164:16-81\n   #6  futhark/microgpt.fut:381:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_77779 = ((double *) mem_88018)[tmp_77769 * (int64_t) 64 + i_87102 * (int64_t) 4 + tmp_77774];
            
            ((double *) mem_88105)[i_87098] = lifted_lambda_res_77779;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88100, i_87102 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88105, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88116_cached_sizze_89908 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88116, &mem_88116_cached_sizze_89908, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88121_cached_sizze_89909 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88121, &mem_88121_cached_sizze_89909, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87110 = 0; i_87110 < (int64_t) 16; i_87110++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87106 = 0; i_87106 < (int64_t) 16; i_87106++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77794;
            double r_77796 = 0.0;
            
            for (int64_t i_77795 = 0; i_77795 < (int64_t) 16; i_77795++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77797 = ((double *) wout_mem_87802.mem)[i_87106 * (int64_t) 16 + i_77795];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77798 = ((double *) mem_88100)[i_87110 * (int64_t) 16 + i_77795];
                
                // futhark/microgpt.fut:165:67-107
                
                double zt_res_77799 = zt_lhs_77797 * zt_rhs_77798;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77800 = r_77796 + zt_res_77799;
                double r_tmp_89589 = zp_res_77800;
                
                r_77796 = r_tmp_89589;
            }
            defunc_0_lifted_lambda_res_77794 = r_77796;
            ((double *) mem_88121)[i_87106] = defunc_0_lifted_lambda_res_77794;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88116, i_87110 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88121, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88132_cached_sizze_89910 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88132, &mem_88132_cached_sizze_89910, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88137_cached_sizze_89911 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88137, &mem_88137_cached_sizze_89911, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87118 = 0; i_87118 < (int64_t) 16; i_87118++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87114 = 0; i_87114 < (int64_t) 16; i_87114++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77815 = ((double *) mem_88116)[i_87118 * (int64_t) 16 + i_87114];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77816 = ((double *) mem_87843)[i_87118 * (int64_t) 16 + i_87114];
            
            // futhark/microgpt.fut:166:46-84
            
            double zp_res_77817 = zp_lhs_77815 + zp_rhs_77816;
            
            ((double *) mem_88137)[i_87114] = zp_res_77817;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88132, i_87118 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88137, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88148_cached_sizze_89912 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88148, &mem_88148_cached_sizze_89912, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88153_cached_sizze_89913 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88153, &mem_88153_cached_sizze_89913, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88160_cached_sizze_89914 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88160, &mem_88160_cached_sizze_89914, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87130 = 0; i_87130 < (int64_t) 16; i_87130++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87122 = 0; i_87122 < (int64_t) 16; i_87122++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77832 = ((double *) mem_88132)[i_87130 * (int64_t) 16 + i_87122];
            
            // futhark/microgpt.fut:167:78-117
            
            double zt_res_77833 = zt_lhs_77832 * zt_lhs_77832;
            
            ((double *) mem_88153)[i_87122] = zt_res_77833;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_77835;
        double r_77837 = 0.0;
        
        for (int64_t i_77836 = 0; i_77836 < (int64_t) 16; i_77836++) {
            // futhark/microgpt.fut:168:37-47
            
            double lifted_lambda_res_77838 = ((double *) mem_88153)[i_77836];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_77839 = r_77837 + lifted_lambda_res_77838;
            double r_tmp_89594 = zp_res_77839;
            
            r_77837 = r_tmp_89594;
        }
        defunc_0_lifted_lambda_res_77835 = r_77837;
        // futhark/microgpt.fut:168:17-64
        
        double zs_res_77840 = defunc_0_lifted_lambda_res_77835 / 16.0;
        
        // futhark/microgpt.fut:169:24-55
        
        double zp_res_77841 = 1.0e-5 + zs_res_77840;
        
        // futhark/microgpt.fut:169:16-55
        
        double sqrt_res_77842 = futrts_sqrt64(zp_res_77841);
        
        // futhark/microgpt.fut:170:28-39
        
        double zs_res_77843 = 1.0 / sqrt_res_77842;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87126 = 0; i_87126 < (int64_t) 16; i_87126++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_77850 = ((double *) mem_88132)[i_87130 * (int64_t) 16 + i_87126];
            
            // futhark/microgpt.fut:170:5-39
            
            double zt_res_77851 = zs_res_77843 * zt_lhs_77850;
            
            ((double *) mem_88160)[i_87126] = zt_res_77851;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88148, i_87130 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88160, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88171_cached_sizze_89915 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88171, &mem_88171_cached_sizze_89915, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88176_cached_sizze_89916 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88176, &mem_88176_cached_sizze_89916, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87138 = 0; i_87138 < (int64_t) 16; i_87138++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87134 = 0; i_87134 < (int64_t) 64; i_87134++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77867;
            double r_77869 = 0.0;
            
            for (int64_t i_77868 = 0; i_77868 < (int64_t) 16; i_77868++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77870 = ((double *) wup_mem_87806.mem)[i_87134 * (int64_t) 16 + i_77868];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77871 = ((double *) mem_88148)[i_87138 * (int64_t) 16 + i_77868];
                
                // futhark/microgpt.fut:171:67-106
                
                double zt_res_77872 = zt_lhs_77870 * zt_rhs_77871;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77873 = r_77869 + zt_res_77872;
                double r_tmp_89598 = zp_res_77873;
                
                r_77869 = r_tmp_89598;
            }
            defunc_0_lifted_lambda_res_77867 = r_77869;
            ((double *) mem_88176)[i_87134] = defunc_0_lifted_lambda_res_77867;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88171, i_87138 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88176, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88187_cached_sizze_89917 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88187, &mem_88187_cached_sizze_89917, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88192_cached_sizze_89918 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88192, &mem_88192_cached_sizze_89918, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87146 = 0; i_87146 < (int64_t) 16; i_87146++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87142 = 0; i_87142 < (int64_t) 64; i_87142++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_77888 = ((double *) mem_88171)[i_87146 * (int64_t) 64 + i_87142];
            
            // futhark/microgpt.fut:172:45-73
            
            double max_res_77889 = fmax64(0.0, max_arg0_77888);
            
            ((double *) mem_88192)[i_87142] = max_res_77889;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88187, i_87146 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88203_cached_sizze_89919 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88203, &mem_88203_cached_sizze_89919, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88208_cached_sizze_89920 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88208, &mem_88208_cached_sizze_89920, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87154 = 0; i_87154 < (int64_t) 16; i_87154++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87150 = 0; i_87150 < (int64_t) 16; i_87150++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77904;
            double r_77906 = 0.0;
            
            for (int64_t i_77905 = 0; i_77905 < (int64_t) 64; i_77905++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77907 = ((double *) wdown_mem_87800.mem)[i_87150 * (int64_t) 64 + i_77905];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77908 = ((double *) mem_88187)[i_87154 * (int64_t) 64 + i_77905];
                
                // futhark/microgpt.fut:173:67-108
                
                double zt_res_77909 = zt_lhs_77907 * zt_rhs_77908;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77910 = r_77906 + zt_res_77909;
                double r_tmp_89603 = zp_res_77910;
                
                r_77906 = r_tmp_89603;
            }
            defunc_0_lifted_lambda_res_77904 = r_77906;
            ((double *) mem_88208)[i_87150] = defunc_0_lifted_lambda_res_77904;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88203, i_87154 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88219_cached_sizze_89921 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88219, &mem_88219_cached_sizze_89921, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88224_cached_sizze_89922 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88224, &mem_88224_cached_sizze_89922, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87162 = 0; i_87162 < (int64_t) 16; i_87162++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87158 = 0; i_87158 < (int64_t) 16; i_87158++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_77925 = ((double *) mem_88203)[i_87162 * (int64_t) 16 + i_87158];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_77926 = ((double *) mem_88132)[i_87162 * (int64_t) 16 + i_87158];
            
            // futhark/microgpt.fut:174:46-85
            
            double zp_res_77927 = zp_lhs_77925 + zp_rhs_77926;
            
            ((double *) mem_88224)[i_87158] = zp_res_77927;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88219, i_87162 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88224, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_88235, (int64_t) 3456, "mem_88235")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88240_cached_sizze_89923 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88240, &mem_88240_cached_sizze_89923, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_87170 = 0; i_87170 < (int64_t) 16; i_87170++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87166 = 0; i_87166 < (int64_t) 27; i_87166++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_77943;
            double r_77945 = 0.0;
            
            for (int64_t i_77944 = 0; i_77944 < (int64_t) 16; i_77944++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_77946 = ((double *) wvoc_mem_87808.mem)[i_87166 * (int64_t) 16 + i_77944];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_77947 = ((double *) mem_88219)[i_87170 * (int64_t) 16 + i_77944];
                
                // futhark/microgpt.fut:175:56-96
                
                double zt_res_77948 = zt_lhs_77946 * zt_rhs_77947;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_77949 = r_77945 + zt_res_77948;
                double r_tmp_89608 = zp_res_77949;
                
                r_77945 = r_tmp_89608;
            }
            defunc_0_lifted_lambda_res_77943 = r_77945;
            ((double *) mem_88240)[i_87166] = defunc_0_lifted_lambda_res_77943;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_88235.mem, i_87170 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88240, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_89540, &mem_88235, "mem_88235") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89870, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_87811);
        free(mem_87816);
        free(mem_87827);
        free(mem_87832);
        free(mem_87843);
        free(mem_87848);
        free(mem_87855);
        free(mem_87866);
        free(mem_87871);
        free(mem_87878);
        free(mem_87889);
        free(mem_87890);
        free(mem_87891);
        free(mem_87904);
        free(mem_87905);
        free(mem_87906);
        free(mem_87937);
        free(mem_87938);
        free(mem_87939);
        free(mem_87955);
        free(mem_87956);
        free(mem_87957);
        free(mem_87970);
        free(mem_87971);
        free(mem_87972);
        free(mem_88018);
        free(mem_88024);
        free(mem_88029);
        free(mem_88040);
        free(mem_88045);
        free(mem_88056);
        free(mem_88061);
        free(mem_88068);
        free(mem_88079);
        free(mem_88084);
        free(mem_88100);
        free(mem_88105);
        free(mem_88116);
        free(mem_88121);
        free(mem_88132);
        free(mem_88137);
        free(mem_88148);
        free(mem_88153);
        free(mem_88160);
        free(mem_88171);
        free(mem_88176);
        free(mem_88187);
        free(mem_88192);
        free(mem_88203);
        free(mem_88208);
        free(mem_88219);
        free(mem_88224);
        free(mem_88240);
        if (memblock_unref(ctx, &mem_88235, "mem_88235") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_89924, struct memblock *mem_out_p_89925, struct memblock *mem_out_p_89926, struct memblock *mem_out_p_89927, struct memblock *mem_out_p_89928, struct memblock *mem_out_p_89929, struct memblock *mem_out_p_89930, struct memblock *mem_out_p_89931, struct memblock *mem_out_p_89932, struct memblock wte_mem_87800, struct memblock wpe_mem_87801, struct memblock wqry_mem_87802, struct memblock wkey_mem_87803, struct memblock wval_mem_87804, struct memblock wout_mem_87805, struct memblock wup_mem_87806, struct memblock wdown_mem_87807, struct memblock wvoc_mem_87808)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    if (memblock_set(ctx, &mem_out_89540, &wdown_mem_87807, "wdown_mem_87807") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89541, &wkey_mem_87803, "wkey_mem_87803") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89542, &wout_mem_87805, "wout_mem_87805") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89543, &wpe_mem_87801, "wpe_mem_87801") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89544, &wqry_mem_87802, "wqry_mem_87802") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89545, &wte_mem_87800, "wte_mem_87800") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89546, &wup_mem_87806, "wup_mem_87806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89547, &wval_mem_87804, "wval_mem_87804") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89548, &wvoc_mem_87808, "wvoc_mem_87808") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89924, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89925, &mem_out_89541, "mem_out_89541") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89926, &mem_out_89542, "mem_out_89542") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89927, &mem_out_89543, "mem_out_89543") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89928, &mem_out_89544, "mem_out_89544") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89929, &mem_out_89545, "mem_out_89545") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89930, &mem_out_89546, "mem_out_89546") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89931, &mem_out_89547, "mem_out_89547") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89932, &mem_out_89548, "mem_out_89548") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_89548, "mem_out_89548") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89547, "mem_out_89547") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89546, "mem_out_89546") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89545, "mem_out_89545") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89544, "mem_out_89544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89543, "mem_out_89543") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89542, "mem_out_89542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89541, "mem_out_89541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_89933, struct memblock *mem_out_p_89934, struct memblock *mem_out_p_89935, struct memblock *mem_out_p_89936, struct memblock *mem_out_p_89937, struct memblock *mem_out_p_89938, struct memblock *mem_out_p_89939, struct memblock *mem_out_p_89940, struct memblock *mem_out_p_89941, struct memblock *mem_out_p_89942, struct memblock *mem_out_p_89943, struct memblock *mem_out_p_89944, struct memblock *mem_out_p_89945, struct memblock *mem_out_p_89946, struct memblock *mem_out_p_89947, struct memblock *mem_out_p_89948, struct memblock *mem_out_p_89949, struct memblock *mem_out_p_89950, struct memblock *mem_out_p_89951, struct memblock *mem_out_p_89952, struct memblock *mem_out_p_89953, struct memblock *mem_out_p_89954, struct memblock *mem_out_p_89955, struct memblock *mem_out_p_89956, struct memblock *mem_out_p_89957, struct memblock *mem_out_p_89958, struct memblock *mem_out_p_89959, struct memblock wdown_mem_87800, struct memblock wkey_mem_87801, struct memblock wout_mem_87802, struct memblock wpe_mem_87803, struct memblock wqry_mem_87804, struct memblock wte_mem_87805, struct memblock wup_mem_87806, struct memblock wval_mem_87807, struct memblock wvoc_mem_87808, struct memblock wdown_mem_87809, struct memblock wkey_mem_87810, struct memblock wout_mem_87811, struct memblock wpe_mem_87812, struct memblock wqry_mem_87813, struct memblock wte_mem_87814, struct memblock wup_mem_87815, struct memblock wval_mem_87816, struct memblock wvoc_mem_87817, struct memblock wdown_mem_87818, struct memblock wkey_mem_87819, struct memblock wout_mem_87820, struct memblock wpe_mem_87821, struct memblock wqry_mem_87822, struct memblock wte_mem_87823, struct memblock wup_mem_87824, struct memblock wval_mem_87825, struct memblock wvoc_mem_87826, struct memblock masks_mem_87827, struct memblock dls_mem_87828, struct memblock seqs_mem_87829)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_87938_cached_sizze_89960 = 0;
    unsigned char *mem_87938 = NULL;
    int64_t mem_87939_cached_sizze_89961 = 0;
    unsigned char *mem_87939 = NULL;
    int64_t mem_87948_cached_sizze_89962 = 0;
    unsigned char *mem_87948 = NULL;
    int64_t mem_87955_cached_sizze_89963 = 0;
    unsigned char *mem_87955 = NULL;
    int64_t mem_87970_cached_sizze_89964 = 0;
    unsigned char *mem_87970 = NULL;
    int64_t mem_87971_cached_sizze_89965 = 0;
    unsigned char *mem_87971 = NULL;
    int64_t mem_87980_cached_sizze_89966 = 0;
    unsigned char *mem_87980 = NULL;
    int64_t mem_87987_cached_sizze_89967 = 0;
    unsigned char *mem_87987 = NULL;
    int64_t mem_88002_cached_sizze_89968 = 0;
    unsigned char *mem_88002 = NULL;
    int64_t mem_88003_cached_sizze_89969 = 0;
    unsigned char *mem_88003 = NULL;
    int64_t mem_88012_cached_sizze_89970 = 0;
    unsigned char *mem_88012 = NULL;
    int64_t mem_88013_cached_sizze_89971 = 0;
    unsigned char *mem_88013 = NULL;
    int64_t mem_88034_cached_sizze_89972 = 0;
    unsigned char *mem_88034 = NULL;
    int64_t mem_88035_cached_sizze_89973 = 0;
    unsigned char *mem_88035 = NULL;
    int64_t mem_88036_cached_sizze_89974 = 0;
    unsigned char *mem_88036 = NULL;
    int64_t mem_88048_cached_sizze_89975 = 0;
    unsigned char *mem_88048 = NULL;
    int64_t mem_88049_cached_sizze_89976 = 0;
    unsigned char *mem_88049 = NULL;
    int64_t mem_88073_cached_sizze_89977 = 0;
    unsigned char *mem_88073 = NULL;
    int64_t mem_88074_cached_sizze_89978 = 0;
    unsigned char *mem_88074 = NULL;
    int64_t mem_88075_cached_sizze_89979 = 0;
    unsigned char *mem_88075 = NULL;
    int64_t mem_88076_cached_sizze_89980 = 0;
    unsigned char *mem_88076 = NULL;
    int64_t mem_88077_cached_sizze_89981 = 0;
    unsigned char *mem_88077 = NULL;
    int64_t mem_88096_cached_sizze_89982 = 0;
    unsigned char *mem_88096 = NULL;
    int64_t mem_88097_cached_sizze_89983 = 0;
    unsigned char *mem_88097 = NULL;
    int64_t mem_88098_cached_sizze_89984 = 0;
    unsigned char *mem_88098 = NULL;
    int64_t mem_88135_cached_sizze_89985 = 0;
    unsigned char *mem_88135 = NULL;
    int64_t mem_88136_cached_sizze_89986 = 0;
    unsigned char *mem_88136 = NULL;
    int64_t mem_88137_cached_sizze_89987 = 0;
    unsigned char *mem_88137 = NULL;
    int64_t mem_88153_cached_sizze_89988 = 0;
    unsigned char *mem_88153 = NULL;
    int64_t mem_88154_cached_sizze_89989 = 0;
    unsigned char *mem_88154 = NULL;
    int64_t mem_88155_cached_sizze_89990 = 0;
    unsigned char *mem_88155 = NULL;
    int64_t mem_88168_cached_sizze_89991 = 0;
    unsigned char *mem_88168 = NULL;
    int64_t mem_88169_cached_sizze_89992 = 0;
    unsigned char *mem_88169 = NULL;
    int64_t mem_88170_cached_sizze_89993 = 0;
    unsigned char *mem_88170 = NULL;
    int64_t mem_88216_cached_sizze_89994 = 0;
    unsigned char *mem_88216 = NULL;
    int64_t mem_88217_cached_sizze_89995 = 0;
    unsigned char *mem_88217 = NULL;
    int64_t mem_88228_cached_sizze_89996 = 0;
    unsigned char *mem_88228 = NULL;
    int64_t mem_88229_cached_sizze_89997 = 0;
    unsigned char *mem_88229 = NULL;
    int64_t mem_88238_cached_sizze_89998 = 0;
    unsigned char *mem_88238 = NULL;
    int64_t mem_88239_cached_sizze_89999 = 0;
    unsigned char *mem_88239 = NULL;
    int64_t mem_88260_cached_sizze_90000 = 0;
    unsigned char *mem_88260 = NULL;
    int64_t mem_88265_cached_sizze_90001 = 0;
    unsigned char *mem_88265 = NULL;
    int64_t mem_88276_cached_sizze_90002 = 0;
    unsigned char *mem_88276 = NULL;
    int64_t mem_88281_cached_sizze_90003 = 0;
    unsigned char *mem_88281 = NULL;
    int64_t mem_88288_cached_sizze_90004 = 0;
    unsigned char *mem_88288 = NULL;
    int64_t mem_88299_cached_sizze_90005 = 0;
    unsigned char *mem_88299 = NULL;
    int64_t mem_88304_cached_sizze_90006 = 0;
    unsigned char *mem_88304 = NULL;
    int64_t mem_88325_cached_sizze_90007 = 0;
    unsigned char *mem_88325 = NULL;
    int64_t mem_88326_cached_sizze_90008 = 0;
    unsigned char *mem_88326 = NULL;
    int64_t mem_88334_cached_sizze_90009 = 0;
    unsigned char *mem_88334 = NULL;
    int64_t mem_88348_cached_sizze_90010 = 0;
    unsigned char *mem_88348 = NULL;
    int64_t mem_88353_cached_sizze_90011 = 0;
    unsigned char *mem_88353 = NULL;
    int64_t mem_88364_cached_sizze_90012 = 0;
    unsigned char *mem_88364 = NULL;
    int64_t mem_88369_cached_sizze_90013 = 0;
    unsigned char *mem_88369 = NULL;
    int64_t mem_88380_cached_sizze_90014 = 0;
    unsigned char *mem_88380 = NULL;
    int64_t mem_88381_cached_sizze_90015 = 0;
    unsigned char *mem_88381 = NULL;
    int64_t mem_88390_cached_sizze_90016 = 0;
    unsigned char *mem_88390 = NULL;
    int64_t mem_88391_cached_sizze_90017 = 0;
    unsigned char *mem_88391 = NULL;
    int64_t mem_88412_cached_sizze_90018 = 0;
    unsigned char *mem_88412 = NULL;
    int64_t mem_88413_cached_sizze_90019 = 0;
    unsigned char *mem_88413 = NULL;
    int64_t mem_88421_cached_sizze_90020 = 0;
    unsigned char *mem_88421 = NULL;
    int64_t mem_88435_cached_sizze_90021 = 0;
    unsigned char *mem_88435 = NULL;
    int64_t mem_88436_cached_sizze_90022 = 0;
    unsigned char *mem_88436 = NULL;
    int64_t mem_88444_cached_sizze_90023 = 0;
    unsigned char *mem_88444 = NULL;
    int64_t mem_88458_cached_sizze_90024 = 0;
    unsigned char *mem_88458 = NULL;
    int64_t mem_88463_cached_sizze_90025 = 0;
    unsigned char *mem_88463 = NULL;
    int64_t mem_88474_cached_sizze_90026 = 0;
    unsigned char *mem_88474 = NULL;
    int64_t mem_88479_cached_sizze_90027 = 0;
    unsigned char *mem_88479 = NULL;
    int64_t mem_88490_cached_sizze_90028 = 0;
    unsigned char *mem_88490 = NULL;
    int64_t mem_88495_cached_sizze_90029 = 0;
    unsigned char *mem_88495 = NULL;
    int64_t mem_88506_cached_sizze_90030 = 0;
    unsigned char *mem_88506 = NULL;
    int64_t mem_88510_cached_sizze_90031 = 0;
    unsigned char *mem_88510 = NULL;
    int64_t mem_88511_cached_sizze_90032 = 0;
    unsigned char *mem_88511 = NULL;
    int64_t mem_88527_cached_sizze_90033 = 0;
    unsigned char *mem_88527 = NULL;
    int64_t mem_88532_cached_sizze_90034 = 0;
    unsigned char *mem_88532 = NULL;
    int64_t mem_88533_cached_sizze_90035 = 0;
    unsigned char *mem_88533 = NULL;
    int64_t mem_88546_cached_sizze_90036 = 0;
    unsigned char *mem_88546 = NULL;
    int64_t mem_88557_cached_sizze_90037 = 0;
    unsigned char *mem_88557 = NULL;
    int64_t mem_88562_cached_sizze_90038 = 0;
    unsigned char *mem_88562 = NULL;
    int64_t mem_88573_cached_sizze_90039 = 0;
    unsigned char *mem_88573 = NULL;
    int64_t mem_88574_cached_sizze_90040 = 0;
    unsigned char *mem_88574 = NULL;
    int64_t mem_88583_cached_sizze_90041 = 0;
    unsigned char *mem_88583 = NULL;
    int64_t mem_88584_cached_sizze_90042 = 0;
    unsigned char *mem_88584 = NULL;
    int64_t mem_88605_cached_sizze_90043 = 0;
    unsigned char *mem_88605 = NULL;
    int64_t mem_88610_cached_sizze_90044 = 0;
    unsigned char *mem_88610 = NULL;
    int64_t mem_88621_cached_sizze_90045 = 0;
    unsigned char *mem_88621 = NULL;
    int64_t mem_88626_cached_sizze_90046 = 0;
    unsigned char *mem_88626 = NULL;
    int64_t mem_88637_cached_sizze_90047 = 0;
    unsigned char *mem_88637 = NULL;
    int64_t mem_88644_cached_sizze_90048 = 0;
    unsigned char *mem_88644 = NULL;
    int64_t mem_88651_cached_sizze_90049 = 0;
    unsigned char *mem_88651 = NULL;
    int64_t mem_88661_cached_sizze_90050 = 0;
    unsigned char *mem_88661 = NULL;
    int64_t mem_88666_cached_sizze_90051 = 0;
    unsigned char *mem_88666 = NULL;
    int64_t mem_88677_cached_sizze_90052 = 0;
    unsigned char *mem_88677 = NULL;
    int64_t mem_88678_cached_sizze_90053 = 0;
    unsigned char *mem_88678 = NULL;
    int64_t mem_88687_cached_sizze_90054 = 0;
    unsigned char *mem_88687 = NULL;
    int64_t mem_88688_cached_sizze_90055 = 0;
    unsigned char *mem_88688 = NULL;
    int64_t mem_88709_cached_sizze_90056 = 0;
    unsigned char *mem_88709 = NULL;
    int64_t mem_88710_cached_sizze_90057 = 0;
    unsigned char *mem_88710 = NULL;
    int64_t mem_88721_cached_sizze_90058 = 0;
    unsigned char *mem_88721 = NULL;
    int64_t mem_88722_cached_sizze_90059 = 0;
    unsigned char *mem_88722 = NULL;
    int64_t mem_88731_cached_sizze_90060 = 0;
    unsigned char *mem_88731 = NULL;
    int64_t mem_88738_cached_sizze_90061 = 0;
    unsigned char *mem_88738 = NULL;
    int64_t mem_88763_cached_sizze_90062 = 0;
    unsigned char *mem_88763 = NULL;
    int64_t mem_88764_cached_sizze_90063 = 0;
    unsigned char *mem_88764 = NULL;
    int64_t mem_88775_cached_sizze_90064 = 0;
    unsigned char *mem_88775 = NULL;
    int64_t mem_88776_cached_sizze_90065 = 0;
    unsigned char *mem_88776 = NULL;
    int64_t mem_88785_cached_sizze_90066 = 0;
    unsigned char *mem_88785 = NULL;
    int64_t mem_88792_cached_sizze_90067 = 0;
    unsigned char *mem_88792 = NULL;
    int64_t mem_88799_cached_sizze_90068 = 0;
    unsigned char *mem_88799 = NULL;
    int64_t mem_88824_cached_sizze_90069 = 0;
    unsigned char *mem_88824 = NULL;
    int64_t mem_88825_cached_sizze_90070 = 0;
    unsigned char *mem_88825 = NULL;
    int64_t mem_88835_cached_sizze_90071 = 0;
    unsigned char *mem_88835 = NULL;
    int64_t mem_88836_cached_sizze_90072 = 0;
    unsigned char *mem_88836 = NULL;
    int64_t mem_88844_cached_sizze_90073 = 0;
    unsigned char *mem_88844 = NULL;
    int64_t mem_88851_cached_sizze_90074 = 0;
    unsigned char *mem_88851 = NULL;
    int64_t mem_88874_cached_sizze_90075 = 0;
    unsigned char *mem_88874 = NULL;
    int64_t mem_88880_cached_sizze_90076 = 0;
    unsigned char *mem_88880 = NULL;
    int64_t mem_88885_cached_sizze_90077 = 0;
    unsigned char *mem_88885 = NULL;
    int64_t mem_88892_cached_sizze_90078 = 0;
    unsigned char *mem_88892 = NULL;
    int64_t mem_88908_cached_sizze_90079 = 0;
    unsigned char *mem_88908 = NULL;
    int64_t mem_88914_cached_sizze_90080 = 0;
    unsigned char *mem_88914 = NULL;
    int64_t mem_88919_cached_sizze_90081 = 0;
    unsigned char *mem_88919 = NULL;
    int64_t mem_88935_cached_sizze_90082 = 0;
    unsigned char *mem_88935 = NULL;
    int64_t mem_88936_cached_sizze_90083 = 0;
    unsigned char *mem_88936 = NULL;
    int64_t mem_88947_cached_sizze_90084 = 0;
    unsigned char *mem_88947 = NULL;
    int64_t mem_88948_cached_sizze_90085 = 0;
    unsigned char *mem_88948 = NULL;
    int64_t mem_88957_cached_sizze_90086 = 0;
    unsigned char *mem_88957 = NULL;
    int64_t mem_88958_cached_sizze_90087 = 0;
    unsigned char *mem_88958 = NULL;
    int64_t mem_88989_cached_sizze_90088 = 0;
    unsigned char *mem_88989 = NULL;
    int64_t mem_88990_cached_sizze_90089 = 0;
    unsigned char *mem_88990 = NULL;
    int64_t mem_88991_cached_sizze_90090 = 0;
    unsigned char *mem_88991 = NULL;
    int64_t mem_89004_cached_sizze_90091 = 0;
    unsigned char *mem_89004 = NULL;
    int64_t mem_89005_cached_sizze_90092 = 0;
    unsigned char *mem_89005 = NULL;
    int64_t mem_89006_cached_sizze_90093 = 0;
    unsigned char *mem_89006 = NULL;
    int64_t mem_89037_cached_sizze_90094 = 0;
    unsigned char *mem_89037 = NULL;
    int64_t mem_89038_cached_sizze_90095 = 0;
    unsigned char *mem_89038 = NULL;
    int64_t mem_89039_cached_sizze_90096 = 0;
    unsigned char *mem_89039 = NULL;
    int64_t mem_89040_cached_sizze_90097 = 0;
    unsigned char *mem_89040 = NULL;
    int64_t mem_89057_cached_sizze_90098 = 0;
    unsigned char *mem_89057 = NULL;
    int64_t mem_89058_cached_sizze_90099 = 0;
    unsigned char *mem_89058 = NULL;
    int64_t mem_89059_cached_sizze_90100 = 0;
    unsigned char *mem_89059 = NULL;
    int64_t mem_89060_cached_sizze_90101 = 0;
    unsigned char *mem_89060 = NULL;
    int64_t mem_89101_cached_sizze_90102 = 0;
    unsigned char *mem_89101 = NULL;
    int64_t mem_89108_cached_sizze_90103 = 0;
    unsigned char *mem_89108 = NULL;
    int64_t mem_89115_cached_sizze_90104 = 0;
    unsigned char *mem_89115 = NULL;
    int64_t mem_89125_cached_sizze_90105 = 0;
    unsigned char *mem_89125 = NULL;
    int64_t mem_89130_cached_sizze_90106 = 0;
    unsigned char *mem_89130 = NULL;
    int64_t mem_89141_cached_sizze_90107 = 0;
    unsigned char *mem_89141 = NULL;
    int64_t mem_89148_cached_sizze_90108 = 0;
    unsigned char *mem_89148 = NULL;
    int64_t mem_89155_cached_sizze_90109 = 0;
    unsigned char *mem_89155 = NULL;
    int64_t mem_89165_cached_sizze_90110 = 0;
    unsigned char *mem_89165 = NULL;
    int64_t mem_89170_cached_sizze_90111 = 0;
    unsigned char *mem_89170 = NULL;
    int64_t mem_89181_cached_sizze_90112 = 0;
    unsigned char *mem_89181 = NULL;
    int64_t mem_89182_cached_sizze_90113 = 0;
    unsigned char *mem_89182 = NULL;
    int64_t mem_89191_cached_sizze_90114 = 0;
    unsigned char *mem_89191 = NULL;
    int64_t mem_89192_cached_sizze_90115 = 0;
    unsigned char *mem_89192 = NULL;
    int64_t mem_89213_cached_sizze_90116 = 0;
    unsigned char *mem_89213 = NULL;
    int64_t mem_89218_cached_sizze_90117 = 0;
    unsigned char *mem_89218 = NULL;
    int64_t mem_89229_cached_sizze_90118 = 0;
    unsigned char *mem_89229 = NULL;
    int64_t mem_89230_cached_sizze_90119 = 0;
    unsigned char *mem_89230 = NULL;
    int64_t mem_89239_cached_sizze_90120 = 0;
    unsigned char *mem_89239 = NULL;
    int64_t mem_89240_cached_sizze_90121 = 0;
    unsigned char *mem_89240 = NULL;
    struct memblock mem_param_tmp_89593;
    
    mem_param_tmp_89593.references = NULL;
    
    struct memblock mem_param_tmp_89592;
    
    mem_param_tmp_89592.references = NULL;
    
    struct memblock mem_param_tmp_89591;
    
    mem_param_tmp_89591.references = NULL;
    
    struct memblock mem_param_tmp_89590;
    
    mem_param_tmp_89590.references = NULL;
    
    struct memblock mem_param_tmp_89589;
    
    mem_param_tmp_89589.references = NULL;
    
    struct memblock mem_param_tmp_89588;
    
    mem_param_tmp_89588.references = NULL;
    
    struct memblock mem_param_tmp_89587;
    
    mem_param_tmp_89587.references = NULL;
    
    struct memblock mem_param_tmp_89586;
    
    mem_param_tmp_89586.references = NULL;
    
    struct memblock mem_param_tmp_89585;
    
    mem_param_tmp_89585.references = NULL;
    
    struct memblock mem_param_tmp_89584;
    
    mem_param_tmp_89584.references = NULL;
    
    struct memblock mem_param_tmp_89583;
    
    mem_param_tmp_89583.references = NULL;
    
    struct memblock mem_param_tmp_89582;
    
    mem_param_tmp_89582.references = NULL;
    
    struct memblock mem_param_tmp_89581;
    
    mem_param_tmp_89581.references = NULL;
    
    struct memblock mem_param_tmp_89580;
    
    mem_param_tmp_89580.references = NULL;
    
    struct memblock mem_param_tmp_89579;
    
    mem_param_tmp_89579.references = NULL;
    
    struct memblock mem_param_tmp_89578;
    
    mem_param_tmp_89578.references = NULL;
    
    struct memblock mem_param_tmp_89577;
    
    mem_param_tmp_89577.references = NULL;
    
    struct memblock mem_param_tmp_89576;
    
    mem_param_tmp_89576.references = NULL;
    
    struct memblock mem_param_tmp_89575;
    
    mem_param_tmp_89575.references = NULL;
    
    struct memblock mem_param_tmp_89574;
    
    mem_param_tmp_89574.references = NULL;
    
    struct memblock mem_param_tmp_89573;
    
    mem_param_tmp_89573.references = NULL;
    
    struct memblock mem_param_tmp_89572;
    
    mem_param_tmp_89572.references = NULL;
    
    struct memblock mem_param_tmp_89571;
    
    mem_param_tmp_89571.references = NULL;
    
    struct memblock mem_param_tmp_89570;
    
    mem_param_tmp_89570.references = NULL;
    
    struct memblock mem_param_tmp_89569;
    
    mem_param_tmp_89569.references = NULL;
    
    struct memblock mem_param_tmp_89568;
    
    mem_param_tmp_89568.references = NULL;
    
    struct memblock mem_param_tmp_89567;
    
    mem_param_tmp_89567.references = NULL;
    
    struct memblock ext_mem_89357;
    
    ext_mem_89357.references = NULL;
    
    struct memblock ext_mem_89358;
    
    ext_mem_89358.references = NULL;
    
    struct memblock ext_mem_89359;
    
    ext_mem_89359.references = NULL;
    
    struct memblock mem_89355;
    
    mem_89355.references = NULL;
    
    struct memblock mem_89353;
    
    mem_89353.references = NULL;
    
    struct memblock mem_89351;
    
    mem_89351.references = NULL;
    
    struct memblock mem_89349;
    
    mem_89349.references = NULL;
    
    struct memblock ext_mem_89346;
    
    ext_mem_89346.references = NULL;
    
    struct memblock ext_mem_89347;
    
    ext_mem_89347.references = NULL;
    
    struct memblock ext_mem_89348;
    
    ext_mem_89348.references = NULL;
    
    struct memblock mem_89344;
    
    mem_89344.references = NULL;
    
    struct memblock mem_89342;
    
    mem_89342.references = NULL;
    
    struct memblock mem_89340;
    
    mem_89340.references = NULL;
    
    struct memblock mem_89338;
    
    mem_89338.references = NULL;
    
    struct memblock ext_mem_89335;
    
    ext_mem_89335.references = NULL;
    
    struct memblock ext_mem_89336;
    
    ext_mem_89336.references = NULL;
    
    struct memblock ext_mem_89337;
    
    ext_mem_89337.references = NULL;
    
    struct memblock mem_89333;
    
    mem_89333.references = NULL;
    
    struct memblock mem_89331;
    
    mem_89331.references = NULL;
    
    struct memblock mem_89329;
    
    mem_89329.references = NULL;
    
    struct memblock mem_89327;
    
    mem_89327.references = NULL;
    
    struct memblock ext_mem_89324;
    
    ext_mem_89324.references = NULL;
    
    struct memblock ext_mem_89325;
    
    ext_mem_89325.references = NULL;
    
    struct memblock ext_mem_89326;
    
    ext_mem_89326.references = NULL;
    
    struct memblock mem_89322;
    
    mem_89322.references = NULL;
    
    struct memblock mem_89320;
    
    mem_89320.references = NULL;
    
    struct memblock mem_89318;
    
    mem_89318.references = NULL;
    
    struct memblock mem_89316;
    
    mem_89316.references = NULL;
    
    struct memblock ext_mem_89313;
    
    ext_mem_89313.references = NULL;
    
    struct memblock ext_mem_89314;
    
    ext_mem_89314.references = NULL;
    
    struct memblock ext_mem_89315;
    
    ext_mem_89315.references = NULL;
    
    struct memblock mem_89311;
    
    mem_89311.references = NULL;
    
    struct memblock mem_89309;
    
    mem_89309.references = NULL;
    
    struct memblock mem_89307;
    
    mem_89307.references = NULL;
    
    struct memblock mem_89305;
    
    mem_89305.references = NULL;
    
    struct memblock ext_mem_89302;
    
    ext_mem_89302.references = NULL;
    
    struct memblock ext_mem_89303;
    
    ext_mem_89303.references = NULL;
    
    struct memblock ext_mem_89304;
    
    ext_mem_89304.references = NULL;
    
    struct memblock mem_89300;
    
    mem_89300.references = NULL;
    
    struct memblock mem_89298;
    
    mem_89298.references = NULL;
    
    struct memblock mem_89296;
    
    mem_89296.references = NULL;
    
    struct memblock mem_89294;
    
    mem_89294.references = NULL;
    
    struct memblock ext_mem_89291;
    
    ext_mem_89291.references = NULL;
    
    struct memblock ext_mem_89292;
    
    ext_mem_89292.references = NULL;
    
    struct memblock ext_mem_89293;
    
    ext_mem_89293.references = NULL;
    
    struct memblock mem_89289;
    
    mem_89289.references = NULL;
    
    struct memblock mem_89287;
    
    mem_89287.references = NULL;
    
    struct memblock mem_89285;
    
    mem_89285.references = NULL;
    
    struct memblock mem_89283;
    
    mem_89283.references = NULL;
    
    struct memblock ext_mem_89280;
    
    ext_mem_89280.references = NULL;
    
    struct memblock ext_mem_89281;
    
    ext_mem_89281.references = NULL;
    
    struct memblock ext_mem_89282;
    
    ext_mem_89282.references = NULL;
    
    struct memblock mem_89278;
    
    mem_89278.references = NULL;
    
    struct memblock mem_89276;
    
    mem_89276.references = NULL;
    
    struct memblock mem_89274;
    
    mem_89274.references = NULL;
    
    struct memblock mem_89272;
    
    mem_89272.references = NULL;
    
    struct memblock ext_mem_89269;
    
    ext_mem_89269.references = NULL;
    
    struct memblock ext_mem_89270;
    
    ext_mem_89270.references = NULL;
    
    struct memblock ext_mem_89271;
    
    ext_mem_89271.references = NULL;
    
    struct memblock mem_89267;
    
    mem_89267.references = NULL;
    
    struct memblock mem_89265;
    
    mem_89265.references = NULL;
    
    struct memblock mem_89263;
    
    mem_89263.references = NULL;
    
    struct memblock mem_89261;
    
    mem_89261.references = NULL;
    
    struct memblock mem_param_87937;
    
    mem_param_87937.references = NULL;
    
    struct memblock mem_param_87933;
    
    mem_param_87933.references = NULL;
    
    struct memblock mem_param_87929;
    
    mem_param_87929.references = NULL;
    
    struct memblock mem_param_87925;
    
    mem_param_87925.references = NULL;
    
    struct memblock mem_param_87921;
    
    mem_param_87921.references = NULL;
    
    struct memblock mem_param_87917;
    
    mem_param_87917.references = NULL;
    
    struct memblock mem_param_87913;
    
    mem_param_87913.references = NULL;
    
    struct memblock mem_param_87909;
    
    mem_param_87909.references = NULL;
    
    struct memblock mem_param_87905;
    
    mem_param_87905.references = NULL;
    
    struct memblock mem_param_87901;
    
    mem_param_87901.references = NULL;
    
    struct memblock mem_param_87897;
    
    mem_param_87897.references = NULL;
    
    struct memblock mem_param_87893;
    
    mem_param_87893.references = NULL;
    
    struct memblock mem_param_87889;
    
    mem_param_87889.references = NULL;
    
    struct memblock mem_param_87885;
    
    mem_param_87885.references = NULL;
    
    struct memblock mem_param_87881;
    
    mem_param_87881.references = NULL;
    
    struct memblock mem_param_87877;
    
    mem_param_87877.references = NULL;
    
    struct memblock mem_param_87873;
    
    mem_param_87873.references = NULL;
    
    struct memblock mem_param_87869;
    
    mem_param_87869.references = NULL;
    
    struct memblock mem_param_87865;
    
    mem_param_87865.references = NULL;
    
    struct memblock mem_param_87861;
    
    mem_param_87861.references = NULL;
    
    struct memblock mem_param_87857;
    
    mem_param_87857.references = NULL;
    
    struct memblock mem_param_87853;
    
    mem_param_87853.references = NULL;
    
    struct memblock mem_param_87849;
    
    mem_param_87849.references = NULL;
    
    struct memblock mem_param_87845;
    
    mem_param_87845.references = NULL;
    
    struct memblock mem_param_87841;
    
    mem_param_87841.references = NULL;
    
    struct memblock mem_param_87837;
    
    mem_param_87837.references = NULL;
    
    struct memblock mem_param_87833;
    
    mem_param_87833.references = NULL;
    
    struct memblock ext_mem_89441;
    
    ext_mem_89441.references = NULL;
    
    struct memblock ext_mem_89442;
    
    ext_mem_89442.references = NULL;
    
    struct memblock ext_mem_89443;
    
    ext_mem_89443.references = NULL;
    
    struct memblock ext_mem_89444;
    
    ext_mem_89444.references = NULL;
    
    struct memblock ext_mem_89445;
    
    ext_mem_89445.references = NULL;
    
    struct memblock ext_mem_89446;
    
    ext_mem_89446.references = NULL;
    
    struct memblock ext_mem_89447;
    
    ext_mem_89447.references = NULL;
    
    struct memblock ext_mem_89448;
    
    ext_mem_89448.references = NULL;
    
    struct memblock ext_mem_89449;
    
    ext_mem_89449.references = NULL;
    
    struct memblock ext_mem_89450;
    
    ext_mem_89450.references = NULL;
    
    struct memblock ext_mem_89451;
    
    ext_mem_89451.references = NULL;
    
    struct memblock ext_mem_89452;
    
    ext_mem_89452.references = NULL;
    
    struct memblock ext_mem_89453;
    
    ext_mem_89453.references = NULL;
    
    struct memblock ext_mem_89454;
    
    ext_mem_89454.references = NULL;
    
    struct memblock ext_mem_89455;
    
    ext_mem_89455.references = NULL;
    
    struct memblock ext_mem_89456;
    
    ext_mem_89456.references = NULL;
    
    struct memblock ext_mem_89457;
    
    ext_mem_89457.references = NULL;
    
    struct memblock ext_mem_89458;
    
    ext_mem_89458.references = NULL;
    
    struct memblock ext_mem_89459;
    
    ext_mem_89459.references = NULL;
    
    struct memblock ext_mem_89460;
    
    ext_mem_89460.references = NULL;
    
    struct memblock ext_mem_89461;
    
    ext_mem_89461.references = NULL;
    
    struct memblock ext_mem_89462;
    
    ext_mem_89462.references = NULL;
    
    struct memblock ext_mem_89463;
    
    ext_mem_89463.references = NULL;
    
    struct memblock ext_mem_89464;
    
    ext_mem_89464.references = NULL;
    
    struct memblock ext_mem_89465;
    
    ext_mem_89465.references = NULL;
    
    struct memblock ext_mem_89466;
    
    ext_mem_89466.references = NULL;
    
    struct memblock ext_mem_89467;
    
    ext_mem_89467.references = NULL;
    
    struct memblock mem_out_89566;
    
    mem_out_89566.references = NULL;
    
    struct memblock mem_out_89565;
    
    mem_out_89565.references = NULL;
    
    struct memblock mem_out_89564;
    
    mem_out_89564.references = NULL;
    
    struct memblock mem_out_89563;
    
    mem_out_89563.references = NULL;
    
    struct memblock mem_out_89562;
    
    mem_out_89562.references = NULL;
    
    struct memblock mem_out_89561;
    
    mem_out_89561.references = NULL;
    
    struct memblock mem_out_89560;
    
    mem_out_89560.references = NULL;
    
    struct memblock mem_out_89559;
    
    mem_out_89559.references = NULL;
    
    struct memblock mem_out_89558;
    
    mem_out_89558.references = NULL;
    
    struct memblock mem_out_89557;
    
    mem_out_89557.references = NULL;
    
    struct memblock mem_out_89556;
    
    mem_out_89556.references = NULL;
    
    struct memblock mem_out_89555;
    
    mem_out_89555.references = NULL;
    
    struct memblock mem_out_89554;
    
    mem_out_89554.references = NULL;
    
    struct memblock mem_out_89553;
    
    mem_out_89553.references = NULL;
    
    struct memblock mem_out_89552;
    
    mem_out_89552.references = NULL;
    
    struct memblock mem_out_89551;
    
    mem_out_89551.references = NULL;
    
    struct memblock mem_out_89550;
    
    mem_out_89550.references = NULL;
    
    struct memblock mem_out_89549;
    
    mem_out_89549.references = NULL;
    
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_87938_cached_sizze_89960 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87938, &mem_87938_cached_sizze_89960, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87939_cached_sizze_89961 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87939, &mem_87939_cached_sizze_89961, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87948_cached_sizze_89962 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87948, &mem_87948_cached_sizze_89962, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87955_cached_sizze_89963 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87955, &mem_87955_cached_sizze_89963, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87970_cached_sizze_89964 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_87970, &mem_87970_cached_sizze_89964, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87971_cached_sizze_89965 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_87971, &mem_87971_cached_sizze_89965, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87980_cached_sizze_89966 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_87980, &mem_87980_cached_sizze_89966, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_87987_cached_sizze_89967 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_87987, &mem_87987_cached_sizze_89967, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88002_cached_sizze_89968 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88002, &mem_88002_cached_sizze_89968, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88003_cached_sizze_89969 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88003, &mem_88003_cached_sizze_89969, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88012_cached_sizze_89970 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88012, &mem_88012_cached_sizze_89970, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88013_cached_sizze_89971 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88013, &mem_88013_cached_sizze_89971, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88034_cached_sizze_89972 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88034, &mem_88034_cached_sizze_89972, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88035_cached_sizze_89973 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88035, &mem_88035_cached_sizze_89973, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88036_cached_sizze_89974 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88036, &mem_88036_cached_sizze_89974, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88048_cached_sizze_89975 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88048, &mem_88048_cached_sizze_89975, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88049_cached_sizze_89976 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88049, &mem_88049_cached_sizze_89976, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88073_cached_sizze_89977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88073, &mem_88073_cached_sizze_89977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88074_cached_sizze_89978 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88074, &mem_88074_cached_sizze_89978, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88075_cached_sizze_89979 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88075, &mem_88075_cached_sizze_89979, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88076_cached_sizze_89980 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88076, &mem_88076_cached_sizze_89980, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88077_cached_sizze_89981 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88077, &mem_88077_cached_sizze_89981, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88096_cached_sizze_89982 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88096, &mem_88096_cached_sizze_89982, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88097_cached_sizze_89983 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88097, &mem_88097_cached_sizze_89983, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88098_cached_sizze_89984 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88098, &mem_88098_cached_sizze_89984, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88135_cached_sizze_89985 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88135, &mem_88135_cached_sizze_89985, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88136_cached_sizze_89986 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88136, &mem_88136_cached_sizze_89986, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88137_cached_sizze_89987 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88137, &mem_88137_cached_sizze_89987, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88153_cached_sizze_89988 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88153, &mem_88153_cached_sizze_89988, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88154_cached_sizze_89989 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88154, &mem_88154_cached_sizze_89989, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88155_cached_sizze_89990 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88155, &mem_88155_cached_sizze_89990, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88168_cached_sizze_89991 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88168, &mem_88168_cached_sizze_89991, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88169_cached_sizze_89992 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88169, &mem_88169_cached_sizze_89992, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88170_cached_sizze_89993 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88170, &mem_88170_cached_sizze_89993, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88216_cached_sizze_89994 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88216, &mem_88216_cached_sizze_89994, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88217_cached_sizze_89995 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88217, &mem_88217_cached_sizze_89995, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88228_cached_sizze_89996 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88228, &mem_88228_cached_sizze_89996, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88229_cached_sizze_89997 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88229, &mem_88229_cached_sizze_89997, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88238_cached_sizze_89998 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88238, &mem_88238_cached_sizze_89998, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88239_cached_sizze_89999 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88239, &mem_88239_cached_sizze_89999, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88260_cached_sizze_90000 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88260, &mem_88260_cached_sizze_90000, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88265_cached_sizze_90001 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88265, &mem_88265_cached_sizze_90001, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88276_cached_sizze_90002 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88276, &mem_88276_cached_sizze_90002, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88281_cached_sizze_90003 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88281, &mem_88281_cached_sizze_90003, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88288_cached_sizze_90004 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88288, &mem_88288_cached_sizze_90004, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88299_cached_sizze_90005 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88299, &mem_88299_cached_sizze_90005, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88304_cached_sizze_90006 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88304, &mem_88304_cached_sizze_90006, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88325_cached_sizze_90007 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88325, &mem_88325_cached_sizze_90007, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88326_cached_sizze_90008 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88326, &mem_88326_cached_sizze_90008, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88334_cached_sizze_90009 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88334, &mem_88334_cached_sizze_90009, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88348_cached_sizze_90010 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88348, &mem_88348_cached_sizze_90010, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88353_cached_sizze_90011 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88353, &mem_88353_cached_sizze_90011, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88364_cached_sizze_90012 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88364, &mem_88364_cached_sizze_90012, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88369_cached_sizze_90013 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88369, &mem_88369_cached_sizze_90013, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88380_cached_sizze_90014 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88380, &mem_88380_cached_sizze_90014, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88381_cached_sizze_90015 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88381, &mem_88381_cached_sizze_90015, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88390_cached_sizze_90016 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88390, &mem_88390_cached_sizze_90016, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88391_cached_sizze_90017 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88391, &mem_88391_cached_sizze_90017, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88412_cached_sizze_90018 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88412, &mem_88412_cached_sizze_90018, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88413_cached_sizze_90019 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88413, &mem_88413_cached_sizze_90019, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88421_cached_sizze_90020 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88421, &mem_88421_cached_sizze_90020, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88435_cached_sizze_90021 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88435, &mem_88435_cached_sizze_90021, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88436_cached_sizze_90022 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88436, &mem_88436_cached_sizze_90022, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88444_cached_sizze_90023 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88444, &mem_88444_cached_sizze_90023, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88458_cached_sizze_90024 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88458, &mem_88458_cached_sizze_90024, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88463_cached_sizze_90025 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88463, &mem_88463_cached_sizze_90025, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88474_cached_sizze_90026 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88474, &mem_88474_cached_sizze_90026, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88479_cached_sizze_90027 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88479, &mem_88479_cached_sizze_90027, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88490_cached_sizze_90028 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88490, &mem_88490_cached_sizze_90028, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88495_cached_sizze_90029 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88495, &mem_88495_cached_sizze_90029, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88506_cached_sizze_90030 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88506, &mem_88506_cached_sizze_90030, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88510_cached_sizze_90031 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88510, &mem_88510_cached_sizze_90031, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88511_cached_sizze_90032 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88511, &mem_88511_cached_sizze_90032, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88527_cached_sizze_90033 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_88527, &mem_88527_cached_sizze_90033, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88532_cached_sizze_90034 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88532, &mem_88532_cached_sizze_90034, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88533_cached_sizze_90035 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88533, &mem_88533_cached_sizze_90035, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88546_cached_sizze_90036 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_88546, &mem_88546_cached_sizze_90036, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88557_cached_sizze_90037 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88557, &mem_88557_cached_sizze_90037, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88562_cached_sizze_90038 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88562, &mem_88562_cached_sizze_90038, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88573_cached_sizze_90039 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88573, &mem_88573_cached_sizze_90039, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88574_cached_sizze_90040 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88574, &mem_88574_cached_sizze_90040, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88583_cached_sizze_90041 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88583, &mem_88583_cached_sizze_90041, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88584_cached_sizze_90042 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88584, &mem_88584_cached_sizze_90042, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88605_cached_sizze_90043 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88605, &mem_88605_cached_sizze_90043, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88610_cached_sizze_90044 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88610, &mem_88610_cached_sizze_90044, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88621_cached_sizze_90045 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88621, &mem_88621_cached_sizze_90045, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88626_cached_sizze_90046 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88626, &mem_88626_cached_sizze_90046, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88637_cached_sizze_90047 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88637, &mem_88637_cached_sizze_90047, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88644_cached_sizze_90048 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88644, &mem_88644_cached_sizze_90048, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88651_cached_sizze_90049 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88651, &mem_88651_cached_sizze_90049, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88661_cached_sizze_90050 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88661, &mem_88661_cached_sizze_90050, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88666_cached_sizze_90051 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88666, &mem_88666_cached_sizze_90051, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88677_cached_sizze_90052 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88677, &mem_88677_cached_sizze_90052, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88678_cached_sizze_90053 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88678, &mem_88678_cached_sizze_90053, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88687_cached_sizze_90054 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88687, &mem_88687_cached_sizze_90054, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88688_cached_sizze_90055 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88688, &mem_88688_cached_sizze_90055, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88709_cached_sizze_90056 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88709, &mem_88709_cached_sizze_90056, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88710_cached_sizze_90057 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88710, &mem_88710_cached_sizze_90057, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88721_cached_sizze_90058 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88721, &mem_88721_cached_sizze_90058, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88722_cached_sizze_90059 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88722, &mem_88722_cached_sizze_90059, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88731_cached_sizze_90060 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88731, &mem_88731_cached_sizze_90060, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88738_cached_sizze_90061 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88738, &mem_88738_cached_sizze_90061, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88763_cached_sizze_90062 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88763, &mem_88763_cached_sizze_90062, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88764_cached_sizze_90063 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88764, &mem_88764_cached_sizze_90063, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88775_cached_sizze_90064 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88775, &mem_88775_cached_sizze_90064, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88776_cached_sizze_90065 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88776, &mem_88776_cached_sizze_90065, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88785_cached_sizze_90066 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88785, &mem_88785_cached_sizze_90066, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88792_cached_sizze_90067 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88792, &mem_88792_cached_sizze_90067, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88799_cached_sizze_90068 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88799, &mem_88799_cached_sizze_90068, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88824_cached_sizze_90069 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88824, &mem_88824_cached_sizze_90069, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88825_cached_sizze_90070 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88825, &mem_88825_cached_sizze_90070, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88835_cached_sizze_90071 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88835, &mem_88835_cached_sizze_90071, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88836_cached_sizze_90072 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88836, &mem_88836_cached_sizze_90072, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88844_cached_sizze_90073 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88844, &mem_88844_cached_sizze_90073, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88851_cached_sizze_90074 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88851, &mem_88851_cached_sizze_90074, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88874_cached_sizze_90075 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88874, &mem_88874_cached_sizze_90075, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88880_cached_sizze_90076 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88880, &mem_88880_cached_sizze_90076, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88885_cached_sizze_90077 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88885, &mem_88885_cached_sizze_90077, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88892_cached_sizze_90078 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88892, &mem_88892_cached_sizze_90078, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88908_cached_sizze_90079 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_88908, &mem_88908_cached_sizze_90079, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88914_cached_sizze_90080 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88914, &mem_88914_cached_sizze_90080, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88919_cached_sizze_90081 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_88919, &mem_88919_cached_sizze_90081, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88935_cached_sizze_90082 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88935, &mem_88935_cached_sizze_90082, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88936_cached_sizze_90083 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88936, &mem_88936_cached_sizze_90083, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88947_cached_sizze_90084 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88947, &mem_88947_cached_sizze_90084, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88948_cached_sizze_90085 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_88948, &mem_88948_cached_sizze_90085, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88957_cached_sizze_90086 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88957, &mem_88957_cached_sizze_90086, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88958_cached_sizze_90087 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_88958, &mem_88958_cached_sizze_90087, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88989_cached_sizze_90088 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88989, &mem_88989_cached_sizze_90088, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88990_cached_sizze_90089 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88990, &mem_88990_cached_sizze_90089, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_88991_cached_sizze_90090 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_88991, &mem_88991_cached_sizze_90090, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89004_cached_sizze_90091 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89004, &mem_89004_cached_sizze_90091, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89005_cached_sizze_90092 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89005, &mem_89005_cached_sizze_90092, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89006_cached_sizze_90093 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89006, &mem_89006_cached_sizze_90093, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89037_cached_sizze_90094 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89037, &mem_89037_cached_sizze_90094, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89038_cached_sizze_90095 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89038, &mem_89038_cached_sizze_90095, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89039_cached_sizze_90096 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89039, &mem_89039_cached_sizze_90096, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89040_cached_sizze_90097 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89040, &mem_89040_cached_sizze_90097, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89057_cached_sizze_90098 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89057, &mem_89057_cached_sizze_90098, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89058_cached_sizze_90099 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89058, &mem_89058_cached_sizze_90099, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89059_cached_sizze_90100 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89059, &mem_89059_cached_sizze_90100, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89060_cached_sizze_90101 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89060, &mem_89060_cached_sizze_90101, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89101_cached_sizze_90102 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89101, &mem_89101_cached_sizze_90102, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89108_cached_sizze_90103 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89108, &mem_89108_cached_sizze_90103, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89115_cached_sizze_90104 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89115, &mem_89115_cached_sizze_90104, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89125_cached_sizze_90105 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89125, &mem_89125_cached_sizze_90105, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89130_cached_sizze_90106 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89130, &mem_89130_cached_sizze_90106, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89141_cached_sizze_90107 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89141, &mem_89141_cached_sizze_90107, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89148_cached_sizze_90108 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89148, &mem_89148_cached_sizze_90108, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89155_cached_sizze_90109 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89155, &mem_89155_cached_sizze_90109, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89165_cached_sizze_90110 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89165, &mem_89165_cached_sizze_90110, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89170_cached_sizze_90111 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89170, &mem_89170_cached_sizze_90111, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89181_cached_sizze_90112 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89181, &mem_89181_cached_sizze_90112, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89182_cached_sizze_90113 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_89182, &mem_89182_cached_sizze_90113, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89191_cached_sizze_90114 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89191, &mem_89191_cached_sizze_90114, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89192_cached_sizze_90115 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89192, &mem_89192_cached_sizze_90115, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89213_cached_sizze_90116 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_89213, &mem_89213_cached_sizze_90116, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89218_cached_sizze_90117 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89218, &mem_89218_cached_sizze_90117, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89229_cached_sizze_90118 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89229, &mem_89229_cached_sizze_90118, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89230_cached_sizze_90119 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_89230, &mem_89230_cached_sizze_90119, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89239_cached_sizze_90120 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89239, &mem_89239_cached_sizze_90120, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_89240_cached_sizze_90121 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_89240, &mem_89240_cached_sizze_90121, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:474:5-479:51
    if (memblock_set(ctx, &mem_param_87833, &wdown_mem_87800, "wdown_mem_87800") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87837, &wkey_mem_87801, "wkey_mem_87801") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87841, &wout_mem_87802, "wout_mem_87802") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87845, &wpe_mem_87803, "wpe_mem_87803") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87849, &wqry_mem_87804, "wqry_mem_87804") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87853, &wte_mem_87805, "wte_mem_87805") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87857, &wup_mem_87806, "wup_mem_87806") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87861, &wval_mem_87807, "wval_mem_87807") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87865, &wvoc_mem_87808, "wvoc_mem_87808") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87869, &wdown_mem_87809, "wdown_mem_87809") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87873, &wkey_mem_87810, "wkey_mem_87810") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87877, &wout_mem_87811, "wout_mem_87811") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87881, &wpe_mem_87812, "wpe_mem_87812") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87885, &wqry_mem_87813, "wqry_mem_87813") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87889, &wte_mem_87814, "wte_mem_87814") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87893, &wup_mem_87815, "wup_mem_87815") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87897, &wval_mem_87816, "wval_mem_87816") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87901, &wvoc_mem_87817, "wvoc_mem_87817") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87905, &wdown_mem_87818, "wdown_mem_87818") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87909, &wkey_mem_87819, "wkey_mem_87819") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87913, &wout_mem_87820, "wout_mem_87820") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87917, &wpe_mem_87821, "wpe_mem_87821") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87921, &wqry_mem_87822, "wqry_mem_87822") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87925, &wte_mem_87823, "wte_mem_87823") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87929, &wup_mem_87824, "wup_mem_87824") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87933, &wval_mem_87825, "wval_mem_87825") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_87937, &wvoc_mem_87826, "wvoc_mem_87826") != 0)
        return 1;
    for (int64_t step_81116 = 0; step_81116 < (int64_t) 500; step_81116++) {
        // futhark/microgpt.fut:476:16-25
        
        int64_t dl_81144 = ((int64_t *) dls_mem_87828.mem)[step_81116];
        
        // futhark/microgpt.fut:389:37-40
        
        int64_t zl_rhs_81149 = sub64(dl_81144, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86976 = 0; i_86976 < (int64_t) 16; i_86976++) {
            // futhark/microgpt.fut:389:25-81
            
            bool cond_82979 = slt64(i_86976, zl_rhs_81149);
            
            // futhark/microgpt.fut:389:56-59
            
            int64_t zeze_lhs_82980 = add64((int64_t) 1, i_86976);
            
            // futhark/microgpt.fut:389:47-60
            
            bool x_82981 = sle64((int64_t) 0, zeze_lhs_82980);
            
            // futhark/microgpt.fut:389:47-60
            
            bool y_82982 = slt64(zeze_lhs_82980, (int64_t) 16);
            
            // futhark/microgpt.fut:389:47-60
            
            bool bounds_check_82983 = x_82981 && y_82982;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_82984 = !cond_82979;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_82985 = bounds_check_82983 || loop_not_taken_82984;
            
            // futhark/microgpt.fut:389:47-60
            
            bool index_certs_82986;
            
            if (!protect_assert_disj_82985) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_82980, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:389:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:389:3-83\n   #6  futhark/microgpt.fut:447:18-38\n   #7  futhark/microgpt.fut:457:26-463:31\n   #8  futhark/microgpt.fut:479:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_83001 = ((int64_t *) seqs_mem_87829.mem)[step_81116 * (int64_t) 16 + i_86976];
            
            // futhark/microgpt.fut:449:37-51
            
            bool x_83002 = sle64((int64_t) 0, tmp_83001);
            
            // futhark/microgpt.fut:449:37-51
            
            bool y_83003 = slt64(tmp_83001, (int64_t) 27);
            
            // futhark/microgpt.fut:449:37-51
            
            bool bounds_check_83004 = x_83002 && y_83003;
            
            // futhark/microgpt.fut:449:37-51
            
            bool index_certs_83005;
            
            if (!bounds_check_83004) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_83001, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:449:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:449:16-55\n   #6  futhark/microgpt.fut:457:26-463:31\n   #7  futhark/microgpt.fut:479:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:389:47-60
            
            int64_t zeze_lhs_82987;
            
            if (cond_82979) {
                int64_t x_86788 = ((int64_t *) seqs_mem_87829.mem)[step_81116 * (int64_t) 16 + zeze_lhs_82980];
                
                zeze_lhs_82987 = x_86788;
            } else {
                zeze_lhs_82987 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86966 = 0; i_86966 < (int64_t) 27; i_86966++) {
                // futhark/microgpt.fut:389:61-65
                
                bool cond_t_res_82991 = zeze_lhs_82987 == i_86966;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_82992 = cond_82979 && cond_t_res_82991;
                
                // futhark/microgpt.fut:389:25-81
                
                double lifted_lambda_res_82993;
                
                if (x_82992) {
                    lifted_lambda_res_82993 = 1.0;
                } else {
                    lifted_lambda_res_82993 = 0.0;
                }
                ((double *) mem_87948)[i_86966] = lifted_lambda_res_82993;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86970 = 0; i_86970 < (int64_t) 16; i_86970++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83012 = ((double *) mem_param_87853.mem)[tmp_83001 * (int64_t) 16 + i_86970];
                
                ((double *) mem_87955)[i_86970] = lifted_lambda_res_83012;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87938, i_86976 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87955, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87939, i_86976 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87948, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_86991 = 0; i_86991 < (int64_t) 16; i_86991++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86981 = 0; i_86981 < (int64_t) 16; i_86981++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_83037 = ((double *) mem_param_87845.mem)[i_86991 * (int64_t) 16 + i_86981];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_83038 = ((double *) mem_87938)[i_86991 * (int64_t) 16 + i_86981];
                
                // futhark/microgpt.fut:224:39-75
                
                double zp_res_83039 = zp_lhs_83037 + zp_rhs_83038;
                
                ((double *) mem_87980)[i_86981] = zp_res_83039;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86985 = 0; i_86985 < (int64_t) 27; i_86985++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_83053 = ((double *) mem_87939)[i_86991 * (int64_t) 27 + i_86985];
                
                // futhark/microgpt.fut:260:43-85
                
                double zt_res_83054 = -6.25e-2 * zt_rhs_83053;
                
                ((double *) mem_87987)[i_86985] = zt_res_83054;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87970, i_86991 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87987, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_87971, i_86991 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_87980, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87005 = 0; i_87005 < (int64_t) 16; i_87005++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83073;
            double r_83075 = 0.0;
            
            for (int64_t i_83074 = 0; i_83074 < (int64_t) 16; i_83074++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83076 = ((double *) mem_87971)[i_87005 * (int64_t) 16 + i_83074];
                
                // futhark/microgpt.fut:225:70-103
                
                double zt_res_83077 = zt_lhs_83076 * zt_lhs_83076;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83078 = r_83075 + zt_res_83077;
                double r_tmp_89631 = zp_res_83078;
                
                r_83075 = r_tmp_89631;
            }
            defunc_0_lifted_lambda_res_83073 = r_83075;
            // futhark/microgpt.fut:225:50-121
            
            double zs_res_83079 = defunc_0_lifted_lambda_res_83073 / 16.0;
            
            // futhark/microgpt.fut:226:23-53
            
            double zp_res_83080 = 1.0e-5 + zs_res_83079;
            
            // futhark/microgpt.fut:226:15-53
            
            double sqrt_res_83081 = futrts_sqrt64(zp_res_83080);
            
            // futhark/microgpt.fut:227:25-35
            
            double zs_res_83082 = 1.0 / sqrt_res_83081;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_86998 = 0; i_86998 < (int64_t) 16; i_86998++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_84973 = ((double *) mem_87971)[i_87005 * (int64_t) 16 + i_86998];
                
                // futhark/microgpt.fut:227:5-35
                
                double zt_res_84974 = zs_res_83082 * zt_lhs_84973;
                
                // futhark/microgpt.fut:318:45-86
                
                double zt_res_84982 = zt_lhs_84973 * zt_lhs_84973;
                
                ((double *) mem_88012)[i_86998] = zt_res_84982;
                ((double *) mem_88013)[i_86998] = zt_res_84974;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88002, i_87005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88012, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88003, i_87005 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88013, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87021 = 0; i_87021 < (int64_t) 16; i_87021++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83181;
            double r_83183 = 0.0;
            
            for (int64_t i_83182 = 0; i_83182 < (int64_t) 16; i_83182++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83184 = ((double *) mem_88003)[i_87021 * (int64_t) 16 + i_83182];
                
                // futhark/microgpt.fut:228:71-106
                
                double zt_res_83185 = zt_lhs_83184 * zt_lhs_83184;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83186 = r_83183 + zt_res_83185;
                double r_tmp_89637 = zp_res_83186;
                
                r_83183 = r_tmp_89637;
            }
            defunc_0_lifted_lambda_res_83181 = r_83183;
            // futhark/microgpt.fut:228:50-124
            
            double zs_res_83187 = defunc_0_lifted_lambda_res_83181 / 16.0;
            
            // futhark/microgpt.fut:229:24-54
            
            double zp_res_83188 = 1.0e-5 + zs_res_83187;
            
            // futhark/microgpt.fut:229:16-54
            
            double sqrt_res_83189 = futrts_sqrt64(zp_res_83188);
            
            // futhark/microgpt.fut:230:25-36
            
            double zs_res_83190 = 1.0 / sqrt_res_83189;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87012 = 0; i_87012 < (int64_t) 16; i_87012++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_85002 = ((double *) mem_88003)[i_87021 * (int64_t) 16 + i_87012];
                
                // futhark/microgpt.fut:230:5-36
                
                double zt_res_85003 = zs_res_83190 * zt_lhs_85002;
                
                // futhark/microgpt.fut:311:45-86
                
                double zt_res_85011 = zt_lhs_85002 * zt_lhs_85002;
                
                ((double *) mem_88048)[i_87012] = zt_res_85011;
                ((double *) mem_88049)[i_87012] = zt_res_85003;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83224;
            double r_83226 = 0.0;
            
            for (int64_t i_83225 = 0; i_83225 < (int64_t) 16; i_83225++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_83227 = ((double *) mem_88002)[i_87021 * (int64_t) 16 + i_83225];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83228 = r_83226 + lifted_lambda_res_83227;
                double r_tmp_89640 = zp_res_83228;
                
                r_83226 = r_tmp_89640;
            }
            defunc_0_lifted_lambda_res_83224 = r_83226;
            // futhark/microgpt.fut:319:36-94
            
            double zs_res_83229 = defunc_0_lifted_lambda_res_83224 / 16.0;
            
            ((double *) mem_88034)[i_87021] = zs_res_83229;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88035, i_87021 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88048, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88036, i_87021 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88049, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87045 = 0; i_87045 < (int64_t) 16; i_87045++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87031 = 0; i_87031 < (int64_t) 16; i_87031++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85074;
                double r_85076 = 0.0;
                
                for (int64_t i_85075 = 0; i_85075 < (int64_t) 16; i_85075++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85077 = ((double *) mem_param_87849.mem)[i_87031 * (int64_t) 16 + i_85075];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85078 = ((double *) mem_88036)[i_87045 * (int64_t) 16 + i_85075];
                    
                    // futhark/microgpt.fut:231:63-102
                    
                    double zt_res_85079 = zt_lhs_85077 * zt_rhs_85078;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85080 = r_85076 + zt_res_85079;
                    double r_tmp_89649 = zp_res_85080;
                    
                    r_85076 = r_tmp_89649;
                }
                defunc_0_lifted_lambda_res_85074 = r_85076;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85087;
                double r_85089 = 0.0;
                
                for (int64_t i_85088 = 0; i_85088 < (int64_t) 16; i_85088++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85090 = ((double *) mem_param_87837.mem)[i_87031 * (int64_t) 16 + i_85088];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85091 = ((double *) mem_88036)[i_87045 * (int64_t) 16 + i_85088];
                    
                    // futhark/microgpt.fut:232:63-102
                    
                    double zt_res_85092 = zt_lhs_85090 * zt_rhs_85091;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85093 = r_85089 + zt_res_85092;
                    double r_tmp_89650 = zp_res_85093;
                    
                    r_85089 = r_tmp_89650;
                }
                defunc_0_lifted_lambda_res_85087 = r_85089;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85103;
                double r_85105 = 0.0;
                
                for (int64_t i_85104 = 0; i_85104 < (int64_t) 16; i_85104++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85106 = ((double *) mem_param_87861.mem)[i_87031 * (int64_t) 16 + i_85104];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85107 = ((double *) mem_88036)[i_87045 * (int64_t) 16 + i_85104];
                    
                    // futhark/microgpt.fut:233:63-102
                    
                    double zt_res_85108 = zt_lhs_85106 * zt_rhs_85107;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85109 = r_85105 + zt_res_85108;
                    double r_tmp_89651 = zp_res_85109;
                    
                    r_85105 = r_tmp_89651;
                }
                defunc_0_lifted_lambda_res_85103 = r_85105;
                ((double *) mem_88096)[i_87031] = defunc_0_lifted_lambda_res_85103;
                ((double *) mem_88097)[i_87031] = defunc_0_lifted_lambda_res_85087;
                ((double *) mem_88098)[i_87031] = defunc_0_lifted_lambda_res_85074;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83571;
            double r_83573 = 0.0;
            
            for (int64_t i_83572 = 0; i_83572 < (int64_t) 16; i_83572++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_83574 = ((double *) mem_88035)[i_87045 * (int64_t) 16 + i_83572];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83575 = r_83573 + lifted_lambda_res_83574;
                double r_tmp_89652 = zp_res_83575;
                
                r_83573 = r_tmp_89652;
            }
            defunc_0_lifted_lambda_res_83571 = r_83573;
            // futhark/microgpt.fut:312:36-94
            
            double zs_res_83576 = defunc_0_lifted_lambda_res_83571 / 16.0;
            
            // futhark/microgpt.fut:320:43-55
            
            double zp_lhs_83590 = ((double *) mem_88034)[i_87045];
            
            // futhark/microgpt.fut:320:43-83
            
            double zp_res_83591 = 1.0e-5 + zp_lhs_83590;
            
            // futhark/microgpt.fut:320:35-83
            
            double sqrt_res_83592 = futrts_sqrt64(zp_res_83591);
            
            ((double *) mem_88073)[i_87045] = sqrt_res_83592;
            ((double *) mem_88074)[i_87045] = zs_res_83576;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88075, i_87045 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88096, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88076, i_87045 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88097, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88077, i_87045 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88098, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87077 = 0; i_87077 < (int64_t) 4; i_87077++) {
            // futhark/microgpt.fut:234:67-70
            
            int64_t zp_lhs_83664 = mul64((int64_t) 4, i_87077);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87067 = 0; i_87067 < (int64_t) 16; i_87067++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87057 = 0; i_87057 < (int64_t) 4; i_87057++) {
                    // futhark/microgpt.fut:234:72-79
                    
                    int64_t tmp_85267 = add64(zp_lhs_83664, i_87057);
                    
                    // futhark/microgpt.fut:234:48-81
                    
                    bool x_85268 = sle64((int64_t) 0, tmp_85267);
                    
                    // futhark/microgpt.fut:234:48-81
                    
                    bool y_85269 = slt64(tmp_85267, (int64_t) 16);
                    
                    // futhark/microgpt.fut:234:48-81
                    
                    bool bounds_check_85270 = x_85268 && y_85269;
                    
                    // futhark/microgpt.fut:234:48-81
                    
                    bool index_certs_85271;
                    
                    if (!bounds_check_85270) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85267, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:234:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:234:12-82\n   #9  futhark/microgpt.fut:452:5-76\n   #10 futhark/microgpt.fut:457:26-463:31\n   #11 futhark/microgpt.fut:479:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85272 = ((double *) mem_88077)[i_87067 * (int64_t) 16 + tmp_85267];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85280 = ((double *) mem_88076)[i_87067 * (int64_t) 16 + tmp_85267];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85291 = ((double *) mem_88075)[i_87067 * (int64_t) 16 + tmp_85267];
                    
                    ((double *) mem_88168)[i_87057] = lifted_lambda_res_85291;
                    ((double *) mem_88169)[i_87057] = lifted_lambda_res_85280;
                    ((double *) mem_88170)[i_87057] = lifted_lambda_res_85272;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88153, i_87067 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88168, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88154, i_87067 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88169, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88155, i_87067 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88135, i_87077 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88153, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88136, i_87077 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88154, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88137, i_87077 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88155, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87129 = 0; i_87129 < (int64_t) 4; i_87129++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87092 = 0; i_87092 < (int64_t) 16; i_87092++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87085 = 0; i_87085 < (int64_t) 16; i_87085++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85370;
                    double r_85372 = 0.0;
                    
                    for (int64_t i_85371 = 0; i_85371 < (int64_t) 4; i_85371++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85373 = ((double *) mem_88137)[i_87129 * (int64_t) 64 + i_87092 * (int64_t) 4 + i_85371];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85374 = ((double *) mem_88136)[i_87129 * (int64_t) 64 + i_87085 * (int64_t) 4 + i_85371];
                        
                        // futhark/microgpt.fut:237:110-163
                        
                        double zt_res_85375 = zt_lhs_85373 * zt_rhs_85374;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85376 = r_85372 + zt_res_85375;
                        double r_tmp_89668 = zp_res_85376;
                        
                        r_85372 = r_tmp_89668;
                    }
                    defunc_0_lifted_lambda_res_85370 = r_85372;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85383;
                    double r_85385 = 0.0;
                    
                    for (int64_t i_85384 = 0; i_85384 < (int64_t) 4; i_85384++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85386 = ((double *) mem_88137)[i_87129 * (int64_t) 64 + i_87092 * (int64_t) 4 + i_85384];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85387 = ((double *) mem_88136)[i_87129 * (int64_t) 64 + i_87085 * (int64_t) 4 + i_85384];
                        
                        // futhark/microgpt.fut:288:75-134
                        
                        double zt_res_85388 = zt_lhs_85386 * zt_rhs_85387;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85389 = r_85385 + zt_res_85388;
                        double r_tmp_89669 = zp_res_85389;
                        
                        r_85385 = r_tmp_89669;
                    }
                    defunc_0_lifted_lambda_res_85383 = r_85385;
                    ((double *) mem_88238)[i_87085] = defunc_0_lifted_lambda_res_85383;
                    ((double *) mem_88239)[i_87085] = defunc_0_lifted_lambda_res_85370;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88228, i_87092 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88238, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88229, i_87092 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88239, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87101 = 0; i_87101 < (int64_t) 16; i_87101++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87097 = 0; i_87097 < (int64_t) 16; i_87097++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_83773 = ((double *) mem_88229)[i_87101 * (int64_t) 16 + i_87097];
                    
                    // futhark/microgpt.fut:238:47-78
                    
                    double zs_res_83774 = zs_lhs_83773 / 2.0;
                    double zp_rhs_83775 = ((double *) masks_mem_87827.mem)[step_81116 * (int64_t) 256 + i_87101 * (int64_t) 16 + i_87097];
                    
                    // futhark/microgpt.fut:238:65-102
                    
                    double zp_res_83776 = zs_res_83774 + zp_rhs_83775;
                    
                    ((double *) mem_88265)[i_87097] = zp_res_83776;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88260, i_87101 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88265, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87115 = 0; i_87115 < (int64_t) 16; i_87115++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_86809;
                double redout_87103 = -INFINITY;
                
                for (int64_t i_87104 = 0; i_87104 < (int64_t) 16; i_87104++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85407 = ((double *) mem_88260)[i_87115 * (int64_t) 16 + i_87104];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_83797 = fmax64(lifted_lambda_res_85407, redout_87103);
                    double redout_tmp_89673 = max_res_83797;
                    
                    redout_87103 = redout_tmp_89673;
                }
                defunc_0_reduce_res_86809 = redout_87103;
                // futhark/microgpt.fut:240:65-74
                
                double neg_res_83798 = -defunc_0_reduce_res_86809;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87107 = 0; i_87107 < (int64_t) 16; i_87107++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_83805 = ((double *) mem_88260)[i_87115 * (int64_t) 16 + i_87107];
                    
                    // futhark/microgpt.fut:240:43-74
                    
                    double zp_res_83806 = neg_res_83798 + zp_lhs_83805;
                    
                    // futhark/microgpt.fut:240:36-74
                    
                    double exp_res_83807 = futrts_exp64(zp_res_83806);
                    
                    ((double *) mem_88281)[i_87107] = exp_res_83807;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83809;
                double r_83811 = 0.0;
                
                for (int64_t i_83810 = 0; i_83810 < (int64_t) 16; i_83810++) {
                    // futhark/microgpt.fut:241:36-46
                    
                    double lifted_lambda_res_83812 = ((double *) mem_88281)[i_83810];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83813 = r_83811 + lifted_lambda_res_83812;
                    double r_tmp_89675 = zp_res_83813;
                    
                    r_83811 = r_tmp_89675;
                }
                defunc_0_lifted_lambda_res_83809 = r_83811;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87111 = 0; i_87111 < (int64_t) 16; i_87111++) {
                    // futhark/microgpt.fut:242:5-15
                    
                    double zs_lhs_83820 = ((double *) mem_88281)[i_87111];
                    
                    // futhark/microgpt.fut:242:5-23
                    
                    double zs_res_83821 = zs_lhs_83820 / defunc_0_lifted_lambda_res_83809;
                    
                    ((double *) mem_88288)[i_87111] = zs_res_83821;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88276, i_87115 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88288, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87123 = 0; i_87123 < (int64_t) 16; i_87123++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87119 = 0; i_87119 < (int64_t) 4; i_87119++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_83836;
                    double r_83838 = 0.0;
                    
                    for (int64_t i_83837 = 0; i_83837 < (int64_t) 16; i_83837++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_83839 = ((double *) mem_88276)[i_87123 * (int64_t) 16 + i_83837];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_83840 = ((double *) mem_88135)[i_87129 * (int64_t) 64 + i_83837 * (int64_t) 4 + i_87119];
                        
                        // futhark/microgpt.fut:243:26-72
                        
                        double zt_res_83841 = zt_lhs_83839 * zt_rhs_83840;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_83842 = r_83838 + zt_res_83841;
                        double r_tmp_89679 = zp_res_83842;
                        
                        r_83838 = r_tmp_89679;
                    }
                    defunc_0_lifted_lambda_res_83836 = r_83838;
                    ((double *) mem_88304)[i_87119] = defunc_0_lifted_lambda_res_83836;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88299, i_87123 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88304, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88216, i_87129 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88228, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88217, i_87129 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88299, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87140 = 0; i_87140 < (int64_t) 16; i_87140++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87134 = 0; i_87134 < (int64_t) 16; i_87134++) {
                // futhark/microgpt.fut:244:52-55
                
                int64_t tmp_83891 = sdiv64(i_87134, (int64_t) 4);
                
                // futhark/microgpt.fut:244:41-57
                
                bool x_83892 = sle64((int64_t) 0, tmp_83891);
                
                // futhark/microgpt.fut:244:41-57
                
                bool y_83893 = slt64(tmp_83891, (int64_t) 4);
                
                // futhark/microgpt.fut:244:41-57
                
                bool bounds_check_83894 = x_83892 && y_83893;
                
                // futhark/microgpt.fut:244:41-57
                
                bool index_certs_83895;
                
                if (!bounds_check_83894) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_83891, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:244:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:244:12-78\n   #6  futhark/microgpt.fut:452:5-76\n   #7  futhark/microgpt.fut:457:26-463:31\n   #8  futhark/microgpt.fut:479:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:244:72-75
                
                int64_t tmp_83896 = smod64(i_87134, (int64_t) 4);
                
                // futhark/microgpt.fut:244:41-77
                
                bool x_83897 = sle64((int64_t) 0, tmp_83896);
                
                // futhark/microgpt.fut:244:41-77
                
                bool y_83898 = slt64(tmp_83896, (int64_t) 4);
                
                // futhark/microgpt.fut:244:41-77
                
                bool bounds_check_83899 = x_83897 && y_83898;
                
                // futhark/microgpt.fut:244:41-77
                
                bool index_certs_83900;
                
                if (!bounds_check_83899) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_83896, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:244:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:244:12-78\n   #6  futhark/microgpt.fut:452:5-76\n   #7  futhark/microgpt.fut:457:26-463:31\n   #8  futhark/microgpt.fut:479:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_83901 = ((double *) mem_88217)[tmp_83891 * (int64_t) 64 + i_87140 * (int64_t) 4 + tmp_83896];
                
                ((double *) mem_88334)[i_87134] = lifted_lambda_res_83901;
            }
            // futhark/microgpt.fut:313:43-55
            
            double zp_lhs_83909 = ((double *) mem_88074)[i_87140];
            
            // futhark/microgpt.fut:313:43-83
            
            double zp_res_83910 = 1.0e-5 + zp_lhs_83909;
            
            // futhark/microgpt.fut:313:35-83
            
            double sqrt_res_83911 = futrts_sqrt64(zp_res_83910);
            
            ((double *) mem_88325)[i_87140] = sqrt_res_83911;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88326, i_87140 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88334, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87149 = 0; i_87149 < (int64_t) 16; i_87149++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87145 = 0; i_87145 < (int64_t) 16; i_87145++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81527;
                double r_81529 = 0.0;
                
                for (int64_t i_81528 = 0; i_81528 < (int64_t) 16; i_81528++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81530 = ((double *) mem_param_87841.mem)[i_87145 * (int64_t) 16 + i_81528];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81531 = ((double *) mem_88326)[i_87149 * (int64_t) 16 + i_81528];
                    
                    // futhark/microgpt.fut:245:63-103
                    
                    double zt_res_81532 = zt_lhs_81530 * zt_rhs_81531;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81533 = r_81529 + zt_res_81532;
                    double r_tmp_89685 = zp_res_81533;
                    
                    r_81529 = r_tmp_89685;
                }
                defunc_0_lifted_lambda_res_81527 = r_81529;
                ((double *) mem_88353)[i_87145] = defunc_0_lifted_lambda_res_81527;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88348, i_87149 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87157 = 0; i_87157 < (int64_t) 16; i_87157++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87153 = 0; i_87153 < (int64_t) 16; i_87153++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_81548 = ((double *) mem_88348)[i_87157 * (int64_t) 16 + i_87153];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_81549 = ((double *) mem_88003)[i_87157 * (int64_t) 16 + i_87153];
                
                // futhark/microgpt.fut:246:42-80
                
                double zp_res_81550 = zp_lhs_81548 + zp_rhs_81549;
                
                ((double *) mem_88369)[i_87153] = zp_res_81550;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88364, i_87157 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88369, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87170 = 0; i_87170 < (int64_t) 16; i_87170++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_83929;
            double r_83931 = 0.0;
            
            for (int64_t i_83930 = 0; i_83930 < (int64_t) 16; i_83930++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_83932 = ((double *) mem_88364)[i_87170 * (int64_t) 16 + i_83930];
                
                // futhark/microgpt.fut:247:75-114
                
                double zt_res_83933 = zt_lhs_83932 * zt_lhs_83932;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_83934 = r_83931 + zt_res_83933;
                double r_tmp_89690 = zp_res_83934;
                
                r_83931 = r_tmp_89690;
            }
            defunc_0_lifted_lambda_res_83929 = r_83931;
            // futhark/microgpt.fut:247:54-132
            
            double zs_res_83935 = defunc_0_lifted_lambda_res_83929 / 16.0;
            
            // futhark/microgpt.fut:248:24-55
            
            double zp_res_83936 = 1.0e-5 + zs_res_83935;
            
            // futhark/microgpt.fut:248:16-55
            
            double sqrt_res_83937 = futrts_sqrt64(zp_res_83936);
            
            // futhark/microgpt.fut:249:28-39
            
            double zs_res_83938 = 1.0 / sqrt_res_83937;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87163 = 0; i_87163 < (int64_t) 16; i_87163++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_85446 = ((double *) mem_88364)[i_87170 * (int64_t) 16 + i_87163];
                
                // futhark/microgpt.fut:249:5-39
                
                double zt_res_85447 = zs_res_83938 * zt_lhs_85446;
                
                // futhark/microgpt.fut:279:45-88
                
                double zt_res_85455 = zt_lhs_85446 * zt_lhs_85446;
                
                ((double *) mem_88390)[i_87163] = zt_res_85455;
                ((double *) mem_88391)[i_87163] = zt_res_85447;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88380, i_87170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88390, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88381, i_87170 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88391, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87181 = 0; i_87181 < (int64_t) 16; i_87181++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87175 = 0; i_87175 < (int64_t) 64; i_87175++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_83986;
                double r_83988 = 0.0;
                
                for (int64_t i_83987 = 0; i_83987 < (int64_t) 16; i_83987++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_83989 = ((double *) mem_param_87857.mem)[i_87175 * (int64_t) 16 + i_83987];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_83990 = ((double *) mem_88381)[i_87181 * (int64_t) 16 + i_83987];
                    
                    // futhark/microgpt.fut:250:63-102
                    
                    double zt_res_83991 = zt_lhs_83989 * zt_rhs_83990;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_83992 = r_83988 + zt_res_83991;
                    double r_tmp_89696 = zp_res_83992;
                    
                    r_83988 = r_tmp_89696;
                }
                defunc_0_lifted_lambda_res_83986 = r_83988;
                ((double *) mem_88421)[i_87175] = defunc_0_lifted_lambda_res_83986;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84000;
            double r_84002 = 0.0;
            
            for (int64_t i_84001 = 0; i_84001 < (int64_t) 16; i_84001++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_84003 = ((double *) mem_88380)[i_87181 * (int64_t) 16 + i_84001];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84004 = r_84002 + lifted_lambda_res_84003;
                double r_tmp_89697 = zp_res_84004;
                
                r_84002 = r_tmp_89697;
            }
            defunc_0_lifted_lambda_res_84000 = r_84002;
            // futhark/microgpt.fut:280:36-94
            
            double zs_res_84005 = defunc_0_lifted_lambda_res_84000 / 16.0;
            
            ((double *) mem_88412)[i_87181] = zs_res_84005;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88413, i_87181 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88421, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87192 = 0; i_87192 < (int64_t) 16; i_87192++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87186 = 0; i_87186 < (int64_t) 64; i_87186++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_84029 = ((double *) mem_88413)[i_87192 * (int64_t) 64 + i_87186];
                
                // futhark/microgpt.fut:251:41-69
                
                double max_res_84030 = fmax64(0.0, max_arg0_84029);
                
                ((double *) mem_88444)[i_87186] = max_res_84030;
            }
            // futhark/microgpt.fut:281:43-55
            
            double zp_lhs_84038 = ((double *) mem_88412)[i_87192];
            
            // futhark/microgpt.fut:281:43-83
            
            double zp_res_84039 = 1.0e-5 + zp_lhs_84038;
            
            // futhark/microgpt.fut:281:35-83
            
            double sqrt_res_84040 = futrts_sqrt64(zp_res_84039);
            
            ((double *) mem_88435)[i_87192] = sqrt_res_84040;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88436, i_87192 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87201 = 0; i_87201 < (int64_t) 16; i_87201++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87197 = 0; i_87197 < (int64_t) 16; i_87197++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81629;
                double r_81631 = 0.0;
                
                for (int64_t i_81630 = 0; i_81630 < (int64_t) 64; i_81630++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81632 = ((double *) mem_param_87833.mem)[i_87197 * (int64_t) 64 + i_81630];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81633 = ((double *) mem_88436)[i_87201 * (int64_t) 64 + i_81630];
                    
                    // futhark/microgpt.fut:252:63-104
                    
                    double zt_res_81634 = zt_lhs_81632 * zt_rhs_81633;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81635 = r_81631 + zt_res_81634;
                    double r_tmp_89703 = zp_res_81635;
                    
                    r_81631 = r_tmp_89703;
                }
                defunc_0_lifted_lambda_res_81629 = r_81631;
                ((double *) mem_88463)[i_87197] = defunc_0_lifted_lambda_res_81629;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88458, i_87201 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88463, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87209 = 0; i_87209 < (int64_t) 16; i_87209++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87205 = 0; i_87205 < (int64_t) 16; i_87205++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_81650 = ((double *) mem_88458)[i_87209 * (int64_t) 16 + i_87205];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_81651 = ((double *) mem_88364)[i_87209 * (int64_t) 16 + i_87205];
                
                // futhark/microgpt.fut:253:42-81
                
                double zp_res_81652 = zp_lhs_81650 + zp_rhs_81651;
                
                ((double *) mem_88479)[i_87205] = zp_res_81652;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88474, i_87209 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88479, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87217 = 0; i_87217 < (int64_t) 16; i_87217++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87213 = 0; i_87213 < (int64_t) 27; i_87213++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81667;
                double r_81669 = 0.0;
                
                for (int64_t i_81668 = 0; i_81668 < (int64_t) 16; i_81668++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81670 = ((double *) mem_param_87865.mem)[i_87213 * (int64_t) 16 + i_81668];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81671 = ((double *) mem_88474)[i_87217 * (int64_t) 16 + i_81668];
                    
                    // futhark/microgpt.fut:254:63-103
                    
                    double zt_res_81672 = zt_lhs_81670 * zt_rhs_81671;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81673 = r_81669 + zt_res_81672;
                    double r_tmp_89708 = zp_res_81673;
                    
                    r_81669 = r_tmp_89708;
                }
                defunc_0_lifted_lambda_res_81667 = r_81669;
                ((double *) mem_88495)[i_87213] = defunc_0_lifted_lambda_res_81667;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88490, i_87217 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88495, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87231 = 0; i_87231 < (int64_t) 16; i_87231++) {
            double x_86830;
            double x_86831;
            double redout_87219;
            double redout_87220;
            
            redout_87219 = -INFINITY;
            redout_87220 = -INFINITY;
            for (int64_t i_87221 = 0; i_87221 < (int64_t) 27; i_87221++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85523 = ((double *) mem_88490)[i_87231 * (int64_t) 27 + i_87221];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_81713 = fmax64(lifted_lambda_res_85523, redout_87219);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_81742 = fmax64(lifted_lambda_res_85523, redout_87220);
                double redout_tmp_89710 = max_res_81713;
                double redout_tmp_89711 = max_res_81742;
                
                redout_87219 = redout_tmp_89710;
                redout_87220 = redout_tmp_89711;
            }
            x_86830 = redout_87219;
            x_86831 = redout_87220;
            // futhark/microgpt.fut:262:65-74
            
            double neg_res_81714 = -x_86830;
            
            // futhark/microgpt.fut:265:65-74
            
            double neg_res_81743 = -x_86831;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_81698;
            double r_81700 = 0.0;
            
            for (int64_t i_81699 = 0; i_81699 < (int64_t) 27; i_81699++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87226 = 0; i_87226 < (int64_t) 27; i_87226++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_85562 = ((double *) mem_88490)[i_87231 * (int64_t) 27 + i_87226];
                    
                    // futhark/microgpt.fut:262:43-74
                    
                    double zp_res_85563 = neg_res_81714 + zp_lhs_85562;
                    
                    // futhark/microgpt.fut:262:36-74
                    
                    double exp_res_85564 = futrts_exp64(zp_res_85563);
                    
                    // futhark/microgpt.fut:265:43-74
                    
                    double zp_res_85572 = neg_res_81743 + zp_lhs_85562;
                    
                    // futhark/microgpt.fut:265:36-74
                    
                    double exp_res_85573 = futrts_exp64(zp_res_85572);
                    
                    ((double *) mem_88510)[i_87226] = exp_res_85573;
                    ((double *) mem_88511)[i_87226] = exp_res_85564;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81725;
                double r_81727 = 0.0;
                
                for (int64_t i_81726 = 0; i_81726 < (int64_t) 27; i_81726++) {
                    // futhark/microgpt.fut:263:36-46
                    
                    double lifted_lambda_res_81728 = ((double *) mem_88511)[i_81726];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81729 = r_81727 + lifted_lambda_res_81728;
                    double r_tmp_89715 = zp_res_81729;
                    
                    r_81727 = r_tmp_89715;
                }
                defunc_0_lifted_lambda_res_81725 = r_81727;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81754;
                double r_81756 = 0.0;
                
                for (int64_t i_81755 = 0; i_81755 < (int64_t) 27; i_81755++) {
                    // futhark/microgpt.fut:266:36-46
                    
                    double lifted_lambda_res_81757 = ((double *) mem_88510)[i_81755];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81758 = r_81756 + lifted_lambda_res_81757;
                    double r_tmp_89716 = zp_res_81758;
                    
                    r_81756 = r_tmp_89716;
                }
                defunc_0_lifted_lambda_res_81754 = r_81756;
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_81759 = ((double *) mem_87970)[i_87231 * (int64_t) 27 + i_81699];
                
                // futhark/microgpt.fut:267:38-48
                
                double zs_lhs_81760 = ((double *) mem_88511)[i_81699];
                
                // futhark/microgpt.fut:267:38-56
                
                double zs_res_81761 = zs_lhs_81760 / defunc_0_lifted_lambda_res_81725;
                
                // futhark/microgpt.fut:267:29-56
                
                double zs_res_81762 = 1.0 / zs_res_81761;
                
                // futhark/microgpt.fut:267:6-56
                
                double zt_res_81763 = zt_lhs_81759 * zs_res_81762;
                
                // futhark/microgpt.fut:267:65-75
                
                double zs_lhs_81764 = ((double *) mem_88510)[i_81699];
                
                // futhark/microgpt.fut:267:65-83
                
                double zs_res_81765 = zs_lhs_81764 / defunc_0_lifted_lambda_res_81754;
                
                // futhark/microgpt.fut:267:24-83
                
                double zt_res_81766 = zt_res_81763 * zs_res_81765;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_81767 = r_81700 + zt_res_81766;
                double r_tmp_89712 = zp_res_81767;
                
                r_81700 = r_tmp_89712;
            }
            defunc_0_lifted_lambda_res_81698 = r_81700;
            ((double *) mem_88506)[i_87231] = defunc_0_lifted_lambda_res_81698;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87249 = 0; i_87249 < (int64_t) 16; i_87249++) {
            // futhark/microgpt.fut:115:13-33
            
            double defunc_0_reduce_res_86832;
            double defunc_0_reduce_res_86833;
            double redout_87233;
            double redout_87234;
            
            redout_87233 = -INFINITY;
            redout_87234 = -INFINITY;
            for (int64_t i_87235 = 0; i_87235 < (int64_t) 27; i_87235++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_85647 = ((double *) mem_88490)[i_87249 * (int64_t) 27 + i_87235];
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_81787 = fmax64(lifted_lambda_res_85647, redout_87233);
                
                // futhark/microgpt.fut:115:13-33
                
                double max_res_81816 = fmax64(lifted_lambda_res_85647, redout_87234);
                double redout_tmp_89718 = max_res_81787;
                double redout_tmp_89719 = max_res_81816;
                
                redout_87233 = redout_tmp_89718;
                redout_87234 = redout_tmp_89719;
            }
            defunc_0_reduce_res_86832 = redout_87233;
            defunc_0_reduce_res_86833 = redout_87234;
            // futhark/microgpt.fut:269:65-74
            
            double neg_res_81788 = -defunc_0_reduce_res_86832;
            
            // futhark/microgpt.fut:272:65-74
            
            double neg_res_81817 = -defunc_0_reduce_res_86833;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87240 = 0; i_87240 < (int64_t) 27; i_87240++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_85686 = ((double *) mem_88490)[i_87249 * (int64_t) 27 + i_87240];
                
                // futhark/microgpt.fut:269:43-74
                
                double zp_res_85687 = neg_res_81788 + zp_lhs_85686;
                
                // futhark/microgpt.fut:269:36-74
                
                double exp_res_85688 = futrts_exp64(zp_res_85687);
                
                // futhark/microgpt.fut:272:43-74
                
                double zp_res_85696 = neg_res_81817 + zp_lhs_85686;
                
                // futhark/microgpt.fut:272:36-74
                
                double exp_res_85697 = futrts_exp64(zp_res_85696);
                
                ((double *) mem_88532)[i_87240] = exp_res_85697;
                ((double *) mem_88533)[i_87240] = exp_res_85688;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_81799;
            double r_81801 = 0.0;
            
            for (int64_t i_81800 = 0; i_81800 < (int64_t) 27; i_81800++) {
                // futhark/microgpt.fut:270:36-46
                
                double lifted_lambda_res_81802 = ((double *) mem_88533)[i_81800];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_81803 = r_81801 + lifted_lambda_res_81802;
                double r_tmp_89722 = zp_res_81803;
                
                r_81801 = r_tmp_89722;
            }
            defunc_0_lifted_lambda_res_81799 = r_81801;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_81828;
            double r_81830 = 0.0;
            
            for (int64_t i_81829 = 0; i_81829 < (int64_t) 27; i_81829++) {
                // futhark/microgpt.fut:273:36-46
                
                double lifted_lambda_res_81831 = ((double *) mem_88532)[i_81829];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_81832 = r_81830 + lifted_lambda_res_81831;
                double r_tmp_89723 = zp_res_81832;
                
                r_81830 = r_tmp_89723;
            }
            defunc_0_lifted_lambda_res_81828 = r_81830;
            // futhark/microgpt.fut:274:97-107
            
            double neg_arg0_81833 = ((double *) mem_88506)[i_87249];
            
            // futhark/microgpt.fut:274:91-107
            
            double neg_res_81834 = -neg_arg0_81833;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87245 = 0; i_87245 < (int64_t) 27; i_87245++) {
                // futhark/microgpt.fut:274:6-16
                
                double zs_lhs_81841 = ((double *) mem_88533)[i_87245];
                
                // futhark/microgpt.fut:274:6-24
                
                double zs_res_81842 = zs_lhs_81841 / defunc_0_lifted_lambda_res_81799;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_81843 = ((double *) mem_87970)[i_87249 * (int64_t) 27 + i_87245];
                
                // futhark/microgpt.fut:274:64-74
                
                double zs_lhs_81844 = ((double *) mem_88532)[i_87245];
                
                // futhark/microgpt.fut:274:64-82
                
                double zs_res_81845 = zs_lhs_81844 / defunc_0_lifted_lambda_res_81828;
                
                // futhark/microgpt.fut:274:55-82
                
                double zs_res_81846 = 1.0 / zs_res_81845;
                
                // futhark/microgpt.fut:274:32-82
                
                double zt_res_81847 = zt_lhs_81843 * zs_res_81846;
                
                // futhark/microgpt.fut:274:50-107
                
                double zp_res_81848 = neg_res_81834 + zt_res_81847;
                
                // futhark/microgpt.fut:274:17-107
                
                double zt_res_81849 = zs_res_81842 * zp_res_81848;
                
                ((double *) mem_88546)[i_87245] = zt_res_81849;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88527, i_87249 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88546, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87257 = 0; i_87257 < (int64_t) 16; i_87257++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87253 = 0; i_87253 < (int64_t) 16; i_87253++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81864;
                double r_81866 = 0.0;
                
                for (int64_t i_81865 = 0; i_81865 < (int64_t) 27; i_81865++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81867 = ((double *) mem_param_87865.mem)[i_81865 * (int64_t) 16 + i_87253];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81868 = ((double *) mem_88527)[i_87257 * (int64_t) 27 + i_81865];
                    
                    // futhark/microgpt.fut:275:63-103
                    
                    double zt_res_81869 = zt_lhs_81867 * zt_rhs_81868;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81870 = r_81866 + zt_res_81869;
                    double r_tmp_89727 = zp_res_81870;
                    
                    r_81866 = r_tmp_89727;
                }
                defunc_0_lifted_lambda_res_81864 = r_81866;
                ((double *) mem_88562)[i_87253] = defunc_0_lifted_lambda_res_81864;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88557, i_87257 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88562, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87270 = 0; i_87270 < (int64_t) 16; i_87270++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87263 = 0; i_87263 < (int64_t) 64; i_87263++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85721;
                double r_85723 = 0.0;
                
                for (int64_t i_85722 = 0; i_85722 < (int64_t) 16; i_85722++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85724 = ((double *) mem_param_87833.mem)[i_85722 * (int64_t) 64 + i_87263];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85725 = ((double *) mem_88557)[i_87270 * (int64_t) 16 + i_85722];
                    
                    // futhark/microgpt.fut:276:63-104
                    
                    double zt_res_85726 = zt_lhs_85724 * zt_rhs_85725;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85727 = r_85723 + zt_res_85726;
                    double r_tmp_89732 = zp_res_85727;
                    
                    r_85723 = r_tmp_89732;
                }
                defunc_0_lifted_lambda_res_85721 = r_85723;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85734;
                double r_85736 = 0.0;
                
                for (int64_t i_85735 = 0; i_85735 < (int64_t) 16; i_85735++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85737 = ((double *) mem_88557)[i_85735 * (int64_t) 16 + i_87270];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85738 = ((double *) mem_88436)[i_85735 * (int64_t) 64 + i_87263];
                    
                    // futhark/microgpt.fut:333:69-112
                    
                    double zt_res_85739 = zt_lhs_85737 * zt_rhs_85738;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85740 = r_85736 + zt_res_85739;
                    double r_tmp_89733 = zp_res_85740;
                    
                    r_85736 = r_tmp_89733;
                }
                defunc_0_lifted_lambda_res_85734 = r_85736;
                ((double *) mem_88583)[i_87263] = defunc_0_lifted_lambda_res_85734;
                ((double *) mem_88584)[i_87263] = defunc_0_lifted_lambda_res_85721;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88573, i_87270 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88583, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88574, i_87270 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88584, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87279 = 0; i_87279 < (int64_t) 16; i_87279++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87275 = 0; i_87275 < (int64_t) 64; i_87275++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_81906 = ((double *) mem_88413)[i_87279 * (int64_t) 64 + i_87275];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_81907 = fmax64(0.0, indicatorp_arg0_81906);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_81908 = fsignum64(max_res_81907);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_81909 = ((double *) mem_88574)[i_87279 * (int64_t) 64 + i_87275];
                
                // futhark/microgpt.fut:277:43-94
                
                double zt_res_81910 = sgn_res_81908 * zt_rhs_81909;
                
                ((double *) mem_88610)[i_87275] = zt_res_81910;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88605, i_87279 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88610, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87287 = 0; i_87287 < (int64_t) 16; i_87287++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87283 = 0; i_87283 < (int64_t) 16; i_87283++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_81925;
                double r_81927 = 0.0;
                
                for (int64_t i_81926 = 0; i_81926 < (int64_t) 64; i_81926++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_81928 = ((double *) mem_param_87857.mem)[i_81926 * (int64_t) 16 + i_87283];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_81929 = ((double *) mem_88605)[i_87287 * (int64_t) 64 + i_81926];
                    
                    // futhark/microgpt.fut:278:66-109
                    
                    double zt_res_81930 = zt_lhs_81928 * zt_rhs_81929;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_81931 = r_81927 + zt_res_81930;
                    double r_tmp_89738 = zp_res_81931;
                    
                    r_81927 = r_tmp_89738;
                }
                defunc_0_lifted_lambda_res_81925 = r_81927;
                ((double *) mem_88626)[i_87283] = defunc_0_lifted_lambda_res_81925;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88621, i_87287 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87291 = 0; i_87291 < (int64_t) 16; i_87291++) {
            // futhark/microgpt.fut:282:51-63
            
            double zs_rhs_81979 = ((double *) mem_88435)[i_87291];
            
            // futhark/microgpt.fut:282:43-63
            
            double zs_res_81980 = 1.0 / zs_rhs_81979;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_81981;
            double r_81983 = 0.0;
            
            for (int64_t i_81982 = 0; i_81982 < (int64_t) 16; i_81982++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_81984 = ((double *) mem_88364)[i_87291 * (int64_t) 16 + i_81982];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_81985 = ((double *) mem_88621)[i_87291 * (int64_t) 16 + i_81982];
                
                // futhark/microgpt.fut:282:93-136
                
                double zt_res_81986 = zt_lhs_81984 * zt_rhs_81985;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_81987 = r_81983 + zt_res_81986;
                double r_tmp_89740 = zp_res_81987;
                
                r_81983 = r_tmp_89740;
            }
            defunc_0_lifted_lambda_res_81981 = r_81983;
            // futhark/microgpt.fut:282:71-165
            
            double zt_res_81988 = zs_res_81980 * defunc_0_lifted_lambda_res_81981;
            
            // futhark/microgpt.fut:282:47-165
            
            double zt_res_81989 = zs_res_81980 * zt_res_81988;
            
            // futhark/microgpt.fut:282:35-165
            
            double neg_res_81990 = -zt_res_81989;
            
            ((double *) mem_88637)[i_87291] = neg_res_81990;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87295 = 0; i_87295 < (int64_t) 16; i_87295++) {
            // futhark/microgpt.fut:283:35-47
            
            double zt_lhs_81998 = ((double *) mem_88637)[i_87295];
            
            // futhark/microgpt.fut:283:89-101
            
            double zp_lhs_81999 = ((double *) mem_88412)[i_87295];
            
            // futhark/microgpt.fut:283:89-129
            
            double zp_res_82000 = 1.0e-5 + zp_lhs_81999;
            
            // futhark/microgpt.fut:283:81-129
            
            double sqrt_res_82001 = futrts_sqrt64(zp_res_82000);
            
            // futhark/microgpt.fut:283:67-131
            
            double zt_res_82002 = 2.0 * sqrt_res_82001;
            
            // futhark/microgpt.fut:283:53-131
            
            double zs_res_82003 = 1.0 / zt_res_82002;
            
            // futhark/microgpt.fut:283:35-131
            
            double zt_res_82004 = zt_lhs_81998 * zs_res_82003;
            
            ((double *) mem_88644)[i_87295] = zt_res_82004;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87299 = 0; i_87299 < (int64_t) 16; i_87299++) {
            // futhark/microgpt.fut:284:45-57
            
            double zs_lhs_82012 = ((double *) mem_88644)[i_87299];
            
            // futhark/microgpt.fut:284:45-72
            
            double zs_res_82013 = zs_lhs_82012 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_89743 = 0; nest_i_89743 < (int64_t) 16; nest_i_89743++) {
                ((double *) mem_88651)[i_87299 * (int64_t) 16 + nest_i_89743] = zs_res_82013;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87307 = 0; i_87307 < (int64_t) 16; i_87307++) {
            // futhark/microgpt.fut:285:105-117
            
            double zs_rhs_82022 = ((double *) mem_88435)[i_87307];
            
            // futhark/microgpt.fut:285:97-117
            
            double zs_res_82023 = 1.0 / zs_rhs_82022;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87303 = 0; i_87303 < (int64_t) 16; i_87303++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82030 = ((double *) mem_88557)[i_87307 * (int64_t) 16 + i_87303];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82031 = ((double *) mem_88621)[i_87307 * (int64_t) 16 + i_87303];
                
                // futhark/microgpt.fut:285:72-117
                
                double zt_res_82032 = zs_res_82023 * zt_lhs_82031;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82033 = ((double *) mem_88364)[i_87307 * (int64_t) 16 + i_87303];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82034 = ((double *) mem_88651)[i_87307 * (int64_t) 16 + i_87303];
                
                // futhark/microgpt.fut:285:125-169
                
                double zt_res_82035 = zt_lhs_82033 * zt_rhs_82034;
                
                // futhark/microgpt.fut:285:92-169
                
                double zp_res_82036 = zt_res_82032 + zt_res_82035;
                
                // futhark/microgpt.fut:285:120-221
                
                double zp_res_82037 = zt_res_82035 + zp_res_82036;
                
                // futhark/microgpt.fut:285:45-221
                
                double zp_res_82038 = zp_lhs_82030 + zp_res_82037;
                
                ((double *) mem_88666)[i_87303] = zp_res_82038;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88661, i_87307 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88666, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87320 = 0; i_87320 < (int64_t) 16; i_87320++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87313 = 0; i_87313 < (int64_t) 16; i_87313++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85763;
                double r_85765 = 0.0;
                
                for (int64_t i_85764 = 0; i_85764 < (int64_t) 16; i_85764++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85766 = ((double *) mem_param_87841.mem)[i_85764 * (int64_t) 16 + i_87313];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85767 = ((double *) mem_88661)[i_87320 * (int64_t) 16 + i_85764];
                    
                    // futhark/microgpt.fut:286:67-112
                    
                    double zt_res_85768 = zt_lhs_85766 * zt_rhs_85767;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85769 = r_85765 + zt_res_85768;
                    double r_tmp_89750 = zp_res_85769;
                    
                    r_85765 = r_tmp_89750;
                }
                defunc_0_lifted_lambda_res_85763 = r_85765;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85776;
                double r_85778 = 0.0;
                
                for (int64_t i_85777 = 0; i_85777 < (int64_t) 16; i_85777++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_85779 = ((double *) mem_88661)[i_85777 * (int64_t) 16 + i_87320];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_85780 = ((double *) mem_88326)[i_85777 * (int64_t) 16 + i_87313];
                    
                    // futhark/microgpt.fut:331:68-112
                    
                    double zt_res_85781 = zt_lhs_85779 * zt_rhs_85780;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85782 = r_85778 + zt_res_85781;
                    double r_tmp_89751 = zp_res_85782;
                    
                    r_85778 = r_tmp_89751;
                }
                defunc_0_lifted_lambda_res_85776 = r_85778;
                ((double *) mem_88687)[i_87313] = defunc_0_lifted_lambda_res_85776;
                ((double *) mem_88688)[i_87313] = defunc_0_lifted_lambda_res_85763;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88677, i_87320 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88687, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88678, i_87320 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88688, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87342 = 0; i_87342 < (int64_t) 4; i_87342++) {
            // futhark/microgpt.fut:287:74-77
            
            int64_t zp_lhs_84156 = mul64((int64_t) 4, i_87342);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87335 = 0; i_87335 < (int64_t) 16; i_87335++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87325 = 0; i_87325 < (int64_t) 4; i_87325++) {
                    // futhark/microgpt.fut:287:79-87
                    
                    int64_t tmp_85804 = add64(zp_lhs_84156, i_87325);
                    
                    // futhark/microgpt.fut:287:52-89
                    
                    bool x_85805 = sle64((int64_t) 0, tmp_85804);
                    
                    // futhark/microgpt.fut:287:52-89
                    
                    bool y_85806 = slt64(tmp_85804, (int64_t) 16);
                    
                    // futhark/microgpt.fut:287:52-89
                    
                    bool bounds_check_85807 = x_85805 && y_85806;
                    
                    // futhark/microgpt.fut:287:52-89
                    
                    bool index_certs_85808;
                    
                    if (!bounds_check_85807) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_85804, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:287:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:287:13-90\n   #9  futhark/microgpt.fut:452:5-76\n   #10 futhark/microgpt.fut:457:26-463:31\n   #11 futhark/microgpt.fut:479:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85809 = ((double *) mem_88678)[i_87335 * (int64_t) 16 + tmp_85804];
                    
                    ((double *) mem_88731)[i_87325] = lifted_lambda_res_85809;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87329 = 0; i_87329 < (int64_t) 16; i_87329++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_85823 = ((double *) mem_88216)[i_87342 * (int64_t) 256 + i_87335 * (int64_t) 16 + i_87329];
                    
                    // futhark/microgpt.fut:289:55-97
                    
                    double zs_res_85824 = zs_lhs_85823 / 2.0;
                    double zp_rhs_85825 = ((double *) masks_mem_87827.mem)[step_81116 * (int64_t) 256 + i_87335 * (int64_t) 16 + i_87329];
                    
                    // futhark/microgpt.fut:289:84-123
                    
                    double zp_res_85826 = zs_res_85824 + zp_rhs_85825;
                    
                    ((double *) mem_88738)[i_87329] = zp_res_85826;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88721, i_87335 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88738, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88722, i_87335 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88731, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88709, i_87342 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88721, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88710, i_87342 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88722, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87369 = 0; i_87369 < (int64_t) 4; i_87369++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87362 = 0; i_87362 < (int64_t) 16; i_87362++) {
                // futhark/microgpt.fut:4:11-25
                
                double defunc_0_reduce_res_86849;
                double redout_87346 = -INFINITY;
                
                for (int64_t i_87348 = 0; i_87348 < (int64_t) 16; i_87348++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_85944 = ((double *) mem_88709)[i_87369 * (int64_t) 256 + i_87362 * (int64_t) 16 + i_87348];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_85955;
                    double r_85957 = 0.0;
                    
                    for (int64_t i_85956 = 0; i_85956 < (int64_t) 4; i_85956++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_85958 = ((double *) mem_88710)[i_87369 * (int64_t) 64 + i_87362 * (int64_t) 4 + i_85956];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_85959 = ((double *) mem_88135)[i_87369 * (int64_t) 64 + i_87348 * (int64_t) 4 + i_85956];
                        
                        // futhark/microgpt.fut:294:75-135
                        
                        double zt_res_85960 = zt_lhs_85958 * zt_rhs_85959;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_85961 = r_85957 + zt_res_85960;
                        double r_tmp_89764 = zp_res_85961;
                        
                        r_85957 = r_tmp_89764;
                    }
                    defunc_0_lifted_lambda_res_85955 = r_85957;
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_85863 = fmax64(lifted_lambda_res_85944, redout_87346);
                    
                    ((double *) mem_88785)[i_87348] = defunc_0_lifted_lambda_res_85955;
                    
                    double redout_tmp_89762 = max_res_85863;
                    
                    redout_87346 = redout_tmp_89762;
                }
                defunc_0_reduce_res_86849 = redout_87346;
                // futhark/microgpt.fut:291:78-88
                
                double neg_res_85864 = -defunc_0_reduce_res_86849;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87352 = 0; i_87352 < (int64_t) 16; i_87352++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_85871 = ((double *) mem_88709)[i_87369 * (int64_t) 256 + i_87362 * (int64_t) 16 + i_87352];
                    
                    // futhark/microgpt.fut:291:45-88
                    
                    double zp_res_85872 = neg_res_85864 + zp_lhs_85871;
                    
                    // futhark/microgpt.fut:291:38-88
                    
                    double exp_res_85873 = futrts_exp64(zp_res_85872);
                    
                    ((double *) mem_88792)[i_87352] = exp_res_85873;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85875;
                double r_85877 = 0.0;
                
                for (int64_t i_85876 = 0; i_85876 < (int64_t) 16; i_85876++) {
                    // futhark/microgpt.fut:292:38-50
                    
                    double lifted_lambda_res_85878 = ((double *) mem_88792)[i_85876];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_85879 = r_85877 + lifted_lambda_res_85878;
                    double r_tmp_89766 = zp_res_85879;
                    
                    r_85877 = r_tmp_89766;
                }
                defunc_0_lifted_lambda_res_85875 = r_85877;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87356 = 0; i_87356 < (int64_t) 16; i_87356++) {
                    // futhark/microgpt.fut:293:5-17
                    
                    double zs_lhs_85886 = ((double *) mem_88792)[i_87356];
                    
                    // futhark/microgpt.fut:293:5-26
                    
                    double zs_res_85887 = zs_lhs_85886 / defunc_0_lifted_lambda_res_85875;
                    
                    ((double *) mem_88799)[i_87356] = zs_res_85887;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88775, i_87362 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88785, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88776, i_87362 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88799, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88763, i_87369 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88775, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88764, i_87369 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88776, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87393 = 0; i_87393 < (int64_t) 4; i_87393++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87386 = 0; i_87386 < (int64_t) 16; i_87386++) {
                double x_86856;
                double redout_87372 = -INFINITY;
                
                for (int64_t i_87373 = 0; i_87373 < (int64_t) 16; i_87373++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86060 = ((double *) mem_88709)[i_87393 * (int64_t) 256 + i_87386 * (int64_t) 16 + i_87373];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_86004 = fmax64(lifted_lambda_res_86060, redout_87372);
                    double redout_tmp_89772 = max_res_86004;
                    
                    redout_87372 = redout_tmp_89772;
                }
                x_86856 = redout_87372;
                // futhark/microgpt.fut:296:78-88
                
                double neg_res_86005 = -x_86856;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_85989;
                double r_85991 = 0.0;
                
                for (int64_t i_85990 = 0; i_85990 < (int64_t) 16; i_85990++) {
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_87376 = 0; i_87376 < (int64_t) 16; i_87376++) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zp_lhs_86012 = ((double *) mem_88709)[i_87393 * (int64_t) 256 + i_87386 * (int64_t) 16 + i_87376];
                        
                        // futhark/microgpt.fut:296:45-88
                        
                        double zp_res_86013 = neg_res_86005 + zp_lhs_86012;
                        
                        // futhark/microgpt.fut:296:38-88
                        
                        double exp_res_86014 = futrts_exp64(zp_res_86013);
                        
                        ((double *) mem_88844)[i_87376] = exp_res_86014;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86016;
                    double r_86018 = 0.0;
                    
                    for (int64_t i_86017 = 0; i_86017 < (int64_t) 16; i_86017++) {
                        // futhark/microgpt.fut:297:38-50
                        
                        double lifted_lambda_res_86019 = ((double *) mem_88844)[i_86017];
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86020 = r_86018 + lifted_lambda_res_86019;
                        double r_tmp_89775 = zp_res_86020;
                        
                        r_86018 = r_tmp_89775;
                    }
                    defunc_0_lifted_lambda_res_86016 = r_86018;
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86021 = ((double *) mem_88763)[i_87393 * (int64_t) 256 + i_87386 * (int64_t) 16 + i_85990];
                    
                    // futhark/microgpt.fut:298:39-51
                    
                    double zs_lhs_86022 = ((double *) mem_88844)[i_85990];
                    
                    // futhark/microgpt.fut:298:39-60
                    
                    double zs_res_86023 = zs_lhs_86022 / defunc_0_lifted_lambda_res_86016;
                    
                    // futhark/microgpt.fut:298:5-60
                    
                    double zt_res_86024 = zt_lhs_86021 * zs_res_86023;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86025 = r_85991 + zt_res_86024;
                    double r_tmp_89773 = zp_res_86025;
                    
                    r_85991 = r_tmp_89773;
                }
                defunc_0_lifted_lambda_res_85989 = r_85991;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87380 = 0; i_87380 < (int64_t) 4; i_87380++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86038;
                    double r_86040 = 0.0;
                    
                    for (int64_t i_86039 = 0; i_86039 < (int64_t) 16; i_86039++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86041 = ((double *) mem_88764)[i_87393 * (int64_t) 256 + i_86039 * (int64_t) 16 + i_87386];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86042 = ((double *) mem_88710)[i_87393 * (int64_t) 64 + i_86039 * (int64_t) 4 + i_87380];
                        
                        // futhark/microgpt.fut:304:75-136
                        
                        double zt_res_86043 = zt_lhs_86041 * zt_rhs_86042;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86044 = r_86040 + zt_res_86043;
                        double r_tmp_89777 = zp_res_86044;
                        
                        r_86040 = r_tmp_89777;
                    }
                    defunc_0_lifted_lambda_res_86038 = r_86040;
                    ((double *) mem_88851)[i_87380] = defunc_0_lifted_lambda_res_86038;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88835, i_87386 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88851, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                ((double *) mem_88836)[i_87386] = defunc_0_lifted_lambda_res_85989;
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88824, i_87393 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88835, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88825, i_87393 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88836, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87412 = 0; i_87412 < (int64_t) 4; i_87412++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87408 = 0; i_87408 < (int64_t) 16; i_87408++) {
                // futhark/microgpt.fut:115:13-33
                
                double defunc_0_reduce_res_86860;
                double redout_87396 = -INFINITY;
                
                for (int64_t i_87397 = 0; i_87397 < (int64_t) 16; i_87397++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_86090 = ((double *) mem_88709)[i_87412 * (int64_t) 256 + i_87408 * (int64_t) 16 + i_87397];
                    
                    // futhark/microgpt.fut:115:13-33
                    
                    double max_res_82290 = fmax64(lifted_lambda_res_86090, redout_87396);
                    double redout_tmp_89780 = max_res_82290;
                    
                    redout_87396 = redout_tmp_89780;
                }
                defunc_0_reduce_res_86860 = redout_87396;
                // futhark/microgpt.fut:300:78-88
                
                double neg_res_82291 = -defunc_0_reduce_res_86860;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87400 = 0; i_87400 < (int64_t) 16; i_87400++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_82298 = ((double *) mem_88709)[i_87412 * (int64_t) 256 + i_87408 * (int64_t) 16 + i_87400];
                    
                    // futhark/microgpt.fut:300:45-88
                    
                    double zp_res_82299 = neg_res_82291 + zp_lhs_82298;
                    
                    // futhark/microgpt.fut:300:38-88
                    
                    double exp_res_82300 = futrts_exp64(zp_res_82299);
                    
                    ((double *) mem_88885)[i_87400] = exp_res_82300;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82302;
                double r_82304 = 0.0;
                
                for (int64_t i_82303 = 0; i_82303 < (int64_t) 16; i_82303++) {
                    // futhark/microgpt.fut:301:38-50
                    
                    double lifted_lambda_res_82305 = ((double *) mem_88885)[i_82303];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82306 = r_82304 + lifted_lambda_res_82305;
                    double r_tmp_89782 = zp_res_82306;
                    
                    r_82304 = r_tmp_89782;
                }
                defunc_0_lifted_lambda_res_82302 = r_82304;
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_82307 = ((double *) mem_88825)[i_87412 * (int64_t) 16 + i_87408];
                
                // futhark/microgpt.fut:302:68-94
                
                double neg_res_82308 = -neg_arg0_82307;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87404 = 0; i_87404 < (int64_t) 16; i_87404++) {
                    // futhark/microgpt.fut:302:6-18
                    
                    double zs_lhs_82315 = ((double *) mem_88885)[i_87404];
                    
                    // futhark/microgpt.fut:302:6-27
                    
                    double zs_res_82316 = zs_lhs_82315 / defunc_0_lifted_lambda_res_82302;
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_82317 = ((double *) mem_88763)[i_87412 * (int64_t) 256 + i_87408 * (int64_t) 16 + i_87404];
                    
                    // futhark/microgpt.fut:302:34-94
                    
                    double zp_res_82318 = neg_res_82308 + zp_lhs_82317;
                    
                    // futhark/microgpt.fut:302:19-94
                    
                    double zt_res_82319 = zs_res_82316 * zp_res_82318;
                    
                    ((double *) mem_88892)[i_87404] = zt_res_82319;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88880, i_87408 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88874, i_87412 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88880, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87424 = 0; i_87424 < (int64_t) 4; i_87424++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87420 = 0; i_87420 < (int64_t) 16; i_87420++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87416 = 0; i_87416 < (int64_t) 16; i_87416++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_82341 = ((double *) mem_88874)[i_87424 * (int64_t) 256 + i_87420 * (int64_t) 16 + i_87416];
                    
                    // futhark/microgpt.fut:303:54-96
                    
                    double zs_res_82342 = zs_lhs_82341 / 2.0;
                    
                    ((double *) mem_88919)[i_87416] = zs_res_82342;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88914, i_87420 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88919, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88908, i_87424 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88914, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87444 = 0; i_87444 < (int64_t) 4; i_87444++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87437 = 0; i_87437 < (int64_t) 16; i_87437++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_87430 = 0; i_87430 < (int64_t) 4; i_87430++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86170;
                    double r_86172 = 0.0;
                    
                    for (int64_t i_86171 = 0; i_86171 < (int64_t) 16; i_86171++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86173 = ((double *) mem_88137)[i_87444 * (int64_t) 64 + i_86171 * (int64_t) 4 + i_87430];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86174 = ((double *) mem_88908)[i_87444 * (int64_t) 256 + i_86171 * (int64_t) 16 + i_87437];
                        
                        // futhark/microgpt.fut:305:75-135
                        
                        double zt_res_86175 = zt_lhs_86173 * zt_rhs_86174;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86176 = r_86172 + zt_res_86175;
                        double r_tmp_89793 = zp_res_86176;
                        
                        r_86172 = r_tmp_89793;
                    }
                    defunc_0_lifted_lambda_res_86170 = r_86172;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_86183;
                    double r_86185 = 0.0;
                    
                    for (int64_t i_86184 = 0; i_86184 < (int64_t) 16; i_86184++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_86186 = ((double *) mem_88908)[i_87444 * (int64_t) 256 + i_87437 * (int64_t) 16 + i_86184];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_86187 = ((double *) mem_88136)[i_87444 * (int64_t) 64 + i_86184 * (int64_t) 4 + i_87430];
                        
                        // futhark/microgpt.fut:306:75-135
                        
                        double zt_res_86188 = zt_lhs_86186 * zt_rhs_86187;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_86189 = r_86185 + zt_res_86188;
                        double r_tmp_89794 = zp_res_86189;
                        
                        r_86185 = r_tmp_89794;
                    }
                    defunc_0_lifted_lambda_res_86183 = r_86185;
                    ((double *) mem_88957)[i_87430] = defunc_0_lifted_lambda_res_86183;
                    ((double *) mem_88958)[i_87430] = defunc_0_lifted_lambda_res_86170;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88947, i_87437 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88957, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_88948, i_87437 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_88958, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88935, i_87444 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88947, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_88936, i_87444 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_88948, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87463 = 0; i_87463 < (int64_t) 16; i_87463++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87453 = 0; i_87453 < (int64_t) 16; i_87453++) {
                // futhark/microgpt.fut:307:57-60
                
                int64_t tmp_86252 = sdiv64(i_87453, (int64_t) 4);
                
                // futhark/microgpt.fut:307:44-62
                
                bool x_86253 = sle64((int64_t) 0, tmp_86252);
                
                // futhark/microgpt.fut:307:44-62
                
                bool y_86254 = slt64(tmp_86252, (int64_t) 4);
                
                // futhark/microgpt.fut:307:44-62
                
                bool bounds_check_86255 = x_86253 && y_86254;
                
                // futhark/microgpt.fut:307:44-62
                
                bool index_certs_86256;
                
                if (!bounds_check_86255) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_86252, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:307:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:307:13-85\n   #6  futhark/microgpt.fut:452:5-76\n   #7  futhark/microgpt.fut:457:26-463:31\n   #8  futhark/microgpt.fut:479:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:307:79-82
                
                int64_t tmp_86257 = smod64(i_87453, (int64_t) 4);
                
                // futhark/microgpt.fut:307:44-84
                
                bool x_86258 = sle64((int64_t) 0, tmp_86257);
                
                // futhark/microgpt.fut:307:44-84
                
                bool y_86259 = slt64(tmp_86257, (int64_t) 4);
                
                // futhark/microgpt.fut:307:44-84
                
                bool bounds_check_86260 = x_86258 && y_86259;
                
                // futhark/microgpt.fut:307:44-84
                
                bool index_certs_86261;
                
                if (!bounds_check_86260) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_86257, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:307:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:307:13-85\n   #6  futhark/microgpt.fut:452:5-76\n   #7  futhark/microgpt.fut:457:26-463:31\n   #8  futhark/microgpt.fut:479:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_86262 = ((double *) mem_88824)[tmp_86252 * (int64_t) 64 + i_87463 * (int64_t) 4 + tmp_86257];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_86275 = ((double *) mem_88936)[tmp_86252 * (int64_t) 64 + i_87463 * (int64_t) 4 + tmp_86257];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_86291 = ((double *) mem_88935)[tmp_86252 * (int64_t) 64 + i_87463 * (int64_t) 4 + tmp_86257];
                
                ((double *) mem_89004)[i_87453] = lifted_lambda_res_86291;
                ((double *) mem_89005)[i_87453] = lifted_lambda_res_86275;
                ((double *) mem_89006)[i_87453] = lifted_lambda_res_86262;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88989, i_87463 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89004, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88990, i_87463 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89005, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_88991, i_87463 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89006, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87488 = 0; i_87488 < (int64_t) 16; i_87488++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87475 = 0; i_87475 < (int64_t) 16; i_87475++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86454;
                double r_86456 = 0.0;
                
                for (int64_t i_86455 = 0; i_86455 < (int64_t) 16; i_86455++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86457 = ((double *) mem_param_87861.mem)[i_86455 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86458 = ((double *) mem_88991)[i_87488 * (int64_t) 16 + i_86455];
                    
                    // futhark/microgpt.fut:310:69-114
                    
                    double zt_res_86459 = zt_lhs_86457 * zt_rhs_86458;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86460 = r_86456 + zt_res_86459;
                    double r_tmp_89809 = zp_res_86460;
                    
                    r_86456 = r_tmp_89809;
                }
                defunc_0_lifted_lambda_res_86454 = r_86456;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86461;
                double r_86463 = 0.0;
                
                for (int64_t i_86462 = 0; i_86462 < (int64_t) 16; i_86462++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86464 = ((double *) mem_param_87837.mem)[i_86462 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86465 = ((double *) mem_88990)[i_87488 * (int64_t) 16 + i_86462];
                    
                    // futhark/microgpt.fut:310:145-190
                    
                    double zt_res_86466 = zt_lhs_86464 * zt_rhs_86465;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86467 = r_86463 + zt_res_86466;
                    double r_tmp_89810 = zp_res_86467;
                    
                    r_86463 = r_tmp_89810;
                }
                defunc_0_lifted_lambda_res_86461 = r_86463;
                // futhark/microgpt.fut:310:47-192
                
                double zp_res_86468 = defunc_0_lifted_lambda_res_86454 + defunc_0_lifted_lambda_res_86461;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86469;
                double r_86471 = 0.0;
                
                for (int64_t i_86470 = 0; i_86470 < (int64_t) 16; i_86470++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86472 = ((double *) mem_param_87849.mem)[i_86470 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86473 = ((double *) mem_88989)[i_87488 * (int64_t) 16 + i_86470];
                    
                    // futhark/microgpt.fut:310:222-267
                    
                    double zt_res_86474 = zt_lhs_86472 * zt_rhs_86473;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86475 = r_86471 + zt_res_86474;
                    double r_tmp_89811 = zp_res_86475;
                    
                    r_86471 = r_tmp_89811;
                }
                defunc_0_lifted_lambda_res_86469 = r_86471;
                // futhark/microgpt.fut:310:118-269
                
                double zp_res_86476 = zp_res_86468 + defunc_0_lifted_lambda_res_86469;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86483;
                double r_86485 = 0.0;
                
                for (int64_t i_86484 = 0; i_86484 < (int64_t) 16; i_86484++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86486 = ((double *) mem_88989)[i_86484 * (int64_t) 16 + i_87488];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86487 = ((double *) mem_88036)[i_86484 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:328:68-111
                    
                    double zt_res_86488 = zt_lhs_86486 * zt_rhs_86487;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86489 = r_86485 + zt_res_86488;
                    double r_tmp_89812 = zp_res_86489;
                    
                    r_86485 = r_tmp_89812;
                }
                defunc_0_lifted_lambda_res_86483 = r_86485;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86499;
                double r_86501 = 0.0;
                
                for (int64_t i_86500 = 0; i_86500 < (int64_t) 16; i_86500++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86502 = ((double *) mem_88990)[i_86500 * (int64_t) 16 + i_87488];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86503 = ((double *) mem_88036)[i_86500 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:329:68-111
                    
                    double zt_res_86504 = zt_lhs_86502 * zt_rhs_86503;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86505 = r_86501 + zt_res_86504;
                    double r_tmp_89813 = zp_res_86505;
                    
                    r_86501 = r_tmp_89813;
                }
                defunc_0_lifted_lambda_res_86499 = r_86501;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86517;
                double r_86519 = 0.0;
                
                for (int64_t i_86518 = 0; i_86518 < (int64_t) 16; i_86518++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86520 = ((double *) mem_88991)[i_86518 * (int64_t) 16 + i_87488];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86521 = ((double *) mem_88036)[i_86518 * (int64_t) 16 + i_87475];
                    
                    // futhark/microgpt.fut:330:68-111
                    
                    double zt_res_86522 = zt_lhs_86520 * zt_rhs_86521;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86523 = r_86519 + zt_res_86522;
                    double r_tmp_89814 = zp_res_86523;
                    
                    r_86519 = r_tmp_89814;
                }
                defunc_0_lifted_lambda_res_86517 = r_86519;
                ((double *) mem_89057)[i_87475] = defunc_0_lifted_lambda_res_86517;
                ((double *) mem_89058)[i_87475] = defunc_0_lifted_lambda_res_86499;
                ((double *) mem_89059)[i_87475] = defunc_0_lifted_lambda_res_86483;
                ((double *) mem_89060)[i_87475] = zp_res_86476;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89037, i_87488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89057, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89038, i_87488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89058, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89039, i_87488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89059, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89040, i_87488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89060, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87495 = 0; i_87495 < (int64_t) 16; i_87495++) {
            // futhark/microgpt.fut:314:51-63
            
            double zs_rhs_82575 = ((double *) mem_88325)[i_87495];
            
            // futhark/microgpt.fut:314:43-63
            
            double zs_res_82576 = 1.0 / zs_rhs_82575;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82577;
            double r_82579 = 0.0;
            
            for (int64_t i_82578 = 0; i_82578 < (int64_t) 16; i_82578++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82580 = ((double *) mem_88003)[i_87495 * (int64_t) 16 + i_82578];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82581 = ((double *) mem_89040)[i_87495 * (int64_t) 16 + i_82578];
                
                // futhark/microgpt.fut:314:93-136
                
                double zt_res_82582 = zt_lhs_82580 * zt_rhs_82581;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82583 = r_82579 + zt_res_82582;
                double r_tmp_89816 = zp_res_82583;
                
                r_82579 = r_tmp_89816;
            }
            defunc_0_lifted_lambda_res_82577 = r_82579;
            // futhark/microgpt.fut:314:71-165
            
            double zt_res_82584 = zs_res_82576 * defunc_0_lifted_lambda_res_82577;
            
            // futhark/microgpt.fut:314:47-165
            
            double zt_res_82585 = zs_res_82576 * zt_res_82584;
            
            // futhark/microgpt.fut:314:35-165
            
            double neg_res_82586 = -zt_res_82585;
            
            ((double *) mem_89101)[i_87495] = neg_res_82586;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87499 = 0; i_87499 < (int64_t) 16; i_87499++) {
            // futhark/microgpt.fut:315:35-47
            
            double zt_lhs_82594 = ((double *) mem_89101)[i_87499];
            
            // futhark/microgpt.fut:315:89-101
            
            double zp_lhs_82595 = ((double *) mem_88074)[i_87499];
            
            // futhark/microgpt.fut:315:89-129
            
            double zp_res_82596 = 1.0e-5 + zp_lhs_82595;
            
            // futhark/microgpt.fut:315:81-129
            
            double sqrt_res_82597 = futrts_sqrt64(zp_res_82596);
            
            // futhark/microgpt.fut:315:67-131
            
            double zt_res_82598 = 2.0 * sqrt_res_82597;
            
            // futhark/microgpt.fut:315:53-131
            
            double zs_res_82599 = 1.0 / zt_res_82598;
            
            // futhark/microgpt.fut:315:35-131
            
            double zt_res_82600 = zt_lhs_82594 * zs_res_82599;
            
            ((double *) mem_89108)[i_87499] = zt_res_82600;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87503 = 0; i_87503 < (int64_t) 16; i_87503++) {
            // futhark/microgpt.fut:316:45-57
            
            double zs_lhs_82608 = ((double *) mem_89108)[i_87503];
            
            // futhark/microgpt.fut:316:45-72
            
            double zs_res_82609 = zs_lhs_82608 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_89819 = 0; nest_i_89819 < (int64_t) 16; nest_i_89819++) {
                ((double *) mem_89115)[i_87503 * (int64_t) 16 + nest_i_89819] = zs_res_82609;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87511 = 0; i_87511 < (int64_t) 16; i_87511++) {
            // futhark/microgpt.fut:317:107-119
            
            double zs_rhs_82618 = ((double *) mem_88325)[i_87511];
            
            // futhark/microgpt.fut:317:99-119
            
            double zs_res_82619 = 1.0 / zs_rhs_82618;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87507 = 0; i_87507 < (int64_t) 16; i_87507++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_82626 = ((double *) mem_88661)[i_87511 * (int64_t) 16 + i_87507];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82627 = ((double *) mem_89040)[i_87511 * (int64_t) 16 + i_87507];
                
                // futhark/microgpt.fut:317:73-119
                
                double zt_res_82628 = zs_res_82619 * zt_lhs_82627;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82629 = ((double *) mem_88003)[i_87511 * (int64_t) 16 + i_87507];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82630 = ((double *) mem_89115)[i_87511 * (int64_t) 16 + i_87507];
                
                // futhark/microgpt.fut:317:127-170
                
                double zt_res_82631 = zt_lhs_82629 * zt_rhs_82630;
                
                // futhark/microgpt.fut:317:94-170
                
                double zp_res_82632 = zt_res_82628 + zt_res_82631;
                
                // futhark/microgpt.fut:317:122-221
                
                double zp_res_82633 = zt_res_82631 + zp_res_82632;
                
                // futhark/microgpt.fut:317:45-221
                
                double zp_res_82634 = zp_lhs_82626 + zp_res_82633;
                
                ((double *) mem_89130)[i_87507] = zp_res_82634;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89125, i_87511 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89130, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87515 = 0; i_87515 < (int64_t) 16; i_87515++) {
            // futhark/microgpt.fut:321:51-63
            
            double zs_rhs_82682 = ((double *) mem_88073)[i_87515];
            
            // futhark/microgpt.fut:321:43-63
            
            double zs_res_82683 = 1.0 / zs_rhs_82682;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_82684;
            double r_82686 = 0.0;
            
            for (int64_t i_82685 = 0; i_82685 < (int64_t) 16; i_82685++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_82687 = ((double *) mem_87971)[i_87515 * (int64_t) 16 + i_82685];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_82688 = ((double *) mem_89125)[i_87515 * (int64_t) 16 + i_82685];
                
                // futhark/microgpt.fut:321:93-136
                
                double zt_res_82689 = zt_lhs_82687 * zt_rhs_82688;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_82690 = r_82686 + zt_res_82689;
                double r_tmp_89823 = zp_res_82690;
                
                r_82686 = r_tmp_89823;
            }
            defunc_0_lifted_lambda_res_82684 = r_82686;
            // futhark/microgpt.fut:321:71-165
            
            double zt_res_82691 = zs_res_82683 * defunc_0_lifted_lambda_res_82684;
            
            // futhark/microgpt.fut:321:47-165
            
            double zt_res_82692 = zs_res_82683 * zt_res_82691;
            
            // futhark/microgpt.fut:321:35-165
            
            double neg_res_82693 = -zt_res_82692;
            
            ((double *) mem_89141)[i_87515] = neg_res_82693;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87519 = 0; i_87519 < (int64_t) 16; i_87519++) {
            // futhark/microgpt.fut:322:35-47
            
            double zt_lhs_82701 = ((double *) mem_89141)[i_87519];
            
            // futhark/microgpt.fut:322:89-101
            
            double zp_lhs_82702 = ((double *) mem_88034)[i_87519];
            
            // futhark/microgpt.fut:322:89-129
            
            double zp_res_82703 = 1.0e-5 + zp_lhs_82702;
            
            // futhark/microgpt.fut:322:81-129
            
            double sqrt_res_82704 = futrts_sqrt64(zp_res_82703);
            
            // futhark/microgpt.fut:322:67-131
            
            double zt_res_82705 = 2.0 * sqrt_res_82704;
            
            // futhark/microgpt.fut:322:53-131
            
            double zs_res_82706 = 1.0 / zt_res_82705;
            
            // futhark/microgpt.fut:322:35-131
            
            double zt_res_82707 = zt_lhs_82701 * zs_res_82706;
            
            ((double *) mem_89148)[i_87519] = zt_res_82707;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87523 = 0; i_87523 < (int64_t) 16; i_87523++) {
            // futhark/microgpt.fut:323:45-57
            
            double zs_lhs_82715 = ((double *) mem_89148)[i_87523];
            
            // futhark/microgpt.fut:323:45-72
            
            double zs_res_82716 = zs_lhs_82715 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_89826 = 0; nest_i_89826 < (int64_t) 16; nest_i_89826++) {
                ((double *) mem_89155)[i_87523 * (int64_t) 16 + nest_i_89826] = zs_res_82716;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87531 = 0; i_87531 < (int64_t) 16; i_87531++) {
            // futhark/microgpt.fut:324:81-93
            
            double zs_rhs_82725 = ((double *) mem_88073)[i_87531];
            
            // futhark/microgpt.fut:324:73-93
            
            double zs_res_82726 = 1.0 / zs_rhs_82725;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87527 = 0; i_87527 < (int64_t) 16; i_87527++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82733 = ((double *) mem_89125)[i_87531 * (int64_t) 16 + i_87527];
                
                // futhark/microgpt.fut:324:47-93
                
                double zt_res_82734 = zs_res_82726 * zt_lhs_82733;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_82735 = ((double *) mem_87971)[i_87531 * (int64_t) 16 + i_87527];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_82736 = ((double *) mem_89155)[i_87531 * (int64_t) 16 + i_87527];
                
                // futhark/microgpt.fut:324:101-144
                
                double zt_res_82737 = zt_lhs_82735 * zt_rhs_82736;
                
                // futhark/microgpt.fut:324:68-144
                
                double zp_res_82738 = zt_res_82734 + zt_res_82737;
                
                // futhark/microgpt.fut:324:96-195
                
                double zp_res_82739 = zt_res_82737 + zp_res_82738;
                
                ((double *) mem_89170)[i_87527] = zp_res_82739;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89165, i_87531 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89170, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87544 = 0; i_87544 < (int64_t) 16; i_87544++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87537 = 0; i_87537 < (int64_t) 16; i_87537++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_86549 = ((double *) mem_89165)[i_87544 * (int64_t) 16 + i_87537];
                
                ((double *) mem_89191)[i_87537] = lifted_lambda_res_86549;
                ((double *) mem_89192)[i_87537] = lifted_lambda_res_86549;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89181, i_87544 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89191, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89182, i_87544 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89192, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87553 = 0; i_87553 < (int64_t) 64; i_87553++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87549 = 0; i_87549 < (int64_t) 16; i_87549++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_82853;
                double r_82855 = 0.0;
                
                for (int64_t i_82854 = 0; i_82854 < (int64_t) 16; i_82854++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_82856 = ((double *) mem_88605)[i_82854 * (int64_t) 64 + i_87553];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_82857 = ((double *) mem_88381)[i_82854 * (int64_t) 16 + i_87549];
                    
                    // futhark/microgpt.fut:332:67-110
                    
                    double zt_res_82858 = zt_lhs_82856 * zt_rhs_82857;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_82859 = r_82855 + zt_res_82858;
                    double r_tmp_89835 = zp_res_82859;
                    
                    r_82855 = r_tmp_89835;
                }
                defunc_0_lifted_lambda_res_82853 = r_82855;
                ((double *) mem_89218)[i_87549] = defunc_0_lifted_lambda_res_82853;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89213, i_87553 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89218, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_87566 = 0; i_87566 < (int64_t) 27; i_87566++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_87559 = 0; i_87559 < (int64_t) 16; i_87559++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86577;
                double r_86579 = 0.0;
                
                for (int64_t i_86578 = 0; i_86578 < (int64_t) 16; i_86578++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_86580 = ((double *) mem_88527)[i_86578 * (int64_t) 27 + i_87566];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_86581 = ((double *) mem_88474)[i_86578 * (int64_t) 16 + i_87559];
                    
                    // futhark/microgpt.fut:334:68-111
                    
                    double zt_res_86582 = zt_lhs_86580 * zt_rhs_86581;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86583 = r_86579 + zt_res_86582;
                    double r_tmp_89840 = zp_res_86583;
                    
                    r_86579 = r_tmp_89840;
                }
                defunc_0_lifted_lambda_res_86577 = r_86579;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_86586;
                double r_86588 = 0.0;
                
                for (int64_t i_86587 = 0; i_86587 < (int64_t) 16; i_86587++) {
                    int64_t zeze_lhs_86589 = ((int64_t *) seqs_mem_87829.mem)[step_81116 * (int64_t) 16 + i_86587];
                    
                    // futhark/microgpt.fut:453:58-109
                    
                    bool cond_86590 = zeze_lhs_86589 == i_87566;
                    
                    // futhark/microgpt.fut:453:58-109
                    
                    double lifted_lambda_res_86591;
                    
                    if (cond_86590) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_86886 = ((double *) mem_89181)[i_86587 * (int64_t) 16 + i_87559];
                        
                        lifted_lambda_res_86591 = lifted_lambda_res_t_res_86886;
                    } else {
                        lifted_lambda_res_86591 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_86597 = r_86588 + lifted_lambda_res_86591;
                    double r_tmp_89841 = zp_res_86597;
                    
                    r_86588 = r_tmp_89841;
                }
                defunc_0_lifted_lambda_res_86586 = r_86588;
                ((double *) mem_89239)[i_87559] = defunc_0_lifted_lambda_res_86586;
                ((double *) mem_89240)[i_87559] = defunc_0_lifted_lambda_res_86577;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89229, i_87566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89239, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_89230, i_87566 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_89240, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_82937 = sitofp_i64_f64(step_81116);
        
        // futhark/microgpt.fut:409:46-65
        
        double zm_rhs_82938 = i64_res_82937 / 500.0;
        
        // futhark/microgpt.fut:409:24-65
        
        double zt_rhs_82939 = 1.0 - zm_rhs_82938;
        
        // futhark/microgpt.fut:409:19-65
        
        double lt_r_82940 = 1.0e-2 * zt_rhs_82939;
        
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_89261, (int64_t) 3456, "mem_89261")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89261.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87853.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_89263, (int64_t) 3456, "mem_89263")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89263.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87889.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_89265, (int64_t) 3456, "mem_89265")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89265.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87925.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (memblock_alloc(ctx, &mem_89267, (int64_t) 3456, "mem_89267")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:411:5-52
        // futhark/microgpt.fut:411:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89267.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89229, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:411:5-52
        if (futrts_adam_opt_w_10264(ctx, &ext_mem_89271, &ext_mem_89270, &ext_mem_89269, mem_89261, mem_89263, mem_89265, mem_89267, (int64_t) 27, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89261, "mem_89261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89263, "mem_89263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89265, "mem_89265") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89267, "mem_89267") != 0)
            return 1;
        // futhark/microgpt.fut:413:5-52
        if (memblock_alloc(ctx, &mem_89272, (int64_t) 2048, "mem_89272")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-52
        // futhark/microgpt.fut:413:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89272.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87845.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:413:5-52
        if (memblock_alloc(ctx, &mem_89274, (int64_t) 2048, "mem_89274")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-52
        // futhark/microgpt.fut:413:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89274.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87881.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:413:5-52
        if (memblock_alloc(ctx, &mem_89276, (int64_t) 2048, "mem_89276")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-52
        // futhark/microgpt.fut:413:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89276.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87917.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:413:5-52
        if (memblock_alloc(ctx, &mem_89278, (int64_t) 2048, "mem_89278")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:413:5-52
        // futhark/microgpt.fut:413:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89278.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89182, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:413:5-52
        if (futrts_adam_opt_w_10265(ctx, &ext_mem_89282, &ext_mem_89281, &ext_mem_89280, mem_89272, mem_89274, mem_89276, mem_89278, (int64_t) 16, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89272, "mem_89272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89274, "mem_89274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89276, "mem_89276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89278, "mem_89278") != 0)
            return 1;
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_89283, (int64_t) 2048, "mem_89283")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89283.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87849.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_89285, (int64_t) 2048, "mem_89285")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89285.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87885.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_89287, (int64_t) 2048, "mem_89287")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89287.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87921.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (memblock_alloc(ctx, &mem_89289, (int64_t) 2048, "mem_89289")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:415:5-56
        // futhark/microgpt.fut:415:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89289.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89039, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:415:5-56
        if (futrts_adam_opt_w_10265(ctx, &ext_mem_89293, &ext_mem_89292, &ext_mem_89291, mem_89283, mem_89285, mem_89287, mem_89289, (int64_t) 16, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89283, "mem_89283") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89285, "mem_89285") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89287, "mem_89287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89289, "mem_89289") != 0)
            return 1;
        // futhark/microgpt.fut:417:5-56
        if (memblock_alloc(ctx, &mem_89294, (int64_t) 2048, "mem_89294")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:417:5-56
        // futhark/microgpt.fut:417:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89294.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87837.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:417:5-56
        if (memblock_alloc(ctx, &mem_89296, (int64_t) 2048, "mem_89296")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:417:5-56
        // futhark/microgpt.fut:417:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89296.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87873.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:417:5-56
        if (memblock_alloc(ctx, &mem_89298, (int64_t) 2048, "mem_89298")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:417:5-56
        // futhark/microgpt.fut:417:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89298.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87909.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:417:5-56
        if (memblock_alloc(ctx, &mem_89300, (int64_t) 2048, "mem_89300")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:417:5-56
        // futhark/microgpt.fut:417:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89300.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89038, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:417:5-56
        if (futrts_adam_opt_w_10265(ctx, &ext_mem_89304, &ext_mem_89303, &ext_mem_89302, mem_89294, mem_89296, mem_89298, mem_89300, (int64_t) 16, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89294, "mem_89294") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89296, "mem_89296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89298, "mem_89298") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89300, "mem_89300") != 0)
            return 1;
        // futhark/microgpt.fut:419:5-56
        if (memblock_alloc(ctx, &mem_89305, (int64_t) 2048, "mem_89305")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:419:5-56
        // futhark/microgpt.fut:419:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89305.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87861.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:419:5-56
        if (memblock_alloc(ctx, &mem_89307, (int64_t) 2048, "mem_89307")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:419:5-56
        // futhark/microgpt.fut:419:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89307.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87897.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:419:5-56
        if (memblock_alloc(ctx, &mem_89309, (int64_t) 2048, "mem_89309")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:419:5-56
        // futhark/microgpt.fut:419:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89309.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87933.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:419:5-56
        if (memblock_alloc(ctx, &mem_89311, (int64_t) 2048, "mem_89311")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:419:5-56
        // futhark/microgpt.fut:419:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89311.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89037, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:419:5-56
        if (futrts_adam_opt_w_10265(ctx, &ext_mem_89315, &ext_mem_89314, &ext_mem_89313, mem_89305, mem_89307, mem_89309, mem_89311, (int64_t) 16, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89305, "mem_89305") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89307, "mem_89307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89309, "mem_89309") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89311, "mem_89311") != 0)
            return 1;
        // futhark/microgpt.fut:421:5-56
        if (memblock_alloc(ctx, &mem_89316, (int64_t) 2048, "mem_89316")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:421:5-56
        // futhark/microgpt.fut:421:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89316.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87841.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:421:5-56
        if (memblock_alloc(ctx, &mem_89318, (int64_t) 2048, "mem_89318")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:421:5-56
        // futhark/microgpt.fut:421:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89318.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87877.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:421:5-56
        if (memblock_alloc(ctx, &mem_89320, (int64_t) 2048, "mem_89320")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:421:5-56
        // futhark/microgpt.fut:421:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89320.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87913.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:421:5-56
        if (memblock_alloc(ctx, &mem_89322, (int64_t) 2048, "mem_89322")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:421:5-56
        // futhark/microgpt.fut:421:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89322.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_88677, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:421:5-56
        if (futrts_adam_opt_w_10265(ctx, &ext_mem_89326, &ext_mem_89325, &ext_mem_89324, mem_89316, mem_89318, mem_89320, mem_89322, (int64_t) 16, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89316, "mem_89316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89318, "mem_89318") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89320, "mem_89320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89322, "mem_89322") != 0)
            return 1;
        // futhark/microgpt.fut:423:5-52
        if (memblock_alloc(ctx, &mem_89327, (int64_t) 8192, "mem_89327")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:423:5-52
        // futhark/microgpt.fut:423:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89327.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87857.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:423:5-52
        if (memblock_alloc(ctx, &mem_89329, (int64_t) 8192, "mem_89329")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:423:5-52
        // futhark/microgpt.fut:423:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89329.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87893.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:423:5-52
        if (memblock_alloc(ctx, &mem_89331, (int64_t) 8192, "mem_89331")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:423:5-52
        // futhark/microgpt.fut:423:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89331.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87929.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:423:5-52
        if (memblock_alloc(ctx, &mem_89333, (int64_t) 8192, "mem_89333")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:423:5-52
        // futhark/microgpt.fut:423:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89333.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89213, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:423:5-52
        if (futrts_adam_opt_w_10264(ctx, &ext_mem_89337, &ext_mem_89336, &ext_mem_89335, mem_89327, mem_89329, mem_89331, mem_89333, (int64_t) 64, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89327, "mem_89327") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89329, "mem_89329") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89331, "mem_89331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89333, "mem_89333") != 0)
            return 1;
        // futhark/microgpt.fut:425:5-60
        if (memblock_alloc(ctx, &mem_89338, (int64_t) 8192, "mem_89338")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:425:5-60
        // futhark/microgpt.fut:425:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89338.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_87833.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:425:5-60
        if (memblock_alloc(ctx, &mem_89340, (int64_t) 8192, "mem_89340")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:425:5-60
        // futhark/microgpt.fut:425:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89340.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_87869.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:425:5-60
        if (memblock_alloc(ctx, &mem_89342, (int64_t) 8192, "mem_89342")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:425:5-60
        // futhark/microgpt.fut:425:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89342.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_87905.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:425:5-60
        if (memblock_alloc(ctx, &mem_89344, (int64_t) 8192, "mem_89344")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:425:5-60
        // futhark/microgpt.fut:425:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89344.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_88573, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:425:5-60
        if (futrts_adam_opt_w_10264(ctx, &ext_mem_89348, &ext_mem_89347, &ext_mem_89346, mem_89338, mem_89340, mem_89342, mem_89344, (int64_t) 16, (int64_t) 64, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89338, "mem_89338") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89340, "mem_89340") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89342, "mem_89342") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89344, "mem_89344") != 0)
            return 1;
        // futhark/microgpt.fut:427:5-56
        if (memblock_alloc(ctx, &mem_89349, (int64_t) 3456, "mem_89349")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:427:5-56
        // futhark/microgpt.fut:427:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89349.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87865.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:427:5-56
        if (memblock_alloc(ctx, &mem_89351, (int64_t) 3456, "mem_89351")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:427:5-56
        // futhark/microgpt.fut:427:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89351.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87901.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:427:5-56
        if (memblock_alloc(ctx, &mem_89353, (int64_t) 3456, "mem_89353")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:427:5-56
        // futhark/microgpt.fut:427:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89353.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_87937.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:427:5-56
        if (memblock_alloc(ctx, &mem_89355, (int64_t) 3456, "mem_89355")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:427:5-56
        // futhark/microgpt.fut:427:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_89355.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_89230, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:427:5-56
        if (futrts_adam_opt_w_10264(ctx, &ext_mem_89359, &ext_mem_89358, &ext_mem_89357, mem_89349, mem_89351, mem_89353, mem_89355, (int64_t) 27, (int64_t) 16, step_81116, lt_r_82940) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_89349, "mem_89349") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89351, "mem_89351") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89353, "mem_89353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89355, "mem_89355") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89567, &ext_mem_89348, "ext_mem_89348") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89568, &ext_mem_89304, "ext_mem_89304") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89569, &ext_mem_89326, "ext_mem_89326") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89570, &ext_mem_89282, "ext_mem_89282") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89571, &ext_mem_89293, "ext_mem_89293") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89572, &ext_mem_89271, "ext_mem_89271") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89573, &ext_mem_89337, "ext_mem_89337") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89574, &ext_mem_89315, "ext_mem_89315") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89575, &ext_mem_89359, "ext_mem_89359") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89576, &ext_mem_89347, "ext_mem_89347") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89577, &ext_mem_89303, "ext_mem_89303") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89578, &ext_mem_89325, "ext_mem_89325") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89579, &ext_mem_89281, "ext_mem_89281") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89580, &ext_mem_89292, "ext_mem_89292") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89581, &ext_mem_89270, "ext_mem_89270") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89582, &ext_mem_89336, "ext_mem_89336") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89583, &ext_mem_89314, "ext_mem_89314") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89584, &ext_mem_89358, "ext_mem_89358") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89585, &ext_mem_89346, "ext_mem_89346") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89586, &ext_mem_89302, "ext_mem_89302") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89587, &ext_mem_89324, "ext_mem_89324") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89588, &ext_mem_89280, "ext_mem_89280") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89589, &ext_mem_89291, "ext_mem_89291") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89590, &ext_mem_89269, "ext_mem_89269") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89591, &ext_mem_89335, "ext_mem_89335") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89592, &ext_mem_89313, "ext_mem_89313") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_89593, &ext_mem_89357, "ext_mem_89357") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87833, &mem_param_tmp_89567, "mem_param_tmp_89567") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87837, &mem_param_tmp_89568, "mem_param_tmp_89568") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87841, &mem_param_tmp_89569, "mem_param_tmp_89569") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87845, &mem_param_tmp_89570, "mem_param_tmp_89570") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87849, &mem_param_tmp_89571, "mem_param_tmp_89571") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87853, &mem_param_tmp_89572, "mem_param_tmp_89572") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87857, &mem_param_tmp_89573, "mem_param_tmp_89573") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87861, &mem_param_tmp_89574, "mem_param_tmp_89574") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87865, &mem_param_tmp_89575, "mem_param_tmp_89575") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87869, &mem_param_tmp_89576, "mem_param_tmp_89576") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87873, &mem_param_tmp_89577, "mem_param_tmp_89577") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87877, &mem_param_tmp_89578, "mem_param_tmp_89578") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87881, &mem_param_tmp_89579, "mem_param_tmp_89579") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87885, &mem_param_tmp_89580, "mem_param_tmp_89580") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87889, &mem_param_tmp_89581, "mem_param_tmp_89581") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87893, &mem_param_tmp_89582, "mem_param_tmp_89582") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87897, &mem_param_tmp_89583, "mem_param_tmp_89583") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87901, &mem_param_tmp_89584, "mem_param_tmp_89584") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87905, &mem_param_tmp_89585, "mem_param_tmp_89585") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87909, &mem_param_tmp_89586, "mem_param_tmp_89586") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87913, &mem_param_tmp_89587, "mem_param_tmp_89587") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87917, &mem_param_tmp_89588, "mem_param_tmp_89588") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87921, &mem_param_tmp_89589, "mem_param_tmp_89589") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87925, &mem_param_tmp_89590, "mem_param_tmp_89590") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87929, &mem_param_tmp_89591, "mem_param_tmp_89591") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87933, &mem_param_tmp_89592, "mem_param_tmp_89592") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_87937, &mem_param_tmp_89593, "mem_param_tmp_89593") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_89467, &mem_param_87833, "mem_param_87833") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89466, &mem_param_87837, "mem_param_87837") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89465, &mem_param_87841, "mem_param_87841") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89464, &mem_param_87845, "mem_param_87845") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89463, &mem_param_87849, "mem_param_87849") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89462, &mem_param_87853, "mem_param_87853") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89461, &mem_param_87857, "mem_param_87857") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89460, &mem_param_87861, "mem_param_87861") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89459, &mem_param_87865, "mem_param_87865") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89458, &mem_param_87869, "mem_param_87869") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89457, &mem_param_87873, "mem_param_87873") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89456, &mem_param_87877, "mem_param_87877") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89455, &mem_param_87881, "mem_param_87881") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89454, &mem_param_87885, "mem_param_87885") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89453, &mem_param_87889, "mem_param_87889") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89452, &mem_param_87893, "mem_param_87893") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89451, &mem_param_87897, "mem_param_87897") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89450, &mem_param_87901, "mem_param_87901") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89449, &mem_param_87905, "mem_param_87905") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89448, &mem_param_87909, "mem_param_87909") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89447, &mem_param_87913, "mem_param_87913") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89446, &mem_param_87917, "mem_param_87917") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89445, &mem_param_87921, "mem_param_87921") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89444, &mem_param_87925, "mem_param_87925") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89443, &mem_param_87929, "mem_param_87929") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89442, &mem_param_87933, "mem_param_87933") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_89441, &mem_param_87937, "mem_param_87937") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89540, &ext_mem_89462, "ext_mem_89462") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89541, &ext_mem_89464, "ext_mem_89464") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89542, &ext_mem_89463, "ext_mem_89463") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89543, &ext_mem_89466, "ext_mem_89466") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89544, &ext_mem_89460, "ext_mem_89460") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89545, &ext_mem_89465, "ext_mem_89465") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89546, &ext_mem_89461, "ext_mem_89461") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89547, &ext_mem_89467, "ext_mem_89467") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89548, &ext_mem_89459, "ext_mem_89459") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89549, &ext_mem_89453, "ext_mem_89453") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89550, &ext_mem_89455, "ext_mem_89455") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89551, &ext_mem_89454, "ext_mem_89454") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89552, &ext_mem_89457, "ext_mem_89457") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89553, &ext_mem_89451, "ext_mem_89451") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89554, &ext_mem_89456, "ext_mem_89456") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89555, &ext_mem_89452, "ext_mem_89452") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89556, &ext_mem_89458, "ext_mem_89458") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89557, &ext_mem_89450, "ext_mem_89450") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89558, &ext_mem_89444, "ext_mem_89444") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89559, &ext_mem_89446, "ext_mem_89446") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89560, &ext_mem_89445, "ext_mem_89445") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89561, &ext_mem_89448, "ext_mem_89448") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89562, &ext_mem_89442, "ext_mem_89442") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89563, &ext_mem_89447, "ext_mem_89447") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89564, &ext_mem_89443, "ext_mem_89443") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89565, &ext_mem_89449, "ext_mem_89449") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89566, &ext_mem_89441, "ext_mem_89441") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89933, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89934, &mem_out_89541, "mem_out_89541") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89935, &mem_out_89542, "mem_out_89542") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89936, &mem_out_89543, "mem_out_89543") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89937, &mem_out_89544, "mem_out_89544") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89938, &mem_out_89545, "mem_out_89545") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89939, &mem_out_89546, "mem_out_89546") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89940, &mem_out_89547, "mem_out_89547") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89941, &mem_out_89548, "mem_out_89548") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89942, &mem_out_89549, "mem_out_89549") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89943, &mem_out_89550, "mem_out_89550") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89944, &mem_out_89551, "mem_out_89551") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89945, &mem_out_89552, "mem_out_89552") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89946, &mem_out_89553, "mem_out_89553") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89947, &mem_out_89554, "mem_out_89554") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89948, &mem_out_89555, "mem_out_89555") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89949, &mem_out_89556, "mem_out_89556") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89950, &mem_out_89557, "mem_out_89557") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89951, &mem_out_89558, "mem_out_89558") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89952, &mem_out_89559, "mem_out_89559") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89953, &mem_out_89560, "mem_out_89560") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89954, &mem_out_89561, "mem_out_89561") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89955, &mem_out_89562, "mem_out_89562") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89956, &mem_out_89563, "mem_out_89563") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89957, &mem_out_89564, "mem_out_89564") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89958, &mem_out_89565, "mem_out_89565") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_89959, &mem_out_89566, "mem_out_89566") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_87938);
        free(mem_87939);
        free(mem_87948);
        free(mem_87955);
        free(mem_87970);
        free(mem_87971);
        free(mem_87980);
        free(mem_87987);
        free(mem_88002);
        free(mem_88003);
        free(mem_88012);
        free(mem_88013);
        free(mem_88034);
        free(mem_88035);
        free(mem_88036);
        free(mem_88048);
        free(mem_88049);
        free(mem_88073);
        free(mem_88074);
        free(mem_88075);
        free(mem_88076);
        free(mem_88077);
        free(mem_88096);
        free(mem_88097);
        free(mem_88098);
        free(mem_88135);
        free(mem_88136);
        free(mem_88137);
        free(mem_88153);
        free(mem_88154);
        free(mem_88155);
        free(mem_88168);
        free(mem_88169);
        free(mem_88170);
        free(mem_88216);
        free(mem_88217);
        free(mem_88228);
        free(mem_88229);
        free(mem_88238);
        free(mem_88239);
        free(mem_88260);
        free(mem_88265);
        free(mem_88276);
        free(mem_88281);
        free(mem_88288);
        free(mem_88299);
        free(mem_88304);
        free(mem_88325);
        free(mem_88326);
        free(mem_88334);
        free(mem_88348);
        free(mem_88353);
        free(mem_88364);
        free(mem_88369);
        free(mem_88380);
        free(mem_88381);
        free(mem_88390);
        free(mem_88391);
        free(mem_88412);
        free(mem_88413);
        free(mem_88421);
        free(mem_88435);
        free(mem_88436);
        free(mem_88444);
        free(mem_88458);
        free(mem_88463);
        free(mem_88474);
        free(mem_88479);
        free(mem_88490);
        free(mem_88495);
        free(mem_88506);
        free(mem_88510);
        free(mem_88511);
        free(mem_88527);
        free(mem_88532);
        free(mem_88533);
        free(mem_88546);
        free(mem_88557);
        free(mem_88562);
        free(mem_88573);
        free(mem_88574);
        free(mem_88583);
        free(mem_88584);
        free(mem_88605);
        free(mem_88610);
        free(mem_88621);
        free(mem_88626);
        free(mem_88637);
        free(mem_88644);
        free(mem_88651);
        free(mem_88661);
        free(mem_88666);
        free(mem_88677);
        free(mem_88678);
        free(mem_88687);
        free(mem_88688);
        free(mem_88709);
        free(mem_88710);
        free(mem_88721);
        free(mem_88722);
        free(mem_88731);
        free(mem_88738);
        free(mem_88763);
        free(mem_88764);
        free(mem_88775);
        free(mem_88776);
        free(mem_88785);
        free(mem_88792);
        free(mem_88799);
        free(mem_88824);
        free(mem_88825);
        free(mem_88835);
        free(mem_88836);
        free(mem_88844);
        free(mem_88851);
        free(mem_88874);
        free(mem_88880);
        free(mem_88885);
        free(mem_88892);
        free(mem_88908);
        free(mem_88914);
        free(mem_88919);
        free(mem_88935);
        free(mem_88936);
        free(mem_88947);
        free(mem_88948);
        free(mem_88957);
        free(mem_88958);
        free(mem_88989);
        free(mem_88990);
        free(mem_88991);
        free(mem_89004);
        free(mem_89005);
        free(mem_89006);
        free(mem_89037);
        free(mem_89038);
        free(mem_89039);
        free(mem_89040);
        free(mem_89057);
        free(mem_89058);
        free(mem_89059);
        free(mem_89060);
        free(mem_89101);
        free(mem_89108);
        free(mem_89115);
        free(mem_89125);
        free(mem_89130);
        free(mem_89141);
        free(mem_89148);
        free(mem_89155);
        free(mem_89165);
        free(mem_89170);
        free(mem_89181);
        free(mem_89182);
        free(mem_89191);
        free(mem_89192);
        free(mem_89213);
        free(mem_89218);
        free(mem_89229);
        free(mem_89230);
        free(mem_89239);
        free(mem_89240);
        if (memblock_unref(ctx, &mem_param_tmp_89593, "mem_param_tmp_89593") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89592, "mem_param_tmp_89592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89591, "mem_param_tmp_89591") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89590, "mem_param_tmp_89590") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89589, "mem_param_tmp_89589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89588, "mem_param_tmp_89588") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89587, "mem_param_tmp_89587") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89586, "mem_param_tmp_89586") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89585, "mem_param_tmp_89585") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89584, "mem_param_tmp_89584") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89583, "mem_param_tmp_89583") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89582, "mem_param_tmp_89582") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89581, "mem_param_tmp_89581") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89580, "mem_param_tmp_89580") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89579, "mem_param_tmp_89579") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89578, "mem_param_tmp_89578") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89577, "mem_param_tmp_89577") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89576, "mem_param_tmp_89576") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89575, "mem_param_tmp_89575") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89574, "mem_param_tmp_89574") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89573, "mem_param_tmp_89573") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89572, "mem_param_tmp_89572") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89571, "mem_param_tmp_89571") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89570, "mem_param_tmp_89570") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89569, "mem_param_tmp_89569") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89568, "mem_param_tmp_89568") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_89567, "mem_param_tmp_89567") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89357, "ext_mem_89357") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89358, "ext_mem_89358") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89359, "ext_mem_89359") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89355, "mem_89355") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89353, "mem_89353") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89351, "mem_89351") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89349, "mem_89349") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89346, "ext_mem_89346") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89347, "ext_mem_89347") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89348, "ext_mem_89348") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89344, "mem_89344") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89342, "mem_89342") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89340, "mem_89340") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89338, "mem_89338") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89335, "ext_mem_89335") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89336, "ext_mem_89336") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89337, "ext_mem_89337") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89333, "mem_89333") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89331, "mem_89331") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89329, "mem_89329") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89327, "mem_89327") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89324, "ext_mem_89324") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89325, "ext_mem_89325") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89326, "ext_mem_89326") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89322, "mem_89322") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89320, "mem_89320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89318, "mem_89318") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89316, "mem_89316") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89313, "ext_mem_89313") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89314, "ext_mem_89314") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89315, "ext_mem_89315") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89311, "mem_89311") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89309, "mem_89309") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89307, "mem_89307") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89305, "mem_89305") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89302, "ext_mem_89302") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89303, "ext_mem_89303") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89304, "ext_mem_89304") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89300, "mem_89300") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89298, "mem_89298") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89296, "mem_89296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89294, "mem_89294") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89291, "ext_mem_89291") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89292, "ext_mem_89292") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89293, "ext_mem_89293") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89289, "mem_89289") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89287, "mem_89287") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89285, "mem_89285") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89283, "mem_89283") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89280, "ext_mem_89280") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89281, "ext_mem_89281") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89282, "ext_mem_89282") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89278, "mem_89278") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89276, "mem_89276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89274, "mem_89274") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89272, "mem_89272") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89269, "ext_mem_89269") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89270, "ext_mem_89270") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89271, "ext_mem_89271") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89267, "mem_89267") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89265, "mem_89265") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89263, "mem_89263") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_89261, "mem_89261") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87937, "mem_param_87937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87933, "mem_param_87933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87929, "mem_param_87929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87925, "mem_param_87925") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87921, "mem_param_87921") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87917, "mem_param_87917") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87913, "mem_param_87913") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87909, "mem_param_87909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87905, "mem_param_87905") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87901, "mem_param_87901") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87897, "mem_param_87897") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87893, "mem_param_87893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87889, "mem_param_87889") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87885, "mem_param_87885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87881, "mem_param_87881") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87877, "mem_param_87877") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87873, "mem_param_87873") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87869, "mem_param_87869") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87865, "mem_param_87865") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87861, "mem_param_87861") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87857, "mem_param_87857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87853, "mem_param_87853") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87849, "mem_param_87849") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87845, "mem_param_87845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87841, "mem_param_87841") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87837, "mem_param_87837") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_87833, "mem_param_87833") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89441, "ext_mem_89441") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89442, "ext_mem_89442") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89443, "ext_mem_89443") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89444, "ext_mem_89444") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89445, "ext_mem_89445") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89446, "ext_mem_89446") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89447, "ext_mem_89447") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89448, "ext_mem_89448") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89449, "ext_mem_89449") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89450, "ext_mem_89450") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89451, "ext_mem_89451") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89452, "ext_mem_89452") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89453, "ext_mem_89453") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89454, "ext_mem_89454") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89455, "ext_mem_89455") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89456, "ext_mem_89456") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89457, "ext_mem_89457") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89458, "ext_mem_89458") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89459, "ext_mem_89459") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89460, "ext_mem_89460") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89461, "ext_mem_89461") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89462, "ext_mem_89462") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89463, "ext_mem_89463") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89464, "ext_mem_89464") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89465, "ext_mem_89465") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89466, "ext_mem_89466") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_89467, "ext_mem_89467") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89566, "mem_out_89566") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89565, "mem_out_89565") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89564, "mem_out_89564") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89563, "mem_out_89563") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89562, "mem_out_89562") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89561, "mem_out_89561") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89560, "mem_out_89560") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89559, "mem_out_89559") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89558, "mem_out_89558") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89557, "mem_out_89557") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89556, "mem_out_89556") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89555, "mem_out_89555") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89554, "mem_out_89554") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89553, "mem_out_89553") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89552, "mem_out_89552") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89551, "mem_out_89551") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89550, "mem_out_89550") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89549, "mem_out_89549") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89548, "mem_out_89548") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89547, "mem_out_89547") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89546, "mem_out_89546") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89545, "mem_out_89545") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89544, "mem_out_89544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89543, "mem_out_89543") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89542, "mem_out_89542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89541, "mem_out_89541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_90122, struct memblock *mem_out_p_90123, struct memblock *mem_out_p_90124, struct memblock *mem_out_p_90125, struct memblock *mem_out_p_90126, struct memblock *mem_out_p_90127, struct memblock *mem_out_p_90128, struct memblock *mem_out_p_90129, struct memblock *mem_out_p_90130)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mem_87791 = ctx->constants->mem_87791;
    struct memblock mem_87792 = ctx->constants->mem_87792;
    struct memblock mem_87793 = ctx->constants->mem_87793;
    struct memblock mem_87794 = ctx->constants->mem_87794;
    struct memblock mem_87795 = ctx->constants->mem_87795;
    struct memblock mem_87796 = ctx->constants->mem_87796;
    struct memblock mem_87797 = ctx->constants->mem_87797;
    struct memblock mem_87798 = ctx->constants->mem_87798;
    struct memblock mem_87799 = ctx->constants->mem_87799;
    
    if (memblock_set(ctx, &mem_out_89540, &mem_87798, "mem_87798") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89541, &mem_87794, "mem_87794") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89542, &mem_87796, "mem_87796") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89543, &mem_87792, "mem_87792") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89544, &mem_87793, "mem_87793") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89545, &mem_87791, "mem_87791") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89546, &mem_87797, "mem_87797") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89547, &mem_87795, "mem_87795") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_89548, &mem_87799, "mem_87799") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90122, &mem_out_89540, "mem_out_89540") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90123, &mem_out_89541, "mem_out_89541") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90124, &mem_out_89542, "mem_out_89542") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90125, &mem_out_89543, "mem_out_89543") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90126, &mem_out_89544, "mem_out_89544") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90127, &mem_out_89545, "mem_out_89545") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90128, &mem_out_89546, "mem_out_89546") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90129, &mem_out_89547, "mem_out_89547") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_90130, &mem_out_89548, "mem_out_89548") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_89548, "mem_out_89548") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89547, "mem_out_89547") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89546, "mem_out_89546") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89545, "mem_out_89545") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89544, "mem_out_89544") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89543, "mem_out_89543") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89542, "mem_out_89542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89541, "mem_out_89541") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_89540, "mem_out_89540") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock mask_mem_87810;
    
    mask_mem_87810.references = NULL;
    
    struct memblock tokens_mem_87809;
    
    tokens_mem_87809.references = NULL;
    
    struct memblock wvoc_mem_87808;
    
    wvoc_mem_87808.references = NULL;
    
    struct memblock wval_mem_87807;
    
    wval_mem_87807.references = NULL;
    
    struct memblock wup_mem_87806;
    
    wup_mem_87806.references = NULL;
    
    struct memblock wte_mem_87805;
    
    wte_mem_87805.references = NULL;
    
    struct memblock wqry_mem_87804;
    
    wqry_mem_87804.references = NULL;
    
    struct memblock wpe_mem_87803;
    
    wpe_mem_87803.references = NULL;
    
    struct memblock wout_mem_87802;
    
    wout_mem_87802.references = NULL;
    
    struct memblock wkey_mem_87801;
    
    wkey_mem_87801.references = NULL;
    
    struct memblock wdown_mem_87800;
    
    wdown_mem_87800.references = NULL;
    wdown_mem_87800 = in0->v0->mem;
    wkey_mem_87801 = in0->v1->mem;
    wout_mem_87802 = in0->v2->mem;
    wpe_mem_87803 = in0->v3->mem;
    wqry_mem_87804 = in0->v4->mem;
    wte_mem_87805 = in0->v5->mem;
    wup_mem_87806 = in0->v6->mem;
    wval_mem_87807 = in0->v7->mem;
    wvoc_mem_87808 = in0->v8->mem;
    tokens_mem_87809 = in1->mem;
    mask_mem_87810 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_89540, wdown_mem_87800, wkey_mem_87801, wout_mem_87802, wpe_mem_87803, wqry_mem_87804, wte_mem_87805, wup_mem_87806, wval_mem_87807, wvoc_mem_87808, tokens_mem_87809, mask_mem_87810);
        if (ret == 0) {
            struct memblock mem_87791 = ctx->constants->mem_87791;
            struct memblock mem_87792 = ctx->constants->mem_87792;
            struct memblock mem_87793 = ctx->constants->mem_87793;
            struct memblock mem_87794 = ctx->constants->mem_87794;
            struct memblock mem_87795 = ctx->constants->mem_87795;
            struct memblock mem_87796 = ctx->constants->mem_87796;
            struct memblock mem_87797 = ctx->constants->mem_87797;
            struct memblock mem_87798 = ctx->constants->mem_87798;
            struct memblock mem_87799 = ctx->constants->mem_87799;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_89540;
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
    
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock wvoc_mem_87808;
    
    wvoc_mem_87808.references = NULL;
    
    struct memblock wdown_mem_87807;
    
    wdown_mem_87807.references = NULL;
    
    struct memblock wup_mem_87806;
    
    wup_mem_87806.references = NULL;
    
    struct memblock wout_mem_87805;
    
    wout_mem_87805.references = NULL;
    
    struct memblock wval_mem_87804;
    
    wval_mem_87804.references = NULL;
    
    struct memblock wkey_mem_87803;
    
    wkey_mem_87803.references = NULL;
    
    struct memblock wqry_mem_87802;
    
    wqry_mem_87802.references = NULL;
    
    struct memblock wpe_mem_87801;
    
    wpe_mem_87801.references = NULL;
    
    struct memblock wte_mem_87800;
    
    wte_mem_87800.references = NULL;
    wte_mem_87800 = in0->mem;
    wpe_mem_87801 = in1->mem;
    wqry_mem_87802 = in2->mem;
    wkey_mem_87803 = in3->mem;
    wval_mem_87804 = in4->mem;
    wout_mem_87805 = in5->mem;
    wup_mem_87806 = in6->mem;
    wdown_mem_87807 = in7->mem;
    wvoc_mem_87808 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_89540, &mem_out_89541, &mem_out_89542, &mem_out_89543, &mem_out_89544, &mem_out_89545, &mem_out_89546, &mem_out_89547, &mem_out_89548, wte_mem_87800, wpe_mem_87801, wqry_mem_87802, wkey_mem_87803, wval_mem_87804, wout_mem_87805, wup_mem_87806, wdown_mem_87807, wvoc_mem_87808);
        if (ret == 0) {
            struct memblock mem_87791 = ctx->constants->mem_87791;
            struct memblock mem_87792 = ctx->constants->mem_87792;
            struct memblock mem_87793 = ctx->constants->mem_87793;
            struct memblock mem_87794 = ctx->constants->mem_87794;
            struct memblock mem_87795 = ctx->constants->mem_87795;
            struct memblock mem_87796 = ctx->constants->mem_87796;
            struct memblock mem_87797 = ctx->constants->mem_87797;
            struct memblock mem_87798 = ctx->constants->mem_87798;
            struct memblock mem_87799 = ctx->constants->mem_87799;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_89540;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_89541;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_89542;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_89543;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_89544;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_89545;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_89546;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_89547;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_89548;
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
    
    struct memblock mem_out_89566;
    
    mem_out_89566.references = NULL;
    
    struct memblock mem_out_89565;
    
    mem_out_89565.references = NULL;
    
    struct memblock mem_out_89564;
    
    mem_out_89564.references = NULL;
    
    struct memblock mem_out_89563;
    
    mem_out_89563.references = NULL;
    
    struct memblock mem_out_89562;
    
    mem_out_89562.references = NULL;
    
    struct memblock mem_out_89561;
    
    mem_out_89561.references = NULL;
    
    struct memblock mem_out_89560;
    
    mem_out_89560.references = NULL;
    
    struct memblock mem_out_89559;
    
    mem_out_89559.references = NULL;
    
    struct memblock mem_out_89558;
    
    mem_out_89558.references = NULL;
    
    struct memblock mem_out_89557;
    
    mem_out_89557.references = NULL;
    
    struct memblock mem_out_89556;
    
    mem_out_89556.references = NULL;
    
    struct memblock mem_out_89555;
    
    mem_out_89555.references = NULL;
    
    struct memblock mem_out_89554;
    
    mem_out_89554.references = NULL;
    
    struct memblock mem_out_89553;
    
    mem_out_89553.references = NULL;
    
    struct memblock mem_out_89552;
    
    mem_out_89552.references = NULL;
    
    struct memblock mem_out_89551;
    
    mem_out_89551.references = NULL;
    
    struct memblock mem_out_89550;
    
    mem_out_89550.references = NULL;
    
    struct memblock mem_out_89549;
    
    mem_out_89549.references = NULL;
    
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    
    struct memblock seqs_mem_87829;
    
    seqs_mem_87829.references = NULL;
    
    struct memblock dls_mem_87828;
    
    dls_mem_87828.references = NULL;
    
    struct memblock masks_mem_87827;
    
    masks_mem_87827.references = NULL;
    
    struct memblock wvoc_mem_87826;
    
    wvoc_mem_87826.references = NULL;
    
    struct memblock wval_mem_87825;
    
    wval_mem_87825.references = NULL;
    
    struct memblock wup_mem_87824;
    
    wup_mem_87824.references = NULL;
    
    struct memblock wte_mem_87823;
    
    wte_mem_87823.references = NULL;
    
    struct memblock wqry_mem_87822;
    
    wqry_mem_87822.references = NULL;
    
    struct memblock wpe_mem_87821;
    
    wpe_mem_87821.references = NULL;
    
    struct memblock wout_mem_87820;
    
    wout_mem_87820.references = NULL;
    
    struct memblock wkey_mem_87819;
    
    wkey_mem_87819.references = NULL;
    
    struct memblock wdown_mem_87818;
    
    wdown_mem_87818.references = NULL;
    
    struct memblock wvoc_mem_87817;
    
    wvoc_mem_87817.references = NULL;
    
    struct memblock wval_mem_87816;
    
    wval_mem_87816.references = NULL;
    
    struct memblock wup_mem_87815;
    
    wup_mem_87815.references = NULL;
    
    struct memblock wte_mem_87814;
    
    wte_mem_87814.references = NULL;
    
    struct memblock wqry_mem_87813;
    
    wqry_mem_87813.references = NULL;
    
    struct memblock wpe_mem_87812;
    
    wpe_mem_87812.references = NULL;
    
    struct memblock wout_mem_87811;
    
    wout_mem_87811.references = NULL;
    
    struct memblock wkey_mem_87810;
    
    wkey_mem_87810.references = NULL;
    
    struct memblock wdown_mem_87809;
    
    wdown_mem_87809.references = NULL;
    
    struct memblock wvoc_mem_87808;
    
    wvoc_mem_87808.references = NULL;
    
    struct memblock wval_mem_87807;
    
    wval_mem_87807.references = NULL;
    
    struct memblock wup_mem_87806;
    
    wup_mem_87806.references = NULL;
    
    struct memblock wte_mem_87805;
    
    wte_mem_87805.references = NULL;
    
    struct memblock wqry_mem_87804;
    
    wqry_mem_87804.references = NULL;
    
    struct memblock wpe_mem_87803;
    
    wpe_mem_87803.references = NULL;
    
    struct memblock wout_mem_87802;
    
    wout_mem_87802.references = NULL;
    
    struct memblock wkey_mem_87801;
    
    wkey_mem_87801.references = NULL;
    
    struct memblock wdown_mem_87800;
    
    wdown_mem_87800.references = NULL;
    wdown_mem_87800 = in0->v0->mem;
    wkey_mem_87801 = in0->v1->mem;
    wout_mem_87802 = in0->v2->mem;
    wpe_mem_87803 = in0->v3->mem;
    wqry_mem_87804 = in0->v4->mem;
    wte_mem_87805 = in0->v5->mem;
    wup_mem_87806 = in0->v6->mem;
    wval_mem_87807 = in0->v7->mem;
    wvoc_mem_87808 = in0->v8->mem;
    wdown_mem_87809 = in1->v0->mem;
    wkey_mem_87810 = in1->v1->mem;
    wout_mem_87811 = in1->v2->mem;
    wpe_mem_87812 = in1->v3->mem;
    wqry_mem_87813 = in1->v4->mem;
    wte_mem_87814 = in1->v5->mem;
    wup_mem_87815 = in1->v6->mem;
    wval_mem_87816 = in1->v7->mem;
    wvoc_mem_87817 = in1->v8->mem;
    wdown_mem_87818 = in2->v0->mem;
    wkey_mem_87819 = in2->v1->mem;
    wout_mem_87820 = in2->v2->mem;
    wpe_mem_87821 = in2->v3->mem;
    wqry_mem_87822 = in2->v4->mem;
    wte_mem_87823 = in2->v5->mem;
    wup_mem_87824 = in2->v6->mem;
    wval_mem_87825 = in2->v7->mem;
    wvoc_mem_87826 = in2->v8->mem;
    masks_mem_87827 = in3->mem;
    dls_mem_87828 = in4->mem;
    seqs_mem_87829 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_89540, &mem_out_89541, &mem_out_89542, &mem_out_89543, &mem_out_89544, &mem_out_89545, &mem_out_89546, &mem_out_89547, &mem_out_89548, &mem_out_89549, &mem_out_89550, &mem_out_89551, &mem_out_89552, &mem_out_89553, &mem_out_89554, &mem_out_89555, &mem_out_89556, &mem_out_89557, &mem_out_89558, &mem_out_89559, &mem_out_89560, &mem_out_89561, &mem_out_89562, &mem_out_89563, &mem_out_89564, &mem_out_89565, &mem_out_89566, wdown_mem_87800, wkey_mem_87801, wout_mem_87802, wpe_mem_87803, wqry_mem_87804, wte_mem_87805, wup_mem_87806, wval_mem_87807, wvoc_mem_87808, wdown_mem_87809, wkey_mem_87810, wout_mem_87811, wpe_mem_87812, wqry_mem_87813, wte_mem_87814, wup_mem_87815, wval_mem_87816, wvoc_mem_87817, wdown_mem_87818, wkey_mem_87819, wout_mem_87820, wpe_mem_87821, wqry_mem_87822, wte_mem_87823, wup_mem_87824, wval_mem_87825, wvoc_mem_87826, masks_mem_87827, dls_mem_87828, seqs_mem_87829);
        if (ret == 0) {
            struct memblock mem_87791 = ctx->constants->mem_87791;
            struct memblock mem_87792 = ctx->constants->mem_87792;
            struct memblock mem_87793 = ctx->constants->mem_87793;
            struct memblock mem_87794 = ctx->constants->mem_87794;
            struct memblock mem_87795 = ctx->constants->mem_87795;
            struct memblock mem_87796 = ctx->constants->mem_87796;
            struct memblock mem_87797 = ctx->constants->mem_87797;
            struct memblock mem_87798 = ctx->constants->mem_87798;
            struct memblock mem_87799 = ctx->constants->mem_87799;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_89540;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_89541;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_89542;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_89543;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_89544;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_89545;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_89546;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_89547;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_89548;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_89549;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_89550;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_89551;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_89552;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_89553;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_89554;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_89555;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_89556;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_89557;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_89558;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_89559;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_89560;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_89561;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_89562;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_89563;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_89564;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_89565;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_89566;
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
    
    struct memblock mem_out_89548;
    
    mem_out_89548.references = NULL;
    
    struct memblock mem_out_89547;
    
    mem_out_89547.references = NULL;
    
    struct memblock mem_out_89546;
    
    mem_out_89546.references = NULL;
    
    struct memblock mem_out_89545;
    
    mem_out_89545.references = NULL;
    
    struct memblock mem_out_89544;
    
    mem_out_89544.references = NULL;
    
    struct memblock mem_out_89543;
    
    mem_out_89543.references = NULL;
    
    struct memblock mem_out_89542;
    
    mem_out_89542.references = NULL;
    
    struct memblock mem_out_89541;
    
    mem_out_89541.references = NULL;
    
    struct memblock mem_out_89540;
    
    mem_out_89540.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_89540, &mem_out_89541, &mem_out_89542, &mem_out_89543, &mem_out_89544, &mem_out_89545, &mem_out_89546, &mem_out_89547, &mem_out_89548);
        if (ret == 0) {
            struct memblock mem_87791 = ctx->constants->mem_87791;
            struct memblock mem_87792 = ctx->constants->mem_87792;
            struct memblock mem_87793 = ctx->constants->mem_87793;
            struct memblock mem_87794 = ctx->constants->mem_87794;
            struct memblock mem_87795 = ctx->constants->mem_87795;
            struct memblock mem_87796 = ctx->constants->mem_87796;
            struct memblock mem_87797 = ctx->constants->mem_87797;
            struct memblock mem_87798 = ctx->constants->mem_87798;
            struct memblock mem_87799 = ctx->constants->mem_87799;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_89540;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_89541;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_89542;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_89543;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_89544;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_89545;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_89546;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_89547;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_89548;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
