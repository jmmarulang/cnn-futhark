
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
    struct memblock mem_96098;
    struct memblock mem_96099;
    struct memblock mem_96100;
    struct memblock mem_96101;
    struct memblock mem_96102;
    struct memblock mem_96103;
    struct memblock mem_96104;
    struct memblock mem_96105;
    struct memblock mem_96106;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10524(struct futhark_context *ctx, struct memblock *mem_out_p_98317, struct memblock *mem_out_p_98318, struct memblock *mem_out_p_98319, struct memblock w_mem_96107, struct memblock mw_mem_96108, struct memblock vw_mem_96109, struct memblock dw_mem_96110, int64_t n_68258, int64_t m_68259, int64_t step_68264, double lt_r_68265);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10525(struct futhark_context *ctx, struct memblock *mem_out_p_98322, struct memblock *mem_out_p_98323, struct memblock *mem_out_p_98324, struct memblock w_mem_96107, struct memblock mw_mem_96108, struct memblock vw_mem_96109, struct memblock dw_mem_96110, int64_t n_69291, int64_t m_69292, int64_t step_69297, double lt_r_69298);
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_98327, struct memblock wdown_mem_96107, struct memblock wkey_mem_96108, struct memblock wout_mem_96109, struct memblock wpe_mem_96110, struct memblock wqry_mem_96111, struct memblock wte_mem_96112, struct memblock wup_mem_96113, struct memblock wval_mem_96114, struct memblock wvoc_mem_96115, struct memblock tokens_mem_96116, struct memblock mask_mem_96117);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_98381, struct memblock *mem_out_p_98382, struct memblock *mem_out_p_98383, struct memblock *mem_out_p_98384, struct memblock *mem_out_p_98385, struct memblock *mem_out_p_98386, struct memblock *mem_out_p_98387, struct memblock *mem_out_p_98388, struct memblock *mem_out_p_98389, struct memblock wte_mem_96107, struct memblock wpe_mem_96108, struct memblock wqry_mem_96109, struct memblock wkey_mem_96110, struct memblock wval_mem_96111, struct memblock wout_mem_96112, struct memblock wup_mem_96113, struct memblock wdown_mem_96114, struct memblock wvoc_mem_96115);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_98390, struct memblock *mem_out_p_98391, struct memblock *mem_out_p_98392, struct memblock *mem_out_p_98393, struct memblock *mem_out_p_98394, struct memblock *mem_out_p_98395, struct memblock *mem_out_p_98396, struct memblock *mem_out_p_98397, struct memblock *mem_out_p_98398, struct memblock *mem_out_p_98399, struct memblock *mem_out_p_98400, struct memblock *mem_out_p_98401, struct memblock *mem_out_p_98402, struct memblock *mem_out_p_98403, struct memblock *mem_out_p_98404, struct memblock *mem_out_p_98405, struct memblock *mem_out_p_98406, struct memblock *mem_out_p_98407, struct memblock *mem_out_p_98408, struct memblock *mem_out_p_98409, struct memblock *mem_out_p_98410, struct memblock *mem_out_p_98411, struct memblock *mem_out_p_98412, struct memblock *mem_out_p_98413, struct memblock *mem_out_p_98414, struct memblock *mem_out_p_98415, struct memblock *mem_out_p_98416, struct memblock wdown_mem_96107, struct memblock wkey_mem_96108, struct memblock wout_mem_96109, struct memblock wpe_mem_96110, struct memblock wqry_mem_96111, struct memblock wte_mem_96112, struct memblock wup_mem_96113, struct memblock wval_mem_96114, struct memblock wvoc_mem_96115, struct memblock wdown_mem_96116, struct memblock wkey_mem_96117, struct memblock wout_mem_96118, struct memblock wpe_mem_96119, struct memblock wqry_mem_96120, struct memblock wte_mem_96121, struct memblock wup_mem_96122, struct memblock wval_mem_96123, struct memblock wvoc_mem_96124, struct memblock wdown_mem_96125, struct memblock wkey_mem_96126, struct memblock wout_mem_96127, struct memblock wpe_mem_96128, struct memblock wqry_mem_96129, struct memblock wte_mem_96130, struct memblock wup_mem_96131, struct memblock wval_mem_96132, struct memblock wvoc_mem_96133, struct memblock masks_mem_96134, struct memblock dls_mem_96135, struct memblock seqs_mem_96136);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_98594, struct memblock *mem_out_p_98595, struct memblock *mem_out_p_98596, struct memblock *mem_out_p_98597, struct memblock *mem_out_p_98598, struct memblock *mem_out_p_98599, struct memblock *mem_out_p_98600, struct memblock *mem_out_p_98601, struct memblock *mem_out_p_98602);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_96098 (ctx->constants->mem_96098)
    #define mem_96099 (ctx->constants->mem_96099)
    #define mem_96100 (ctx->constants->mem_96100)
    #define mem_96101 (ctx->constants->mem_96101)
    #define mem_96102 (ctx->constants->mem_96102)
    #define mem_96103 (ctx->constants->mem_96103)
    #define mem_96104 (ctx->constants->mem_96104)
    #define mem_96105 (ctx->constants->mem_96105)
    #define mem_96106 (ctx->constants->mem_96106)
    mem_96098.references = NULL;
    mem_96099.references = NULL;
    mem_96100.references = NULL;
    mem_96101.references = NULL;
    mem_96102.references = NULL;
    mem_96103.references = NULL;
    mem_96104.references = NULL;
    mem_96105.references = NULL;
    mem_96106.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96098, (int64_t) 3456, "mem_96098")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98299 = 0; nest_i_98299 < (int64_t) 27; nest_i_98299++) {
        for (int64_t nest_i_98300 = 0; nest_i_98300 < (int64_t) 16; nest_i_98300++) {
            ((double *) mem_96098.mem)[nest_i_98299 * (int64_t) 16 + nest_i_98300] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96099, (int64_t) 2048, "mem_96099")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98301 = 0; nest_i_98301 < (int64_t) 16; nest_i_98301++) {
        for (int64_t nest_i_98302 = 0; nest_i_98302 < (int64_t) 16; nest_i_98302++) {
            ((double *) mem_96099.mem)[nest_i_98301 * (int64_t) 16 + nest_i_98302] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96100, (int64_t) 2048, "mem_96100")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98303 = 0; nest_i_98303 < (int64_t) 16; nest_i_98303++) {
        for (int64_t nest_i_98304 = 0; nest_i_98304 < (int64_t) 16; nest_i_98304++) {
            ((double *) mem_96100.mem)[nest_i_98303 * (int64_t) 16 + nest_i_98304] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96101, (int64_t) 2048, "mem_96101")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98305 = 0; nest_i_98305 < (int64_t) 16; nest_i_98305++) {
        for (int64_t nest_i_98306 = 0; nest_i_98306 < (int64_t) 16; nest_i_98306++) {
            ((double *) mem_96101.mem)[nest_i_98305 * (int64_t) 16 + nest_i_98306] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96102, (int64_t) 2048, "mem_96102")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98307 = 0; nest_i_98307 < (int64_t) 16; nest_i_98307++) {
        for (int64_t nest_i_98308 = 0; nest_i_98308 < (int64_t) 16; nest_i_98308++) {
            ((double *) mem_96102.mem)[nest_i_98307 * (int64_t) 16 + nest_i_98308] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96103, (int64_t) 2048, "mem_96103")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98309 = 0; nest_i_98309 < (int64_t) 16; nest_i_98309++) {
        for (int64_t nest_i_98310 = 0; nest_i_98310 < (int64_t) 16; nest_i_98310++) {
            ((double *) mem_96103.mem)[nest_i_98309 * (int64_t) 16 + nest_i_98310] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96104, (int64_t) 8192, "mem_96104")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98311 = 0; nest_i_98311 < (int64_t) 64; nest_i_98311++) {
        for (int64_t nest_i_98312 = 0; nest_i_98312 < (int64_t) 16; nest_i_98312++) {
            ((double *) mem_96104.mem)[nest_i_98311 * (int64_t) 16 + nest_i_98312] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96105, (int64_t) 8192, "mem_96105")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98313 = 0; nest_i_98313 < (int64_t) 16; nest_i_98313++) {
        for (int64_t nest_i_98314 = 0; nest_i_98314 < (int64_t) 64; nest_i_98314++) {
            ((double *) mem_96105.mem)[nest_i_98313 * (int64_t) 64 + nest_i_98314] = 0.0;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96106, (int64_t) 3456, "mem_96106")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_98315 = 0; nest_i_98315 < (int64_t) 27; nest_i_98315++) {
        for (int64_t nest_i_98316 = 0; nest_i_98316 < (int64_t) 16; nest_i_98316++) {
            ((double *) mem_96106.mem)[nest_i_98315 * (int64_t) 16 + nest_i_98316] = 0.0;
        }
    }
    #undef mem_96098
    #undef mem_96099
    #undef mem_96100
    #undef mem_96101
    #undef mem_96102
    #undef mem_96103
    #undef mem_96104
    #undef mem_96105
    #undef mem_96106
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_96098, "ctx->constants->mem_96098") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96099, "ctx->constants->mem_96099") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96100, "ctx->constants->mem_96100") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96101, "ctx->constants->mem_96101") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96102, "ctx->constants->mem_96102") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96103, "ctx->constants->mem_96103") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96104, "ctx->constants->mem_96104") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96105, "ctx->constants->mem_96105") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_96106, "ctx->constants->mem_96106") != 0)
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_10524(struct futhark_context *ctx, struct memblock *mem_out_p_98317, struct memblock *mem_out_p_98318, struct memblock *mem_out_p_98319, struct memblock w_mem_96107, struct memblock mw_mem_96108, struct memblock vw_mem_96109, struct memblock dw_mem_96110, int64_t n_68258, int64_t m_68259, int64_t step_68264, double lt_r_68265)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_96151_cached_sizze_98320 = 0;
    unsigned char *mem_96151 = NULL;
    int64_t mem_96154_cached_sizze_98321 = 0;
    unsigned char *mem_96154 = NULL;
    struct memblock mem_96189;
    
    mem_96189.references = NULL;
    
    struct memblock mem_96116;
    
    mem_96116.references = NULL;
    
    struct memblock mem_96113;
    
    mem_96113.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_96111 = (int64_t) 8 * n_68258;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_96112 = m_68259 * binop_x_96111;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96113, bytes_96112, "mem_96113")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96116, bytes_96112, "mem_96116")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95181 = 0; i_95181 < n_68258; i_95181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95174 = 0; i_95174 < m_68259; i_95174++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90932 = ((double *) mw_mem_96108.mem)[i_95181 * m_68259 + i_95174];
            
            // futhark/microgpt.fut:412:10-20
            
            double zp_lhs_90933 = 0.85 * zt_rhs_90932;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90934 = ((double *) dw_mem_96110.mem)[i_95181 * m_68259 + i_95174];
            
            // futhark/microgpt.fut:412:35-45
            
            double zp_rhs_90935 = 0.15000000000000002 * zt_rhs_90934;
            
            // futhark/microgpt.fut:412:21-45
            
            double lifted_lambda_res_90936 = zp_lhs_90933 + zp_rhs_90935;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90943 = ((double *) vw_mem_96109.mem)[i_95181 * m_68259 + i_95174];
            
            // futhark/microgpt.fut:414:10-20
            
            double zp_lhs_90944 = 0.99 * zt_rhs_90943;
            
            // futhark/microgpt.fut:414:35-45
            
            double zt_lhs_90946 = 1.0000000000000009e-2 * zt_rhs_90934;
            
            // futhark/microgpt.fut:414:46-56
            
            double zp_rhs_90947 = zt_rhs_90934 * zt_lhs_90946;
            
            // futhark/microgpt.fut:414:21-56
            
            double lifted_lambda_res_90948 = zp_lhs_90944 + zp_rhs_90947;
            
            ((double *) mem_96113.mem)[i_95181 * m_68259 + i_95174] = lifted_lambda_res_90948;
            ((double *) mem_96116.mem)[i_95181 * m_68259 + i_95174] = lifted_lambda_res_90936;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_73326 = sitofp_i64_f64(step_68264);
    
    // futhark/microgpt.fut:416:54-57
    
    double ztzt_rhs_73327 = 1.0 + i64_res_73326;
    
    // futhark/microgpt.fut:416:30-57
    
    double zm_rhs_73328 = fpow64(0.85, ztzt_rhs_73327);
    
    // futhark/microgpt.fut:416:23-57
    
    double zs_rhs_73329 = 1.0 - zm_rhs_73328;
    
    // futhark/microgpt.fut:418:31-58
    
    double zm_rhs_73367 = fpow64(0.99, ztzt_rhs_73327);
    
    // futhark/microgpt.fut:418:23-58
    
    double zs_rhs_73368 = 1.0 - zm_rhs_73367;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_96151_cached_sizze_98320 < bytes_96112) {
        err = lexical_realloc(ctx, &mem_96151, &mem_96151_cached_sizze_98320, bytes_96112);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96154_cached_sizze_98321 < bytes_96112) {
        err = lexical_realloc(ctx, &mem_96154, &mem_96154_cached_sizze_98321, bytes_96112);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95195 = 0; i_95195 < n_68258; i_95195++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95188 = 0; i_95188 < m_68259; i_95188++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_90968 = ((double *) mem_96116.mem)[i_95195 * m_68259 + i_95188];
            
            // futhark/microgpt.fut:416:18-57
            
            double lifted_lambda_res_90969 = zs_lhs_90968 / zs_rhs_73329;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_90976 = ((double *) mem_96113.mem)[i_95195 * m_68259 + i_95188];
            
            // futhark/microgpt.fut:418:18-58
            
            double lifted_lambda_res_90977 = zs_lhs_90976 / zs_rhs_73368;
            
            ((double *) mem_96151)[i_95195 * m_68259 + i_95188] = lifted_lambda_res_90977;
            ((double *) mem_96154)[i_95195 * m_68259 + i_95188] = lifted_lambda_res_90969;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96189, bytes_96112, "mem_96189")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95204 = 0; i_95204 < n_68258; i_95204++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95200 = 0; i_95200 < m_68259; i_95200++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_72490 = ((double *) w_mem_96107.mem)[i_95204 * m_68259 + i_95200];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_72491 = ((double *) mem_96154)[i_95204 * m_68259 + i_95200];
            
            // futhark/microgpt.fut:420:21-34
            
            double zs_lhs_72492 = lt_r_68265 * zt_rhs_72491;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_72493 = ((double *) mem_96151)[i_95204 * m_68259 + i_95200];
            
            // futhark/microgpt.fut:420:51-57
            
            double zp_lhs_72494 = fpow64(ztzt_lhs_72493, 0.5);
            
            // futhark/microgpt.fut:420:59-71
            
            double zs_rhs_72495 = 1.0e-8 + zp_lhs_72494;
            
            // futhark/microgpt.fut:420:35-71
            
            double zm_rhs_72496 = zs_lhs_72492 / zs_rhs_72495;
            
            // futhark/microgpt.fut:420:13-71
            
            double lifted_lambda_res_72497 = zm_lhs_72490 - zm_rhs_72496;
            
            ((double *) mem_96189.mem)[i_95204 * m_68259 + i_95200] = lifted_lambda_res_72497;
        }
    }
    if (memblock_set(ctx, &mem_out_97974, &mem_96189, "mem_96189") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97975, &mem_96116, "mem_96116") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97976, &mem_96113, "mem_96113") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98317, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98318, &mem_out_97975, "mem_out_97975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98319, &mem_out_97976, "mem_out_97976") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_96151);
        free(mem_96154);
        if (memblock_unref(ctx, &mem_96189, "mem_96189") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_96116, "mem_96116") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_96113, "mem_96113") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97976, "mem_out_97976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97975, "mem_out_97975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_10525(struct futhark_context *ctx, struct memblock *mem_out_p_98322, struct memblock *mem_out_p_98323, struct memblock *mem_out_p_98324, struct memblock w_mem_96107, struct memblock mw_mem_96108, struct memblock vw_mem_96109, struct memblock dw_mem_96110, int64_t n_69291, int64_t m_69292, int64_t step_69297, double lt_r_69298)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_96151_cached_sizze_98325 = 0;
    unsigned char *mem_96151 = NULL;
    int64_t mem_96154_cached_sizze_98326 = 0;
    unsigned char *mem_96154 = NULL;
    struct memblock mem_96189;
    
    mem_96189.references = NULL;
    
    struct memblock mem_96116;
    
    mem_96116.references = NULL;
    
    struct memblock mem_96113;
    
    mem_96113.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_96111 = (int64_t) 8 * n_69291;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_96112 = m_69292 * binop_x_96111;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96113, bytes_96112, "mem_96113")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96116, bytes_96112, "mem_96116")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95181 = 0; i_95181 < n_69291; i_95181++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95174 = 0; i_95174 < m_69292; i_95174++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90932 = ((double *) mw_mem_96108.mem)[i_95181 * m_69292 + i_95174];
            
            // futhark/microgpt.fut:412:10-20
            
            double zp_lhs_90933 = 0.85 * zt_rhs_90932;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90934 = ((double *) dw_mem_96110.mem)[i_95181 * m_69292 + i_95174];
            
            // futhark/microgpt.fut:412:35-45
            
            double zp_rhs_90935 = 0.15000000000000002 * zt_rhs_90934;
            
            // futhark/microgpt.fut:412:21-45
            
            double lifted_lambda_res_90936 = zp_lhs_90933 + zp_rhs_90935;
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_90943 = ((double *) vw_mem_96109.mem)[i_95181 * m_69292 + i_95174];
            
            // futhark/microgpt.fut:414:10-20
            
            double zp_lhs_90944 = 0.99 * zt_rhs_90943;
            
            // futhark/microgpt.fut:414:35-45
            
            double zt_lhs_90946 = 1.0000000000000009e-2 * zt_rhs_90934;
            
            // futhark/microgpt.fut:414:46-56
            
            double zp_rhs_90947 = zt_rhs_90934 * zt_lhs_90946;
            
            // futhark/microgpt.fut:414:21-56
            
            double lifted_lambda_res_90948 = zp_lhs_90944 + zp_rhs_90947;
            
            ((double *) mem_96113.mem)[i_95181 * m_69292 + i_95174] = lifted_lambda_res_90948;
            ((double *) mem_96116.mem)[i_95181 * m_69292 + i_95174] = lifted_lambda_res_90936;
        }
    }
    // futhark/microgpt.fut:66:26-45
    
    double i64_res_73326 = sitofp_i64_f64(step_69297);
    
    // futhark/microgpt.fut:416:54-57
    
    double ztzt_rhs_73327 = 1.0 + i64_res_73326;
    
    // futhark/microgpt.fut:416:30-57
    
    double zm_rhs_73328 = fpow64(0.85, ztzt_rhs_73327);
    
    // futhark/microgpt.fut:416:23-57
    
    double zs_rhs_73329 = 1.0 - zm_rhs_73328;
    
    // futhark/microgpt.fut:418:31-58
    
    double zm_rhs_73367 = fpow64(0.99, ztzt_rhs_73327);
    
    // futhark/microgpt.fut:418:23-58
    
    double zs_rhs_73368 = 1.0 - zm_rhs_73367;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_96151_cached_sizze_98325 < bytes_96112) {
        err = lexical_realloc(ctx, &mem_96151, &mem_96151_cached_sizze_98325, bytes_96112);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96154_cached_sizze_98326 < bytes_96112) {
        err = lexical_realloc(ctx, &mem_96154, &mem_96154_cached_sizze_98326, bytes_96112);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95195 = 0; i_95195 < n_69291; i_95195++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95188 = 0; i_95188 < m_69292; i_95188++) {
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_90968 = ((double *) mem_96116.mem)[i_95195 * m_69292 + i_95188];
            
            // futhark/microgpt.fut:416:18-57
            
            double lifted_lambda_res_90969 = zs_lhs_90968 / zs_rhs_73329;
            
            // futhark/microgpt.fut:4:11-25
            
            double zs_lhs_90976 = ((double *) mem_96113.mem)[i_95195 * m_69292 + i_95188];
            
            // futhark/microgpt.fut:418:18-58
            
            double lifted_lambda_res_90977 = zs_lhs_90976 / zs_rhs_73368;
            
            ((double *) mem_96151)[i_95195 * m_69292 + i_95188] = lifted_lambda_res_90977;
            ((double *) mem_96154)[i_95195 * m_69292 + i_95188] = lifted_lambda_res_90969;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96189, bytes_96112, "mem_96189")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95204 = 0; i_95204 < n_69291; i_95204++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95200 = 0; i_95200 < m_69292; i_95200++) {
            // futhark/microgpt.fut:4:11-25
            
            double zm_lhs_72490 = ((double *) w_mem_96107.mem)[i_95204 * m_69292 + i_95200];
            
            // futhark/microgpt.fut:4:11-25
            
            double zt_rhs_72491 = ((double *) mem_96154)[i_95204 * m_69292 + i_95200];
            
            // futhark/microgpt.fut:420:21-34
            
            double zs_lhs_72492 = lt_r_69298 * zt_rhs_72491;
            
            // futhark/microgpt.fut:4:11-25
            
            double ztzt_lhs_72493 = ((double *) mem_96151)[i_95204 * m_69292 + i_95200];
            
            // futhark/microgpt.fut:420:51-57
            
            double zp_lhs_72494 = fpow64(ztzt_lhs_72493, 0.5);
            
            // futhark/microgpt.fut:420:59-71
            
            double zs_rhs_72495 = 1.0e-8 + zp_lhs_72494;
            
            // futhark/microgpt.fut:420:35-71
            
            double zm_rhs_72496 = zs_lhs_72492 / zs_rhs_72495;
            
            // futhark/microgpt.fut:420:13-71
            
            double lifted_lambda_res_72497 = zm_lhs_72490 - zm_rhs_72496;
            
            ((double *) mem_96189.mem)[i_95204 * m_69292 + i_95200] = lifted_lambda_res_72497;
        }
    }
    if (memblock_set(ctx, &mem_out_97974, &mem_96189, "mem_96189") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97975, &mem_96116, "mem_96116") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97976, &mem_96113, "mem_96113") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98322, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98323, &mem_out_97975, "mem_out_97975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98324, &mem_out_97976, "mem_out_97976") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_96151);
        free(mem_96154);
        if (memblock_unref(ctx, &mem_96189, "mem_96189") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_96116, "mem_96116") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_96113, "mem_96113") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97976, "mem_out_97976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97975, "mem_out_97975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward_seq(struct futhark_context *ctx, struct memblock *mem_out_p_98327, struct memblock wdown_mem_96107, struct memblock wkey_mem_96108, struct memblock wout_mem_96109, struct memblock wpe_mem_96110, struct memblock wqry_mem_96111, struct memblock wte_mem_96112, struct memblock wup_mem_96113, struct memblock wval_mem_96114, struct memblock wvoc_mem_96115, struct memblock tokens_mem_96116, struct memblock mask_mem_96117)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_96118_cached_sizze_98328 = 0;
    unsigned char *mem_96118 = NULL;
    int64_t mem_96123_cached_sizze_98329 = 0;
    unsigned char *mem_96123 = NULL;
    int64_t mem_96134_cached_sizze_98330 = 0;
    unsigned char *mem_96134 = NULL;
    int64_t mem_96139_cached_sizze_98331 = 0;
    unsigned char *mem_96139 = NULL;
    int64_t mem_96150_cached_sizze_98332 = 0;
    unsigned char *mem_96150 = NULL;
    int64_t mem_96155_cached_sizze_98333 = 0;
    unsigned char *mem_96155 = NULL;
    int64_t mem_96162_cached_sizze_98334 = 0;
    unsigned char *mem_96162 = NULL;
    int64_t mem_96173_cached_sizze_98335 = 0;
    unsigned char *mem_96173 = NULL;
    int64_t mem_96178_cached_sizze_98336 = 0;
    unsigned char *mem_96178 = NULL;
    int64_t mem_96185_cached_sizze_98337 = 0;
    unsigned char *mem_96185 = NULL;
    int64_t mem_96196_cached_sizze_98338 = 0;
    unsigned char *mem_96196 = NULL;
    int64_t mem_96197_cached_sizze_98339 = 0;
    unsigned char *mem_96197 = NULL;
    int64_t mem_96198_cached_sizze_98340 = 0;
    unsigned char *mem_96198 = NULL;
    int64_t mem_96211_cached_sizze_98341 = 0;
    unsigned char *mem_96211 = NULL;
    int64_t mem_96212_cached_sizze_98342 = 0;
    unsigned char *mem_96212 = NULL;
    int64_t mem_96213_cached_sizze_98343 = 0;
    unsigned char *mem_96213 = NULL;
    int64_t mem_96244_cached_sizze_98344 = 0;
    unsigned char *mem_96244 = NULL;
    int64_t mem_96245_cached_sizze_98345 = 0;
    unsigned char *mem_96245 = NULL;
    int64_t mem_96246_cached_sizze_98346 = 0;
    unsigned char *mem_96246 = NULL;
    int64_t mem_96262_cached_sizze_98347 = 0;
    unsigned char *mem_96262 = NULL;
    int64_t mem_96263_cached_sizze_98348 = 0;
    unsigned char *mem_96263 = NULL;
    int64_t mem_96264_cached_sizze_98349 = 0;
    unsigned char *mem_96264 = NULL;
    int64_t mem_96277_cached_sizze_98350 = 0;
    unsigned char *mem_96277 = NULL;
    int64_t mem_96278_cached_sizze_98351 = 0;
    unsigned char *mem_96278 = NULL;
    int64_t mem_96279_cached_sizze_98352 = 0;
    unsigned char *mem_96279 = NULL;
    int64_t mem_96325_cached_sizze_98353 = 0;
    unsigned char *mem_96325 = NULL;
    int64_t mem_96331_cached_sizze_98354 = 0;
    unsigned char *mem_96331 = NULL;
    int64_t mem_96336_cached_sizze_98355 = 0;
    unsigned char *mem_96336 = NULL;
    int64_t mem_96347_cached_sizze_98356 = 0;
    unsigned char *mem_96347 = NULL;
    int64_t mem_96352_cached_sizze_98357 = 0;
    unsigned char *mem_96352 = NULL;
    int64_t mem_96363_cached_sizze_98358 = 0;
    unsigned char *mem_96363 = NULL;
    int64_t mem_96368_cached_sizze_98359 = 0;
    unsigned char *mem_96368 = NULL;
    int64_t mem_96375_cached_sizze_98360 = 0;
    unsigned char *mem_96375 = NULL;
    int64_t mem_96386_cached_sizze_98361 = 0;
    unsigned char *mem_96386 = NULL;
    int64_t mem_96391_cached_sizze_98362 = 0;
    unsigned char *mem_96391 = NULL;
    int64_t mem_96407_cached_sizze_98363 = 0;
    unsigned char *mem_96407 = NULL;
    int64_t mem_96412_cached_sizze_98364 = 0;
    unsigned char *mem_96412 = NULL;
    int64_t mem_96423_cached_sizze_98365 = 0;
    unsigned char *mem_96423 = NULL;
    int64_t mem_96428_cached_sizze_98366 = 0;
    unsigned char *mem_96428 = NULL;
    int64_t mem_96439_cached_sizze_98367 = 0;
    unsigned char *mem_96439 = NULL;
    int64_t mem_96444_cached_sizze_98368 = 0;
    unsigned char *mem_96444 = NULL;
    int64_t mem_96455_cached_sizze_98369 = 0;
    unsigned char *mem_96455 = NULL;
    int64_t mem_96460_cached_sizze_98370 = 0;
    unsigned char *mem_96460 = NULL;
    int64_t mem_96467_cached_sizze_98371 = 0;
    unsigned char *mem_96467 = NULL;
    int64_t mem_96478_cached_sizze_98372 = 0;
    unsigned char *mem_96478 = NULL;
    int64_t mem_96483_cached_sizze_98373 = 0;
    unsigned char *mem_96483 = NULL;
    int64_t mem_96494_cached_sizze_98374 = 0;
    unsigned char *mem_96494 = NULL;
    int64_t mem_96499_cached_sizze_98375 = 0;
    unsigned char *mem_96499 = NULL;
    int64_t mem_96510_cached_sizze_98376 = 0;
    unsigned char *mem_96510 = NULL;
    int64_t mem_96515_cached_sizze_98377 = 0;
    unsigned char *mem_96515 = NULL;
    int64_t mem_96526_cached_sizze_98378 = 0;
    unsigned char *mem_96526 = NULL;
    int64_t mem_96531_cached_sizze_98379 = 0;
    unsigned char *mem_96531 = NULL;
    int64_t mem_96547_cached_sizze_98380 = 0;
    unsigned char *mem_96547 = NULL;
    struct memblock mem_96542;
    
    mem_96542.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_96118_cached_sizze_98328 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96118, &mem_96118_cached_sizze_98328, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96123_cached_sizze_98329 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96123, &mem_96123_cached_sizze_98329, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95176 = 0; i_95176 < (int64_t) 16; i_95176++) {
        // futhark/microgpt.fut:397:41-50
        
        int64_t tmp_84515 = ((int64_t *) tokens_mem_96116.mem)[i_95176];
        
        // futhark/microgpt.fut:397:37-51
        
        bool x_84516 = sle64((int64_t) 0, tmp_84515);
        
        // futhark/microgpt.fut:397:37-51
        
        bool y_84517 = slt64(tmp_84515, (int64_t) 27);
        
        // futhark/microgpt.fut:397:37-51
        
        bool bounds_check_84518 = x_84516 && y_84517;
        
        // futhark/microgpt.fut:397:37-51
        
        bool index_certs_84519;
        
        if (!bounds_check_84518) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84515, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:397:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:397:16-55\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95172 = 0; i_95172 < (int64_t) 16; i_95172++) {
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_84526 = ((double *) wte_mem_96112.mem)[tmp_84515 * (int64_t) 16 + i_95172];
            
            ((double *) mem_96123)[i_95172] = lifted_lambda_res_84526;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96118, i_95176 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96123, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96134_cached_sizze_98330 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96134, &mem_96134_cached_sizze_98330, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96139_cached_sizze_98331 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96139, &mem_96139_cached_sizze_98331, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95184 = 0; i_95184 < (int64_t) 16; i_95184++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95180 = 0; i_95180 < (int64_t) 16; i_95180++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_84558 = ((double *) wpe_mem_96110.mem)[i_95184 * (int64_t) 16 + i_95180];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_84559 = ((double *) mem_96118)[i_95184 * (int64_t) 16 + i_95180];
            
            // futhark/microgpt.fut:158:46-86
            
            double zp_res_84560 = zp_lhs_84558 + zp_rhs_84559;
            
            ((double *) mem_96139)[i_95180] = zp_res_84560;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96134, i_95184 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96139, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96150_cached_sizze_98332 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96150, &mem_96150_cached_sizze_98332, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96155_cached_sizze_98333 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96155, &mem_96155_cached_sizze_98333, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96162_cached_sizze_98334 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96162, &mem_96162_cached_sizze_98334, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95196 = 0; i_95196 < (int64_t) 16; i_95196++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95188 = 0; i_95188 < (int64_t) 16; i_95188++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84575 = ((double *) mem_96134)[i_95196 * (int64_t) 16 + i_95188];
            
            // futhark/microgpt.fut:159:77-114
            
            double zt_res_84576 = zt_lhs_84575 * zt_lhs_84575;
            
            ((double *) mem_96155)[i_95188] = zt_res_84576;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_84578;
        double r_84580 = 0.0;
        
        for (int64_t i_84579 = 0; i_84579 < (int64_t) 16; i_84579++) {
            // futhark/microgpt.fut:160:37-47
            
            double lifted_lambda_res_84581 = ((double *) mem_96155)[i_84579];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_84582 = r_84580 + lifted_lambda_res_84581;
            double r_tmp_97981 = zp_res_84582;
            
            r_84580 = r_tmp_97981;
        }
        defunc_0_lifted_lambda_res_84578 = r_84580;
        // futhark/microgpt.fut:160:17-64
        
        double zs_res_84583 = defunc_0_lifted_lambda_res_84578 / 16.0;
        
        // futhark/microgpt.fut:161:24-55
        
        double zp_res_84584 = 1.0e-5 + zs_res_84583;
        
        // futhark/microgpt.fut:161:16-55
        
        double sqrt_res_84585 = futrts_sqrt64(zp_res_84584);
        
        // futhark/microgpt.fut:162:27-38
        
        double zs_res_84586 = 1.0 / sqrt_res_84585;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95192 = 0; i_95192 < (int64_t) 16; i_95192++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84593 = ((double *) mem_96134)[i_95196 * (int64_t) 16 + i_95192];
            
            // futhark/microgpt.fut:162:5-38
            
            double zt_res_84594 = zs_res_84586 * zt_lhs_84593;
            
            ((double *) mem_96162)[i_95192] = zt_res_84594;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96150, i_95196 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96162, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96173_cached_sizze_98335 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96173, &mem_96173_cached_sizze_98335, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96178_cached_sizze_98336 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96178, &mem_96178_cached_sizze_98336, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96185_cached_sizze_98337 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96185, &mem_96185_cached_sizze_98337, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95208 = 0; i_95208 < (int64_t) 16; i_95208++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95200 = 0; i_95200 < (int64_t) 16; i_95200++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84609 = ((double *) mem_96150)[i_95208 * (int64_t) 16 + i_95200];
            
            // futhark/microgpt.fut:163:77-114
            
            double zt_res_84610 = zt_lhs_84609 * zt_lhs_84609;
            
            ((double *) mem_96178)[i_95200] = zt_res_84610;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_84612;
        double r_84614 = 0.0;
        
        for (int64_t i_84613 = 0; i_84613 < (int64_t) 16; i_84613++) {
            // futhark/microgpt.fut:164:37-47
            
            double lifted_lambda_res_84615 = ((double *) mem_96178)[i_84613];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_84616 = r_84614 + lifted_lambda_res_84615;
            double r_tmp_97985 = zp_res_84616;
            
            r_84614 = r_tmp_97985;
        }
        defunc_0_lifted_lambda_res_84612 = r_84614;
        // futhark/microgpt.fut:164:17-64
        
        double zs_res_84617 = defunc_0_lifted_lambda_res_84612 / 16.0;
        
        // futhark/microgpt.fut:165:24-55
        
        double zp_res_84618 = 1.0e-5 + zs_res_84617;
        
        // futhark/microgpt.fut:165:16-55
        
        double sqrt_res_84619 = futrts_sqrt64(zp_res_84618);
        
        // futhark/microgpt.fut:166:27-38
        
        double zs_res_84620 = 1.0 / sqrt_res_84619;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95204 = 0; i_95204 < (int64_t) 16; i_95204++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84627 = ((double *) mem_96150)[i_95208 * (int64_t) 16 + i_95204];
            
            // futhark/microgpt.fut:166:5-38
            
            double zt_res_84628 = zs_res_84620 * zt_lhs_84627;
            
            ((double *) mem_96185)[i_95204] = zt_res_84628;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96173, i_95208 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96185, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96196_cached_sizze_98338 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96196, &mem_96196_cached_sizze_98338, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96197_cached_sizze_98339 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96197, &mem_96197_cached_sizze_98339, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96198_cached_sizze_98340 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96198, &mem_96198_cached_sizze_98340, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96211_cached_sizze_98341 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96211, &mem_96211_cached_sizze_98341, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96212_cached_sizze_98342 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96212, &mem_96212_cached_sizze_98342, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96213_cached_sizze_98343 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96213, &mem_96213_cached_sizze_98343, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95226 = 0; i_95226 < (int64_t) 16; i_95226++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95216 = 0; i_95216 < (int64_t) 16; i_95216++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91151;
            double r_91153 = 0.0;
            
            for (int64_t i_91152 = 0; i_91152 < (int64_t) 16; i_91152++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91154 = ((double *) wqry_mem_96111.mem)[i_95216 * (int64_t) 16 + i_91152];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91155 = ((double *) mem_96173)[i_95226 * (int64_t) 16 + i_91152];
                
                // futhark/microgpt.fut:167:66-105
                
                double zt_res_91156 = zt_lhs_91154 * zt_rhs_91155;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91157 = r_91153 + zt_res_91156;
                double r_tmp_97993 = zp_res_91157;
                
                r_91153 = r_tmp_97993;
            }
            defunc_0_lifted_lambda_res_91151 = r_91153;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91164;
            double r_91166 = 0.0;
            
            for (int64_t i_91165 = 0; i_91165 < (int64_t) 16; i_91165++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91167 = ((double *) wkey_mem_96108.mem)[i_95216 * (int64_t) 16 + i_91165];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91168 = ((double *) mem_96173)[i_95226 * (int64_t) 16 + i_91165];
                
                // futhark/microgpt.fut:168:66-105
                
                double zt_res_91169 = zt_lhs_91167 * zt_rhs_91168;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91170 = r_91166 + zt_res_91169;
                double r_tmp_97994 = zp_res_91170;
                
                r_91166 = r_tmp_97994;
            }
            defunc_0_lifted_lambda_res_91164 = r_91166;
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91180;
            double r_91182 = 0.0;
            
            for (int64_t i_91181 = 0; i_91181 < (int64_t) 16; i_91181++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91183 = ((double *) wval_mem_96114.mem)[i_95216 * (int64_t) 16 + i_91181];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_91184 = ((double *) mem_96173)[i_95226 * (int64_t) 16 + i_91181];
                
                // futhark/microgpt.fut:169:66-105
                
                double zt_res_91185 = zt_lhs_91183 * zt_rhs_91184;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91186 = r_91182 + zt_res_91185;
                double r_tmp_97995 = zp_res_91186;
                
                r_91182 = r_tmp_97995;
            }
            defunc_0_lifted_lambda_res_91180 = r_91182;
            ((double *) mem_96211)[i_95216] = defunc_0_lifted_lambda_res_91180;
            ((double *) mem_96212)[i_95216] = defunc_0_lifted_lambda_res_91164;
            ((double *) mem_96213)[i_95216] = defunc_0_lifted_lambda_res_91151;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96196, i_95226 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96211, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96197, i_95226 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96212, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96198, i_95226 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96213, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96244_cached_sizze_98344 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96244, &mem_96244_cached_sizze_98344, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96245_cached_sizze_98345 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96245, &mem_96245_cached_sizze_98345, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96246_cached_sizze_98346 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96246, &mem_96246_cached_sizze_98346, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96262_cached_sizze_98347 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96262, &mem_96262_cached_sizze_98347, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96263_cached_sizze_98348 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96263, &mem_96263_cached_sizze_98348, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96264_cached_sizze_98349 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96264, &mem_96264_cached_sizze_98349, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96277_cached_sizze_98350 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96277, &mem_96277_cached_sizze_98350, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96278_cached_sizze_98351 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96278, &mem_96278_cached_sizze_98351, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96279_cached_sizze_98352 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96279, &mem_96279_cached_sizze_98352, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95256 = 0; i_95256 < (int64_t) 4; i_95256++) {
        // futhark/microgpt.fut:170:69-72
        
        int64_t zp_lhs_91026 = mul64((int64_t) 4, i_95256);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95246 = 0; i_95246 < (int64_t) 16; i_95246++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95236 = 0; i_95236 < (int64_t) 4; i_95236++) {
                // futhark/microgpt.fut:170:74-81
                
                int64_t tmp_91344 = add64(zp_lhs_91026, i_95236);
                
                // futhark/microgpt.fut:170:51-83
                
                bool x_91345 = sle64((int64_t) 0, tmp_91344);
                
                // futhark/microgpt.fut:170:51-83
                
                bool y_91346 = slt64(tmp_91344, (int64_t) 16);
                
                // futhark/microgpt.fut:170:51-83
                
                bool bounds_check_91347 = x_91345 && y_91346;
                
                // futhark/microgpt.fut:170:51-83
                
                bool index_certs_91348;
                
                if (!bounds_check_91347) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_91344, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:170:51-83\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:170:15-84\n   #9  futhark/microgpt.fut:398:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_91349 = ((double *) mem_96198)[i_95246 * (int64_t) 16 + tmp_91344];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_91357 = ((double *) mem_96197)[i_95246 * (int64_t) 16 + tmp_91344];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_91368 = ((double *) mem_96196)[i_95246 * (int64_t) 16 + tmp_91344];
                
                ((double *) mem_96277)[i_95236] = lifted_lambda_res_91368;
                ((double *) mem_96278)[i_95236] = lifted_lambda_res_91357;
                ((double *) mem_96279)[i_95236] = lifted_lambda_res_91349;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96262, i_95246 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96263, i_95246 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96278, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96264, i_95246 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96279, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_96244, i_95256 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96262, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_96245, i_95256 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96263, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_96246, i_95256 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96264, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96325_cached_sizze_98353 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96325, &mem_96325_cached_sizze_98353, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96331_cached_sizze_98354 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96331, &mem_96331_cached_sizze_98354, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96336_cached_sizze_98355 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96336, &mem_96336_cached_sizze_98355, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96347_cached_sizze_98356 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96347, &mem_96347_cached_sizze_98356, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96352_cached_sizze_98357 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96352, &mem_96352_cached_sizze_98357, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96363_cached_sizze_98358 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96363, &mem_96363_cached_sizze_98358, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96368_cached_sizze_98359 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96368, &mem_96368_cached_sizze_98359, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96375_cached_sizze_98360 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96375, &mem_96375_cached_sizze_98360, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96386_cached_sizze_98361 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96386, &mem_96386_cached_sizze_98361, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96391_cached_sizze_98362 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96391, &mem_96391_cached_sizze_98362, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95301 = 0; i_95301 < (int64_t) 4; i_95301++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95266 = 0; i_95266 < (int64_t) 16; i_95266++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95262 = 0; i_95262 < (int64_t) 16; i_95262++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84773;
                double r_84775 = 0.0;
                
                for (int64_t i_84774 = 0; i_84774 < (int64_t) 4; i_84774++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84776 = ((double *) mem_96246)[i_95301 * (int64_t) 64 + i_95266 * (int64_t) 4 + i_84774];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84777 = ((double *) mem_96245)[i_95301 * (int64_t) 64 + i_95262 * (int64_t) 4 + i_84774];
                    
                    // futhark/microgpt.fut:173:113-164
                    
                    double zt_res_84778 = zt_lhs_84776 * zt_rhs_84777;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84779 = r_84775 + zt_res_84778;
                    double r_tmp_98008 = zp_res_84779;
                    
                    r_84775 = r_tmp_98008;
                }
                defunc_0_lifted_lambda_res_84773 = r_84775;
                ((double *) mem_96336)[i_95262] = defunc_0_lifted_lambda_res_84773;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96331, i_95266 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96336, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95274 = 0; i_95274 < (int64_t) 16; i_95274++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95270 = 0; i_95270 < (int64_t) 16; i_95270++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_lhs_84794 = ((double *) mem_96331)[i_95274 * (int64_t) 16 + i_95270];
                
                // futhark/microgpt.fut:174:47-78
                
                double zs_res_84795 = zs_lhs_84794 / 2.0;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_84796 = ((double *) mask_mem_96117.mem)[i_95274 * (int64_t) 16 + i_95270];
                
                // futhark/microgpt.fut:174:65-102
                
                double zp_res_84797 = zs_res_84795 + zp_rhs_84796;
                
                ((double *) mem_96352)[i_95270] = zp_res_84797;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96347, i_95274 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96352, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95289 = 0; i_95289 < (int64_t) 16; i_95289++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_91447;
            int64_t defunc_0_reduce_res_91448;
            double redout_95276;
            int64_t redout_95277;
            
            redout_95276 = -INFINITY;
            redout_95277 = (int64_t) 16;
            for (int64_t i_95278 = 0; i_95278 < (int64_t) 16; i_95278++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_91398 = ((double *) mem_96347)[i_95289 * (int64_t) 16 + i_95278];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_84822 = lifted_lambda_res_91398 < redout_95276;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_84823;
                
                if (zg_res_84822) {
                    lifted_lambda_res_84823 = redout_95276;
                } else {
                    lifted_lambda_res_84823 = lifted_lambda_res_91398;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_84824;
                
                if (zg_res_84822) {
                    lifted_lambda_res_84824 = redout_95277;
                } else {
                    lifted_lambda_res_84824 = i_95278;
                }
                
                double redout_tmp_98012 = lifted_lambda_res_84823;
                int64_t redout_tmp_98013 = lifted_lambda_res_84824;
                
                redout_95276 = redout_tmp_98012;
                redout_95277 = redout_tmp_98013;
            }
            defunc_0_reduce_res_91447 = redout_95276;
            defunc_0_reduce_res_91448 = redout_95277;
            // futhark/microgpt.fut:175:56-112
            
            bool x_84825 = sle64((int64_t) 0, defunc_0_reduce_res_91448);
            
            // futhark/microgpt.fut:175:56-112
            
            bool y_84826 = slt64(defunc_0_reduce_res_91448, (int64_t) 16);
            
            // futhark/microgpt.fut:175:56-112
            
            bool bounds_check_84827 = x_84825 && y_84826;
            
            // futhark/microgpt.fut:175:56-112
            
            bool index_certs_84828;
            
            if (!bounds_check_84827) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_91448, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:175:56-112\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:175:16-178:38\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:27-39\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:9:13-40\n   #10 futhark/microgpt.fut:15:29-44\n   #11 futhark/microgpt.fut:4:11-25\n   #12 futhark/microgpt.fut:15:15-45\n   #13 futhark/microgpt.fut:173:15-179:78\n   #14 futhark/microgpt.fut:398:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double x49_84829 = ((double *) mem_96347)[i_95289 * (int64_t) 16 + defunc_0_reduce_res_91448];
            
            // futhark/microgpt.fut:176:67-76
            
            double neg_res_84830 = -x49_84829;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95281 = 0; i_95281 < (int64_t) 16; i_95281++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_84837 = ((double *) mem_96347)[i_95289 * (int64_t) 16 + i_95281];
                
                // futhark/microgpt.fut:176:44-76
                
                double zp_res_84838 = neg_res_84830 + zp_lhs_84837;
                
                // futhark/microgpt.fut:176:37-76
                
                double exp_res_84839 = futrts_exp64(zp_res_84838);
                
                ((double *) mem_96368)[i_95281] = exp_res_84839;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84841;
            double r_84843 = 0.0;
            
            for (int64_t i_84842 = 0; i_84842 < (int64_t) 16; i_84842++) {
                // futhark/microgpt.fut:177:36-46
                
                double lifted_lambda_res_84844 = ((double *) mem_96368)[i_84842];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84845 = r_84843 + lifted_lambda_res_84844;
                double r_tmp_98015 = zp_res_84845;
                
                r_84843 = r_tmp_98015;
            }
            defunc_0_lifted_lambda_res_84841 = r_84843;
            // futhark/microgpt.fut:178:21-32
            
            double zs_res_84846 = 1.0 / defunc_0_lifted_lambda_res_84841;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95285 = 0; i_95285 < (int64_t) 16; i_95285++) {
                // futhark/microgpt.fut:178:5-15
                
                double zt_lhs_84853 = ((double *) mem_96368)[i_95285];
                
                // futhark/microgpt.fut:178:5-32
                
                double zt_res_84854 = zs_res_84846 * zt_lhs_84853;
                
                ((double *) mem_96375)[i_95285] = zt_res_84854;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96363, i_95289 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96375, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95297 = 0; i_95297 < (int64_t) 16; i_95297++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95293 = 0; i_95293 < (int64_t) 4; i_95293++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_84869;
                double r_84871 = 0.0;
                
                for (int64_t i_84870 = 0; i_84870 < (int64_t) 16; i_84870++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_84872 = ((double *) mem_96363)[i_95297 * (int64_t) 16 + i_84870];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_84873 = ((double *) mem_96244)[i_95301 * (int64_t) 64 + i_84870 * (int64_t) 4 + i_95293];
                    
                    // futhark/microgpt.fut:179:26-71
                    
                    double zt_res_84874 = zt_lhs_84872 * zt_rhs_84873;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_84875 = r_84871 + zt_res_84874;
                    double r_tmp_98019 = zp_res_84875;
                    
                    r_84871 = r_tmp_98019;
                }
                defunc_0_lifted_lambda_res_84869 = r_84871;
                ((double *) mem_96391)[i_95293] = defunc_0_lifted_lambda_res_84869;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96386, i_95297 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96391, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_96325, i_95301 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96386, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96407_cached_sizze_98363 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96407, &mem_96407_cached_sizze_98363, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96412_cached_sizze_98364 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96412, &mem_96412_cached_sizze_98364, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95309 = 0; i_95309 < (int64_t) 16; i_95309++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95305 = 0; i_95305 < (int64_t) 16; i_95305++) {
            // futhark/microgpt.fut:180:55-58
            
            int64_t tmp_84887 = sdiv64(i_95305, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-60
            
            bool x_84888 = sle64((int64_t) 0, tmp_84887);
            
            // futhark/microgpt.fut:180:45-60
            
            bool y_84889 = slt64(tmp_84887, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-60
            
            bool bounds_check_84890 = x_84888 && y_84889;
            
            // futhark/microgpt.fut:180:45-60
            
            bool index_certs_84891;
            
            if (!bounds_check_84890) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84887, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:180:45-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:180:16-81\n   #6  futhark/microgpt.fut:398:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:180:75-78
            
            int64_t tmp_84892 = smod64(i_95305, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-80
            
            bool x_84893 = sle64((int64_t) 0, tmp_84892);
            
            // futhark/microgpt.fut:180:45-80
            
            bool y_84894 = slt64(tmp_84892, (int64_t) 4);
            
            // futhark/microgpt.fut:180:45-80
            
            bool bounds_check_84895 = x_84893 && y_84894;
            
            // futhark/microgpt.fut:180:45-80
            
            bool index_certs_84896;
            
            if (!bounds_check_84895) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_84892, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:180:45-80\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:180:16-81\n   #6  futhark/microgpt.fut:398:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_84897 = ((double *) mem_96325)[tmp_84887 * (int64_t) 64 + i_95309 * (int64_t) 4 + tmp_84892];
            
            ((double *) mem_96412)[i_95305] = lifted_lambda_res_84897;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96407, i_95309 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96412, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96423_cached_sizze_98365 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96423, &mem_96423_cached_sizze_98365, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96428_cached_sizze_98366 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96428, &mem_96428_cached_sizze_98366, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95317 = 0; i_95317 < (int64_t) 16; i_95317++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95313 = 0; i_95313 < (int64_t) 16; i_95313++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84912;
            double r_84914 = 0.0;
            
            for (int64_t i_84913 = 0; i_84913 < (int64_t) 16; i_84913++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84915 = ((double *) wout_mem_96109.mem)[i_95313 * (int64_t) 16 + i_84913];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_84916 = ((double *) mem_96407)[i_95317 * (int64_t) 16 + i_84913];
                
                // futhark/microgpt.fut:181:67-107
                
                double zt_res_84917 = zt_lhs_84915 * zt_rhs_84916;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84918 = r_84914 + zt_res_84917;
                double r_tmp_98024 = zp_res_84918;
                
                r_84914 = r_tmp_98024;
            }
            defunc_0_lifted_lambda_res_84912 = r_84914;
            ((double *) mem_96428)[i_95313] = defunc_0_lifted_lambda_res_84912;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96423, i_95317 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96428, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96439_cached_sizze_98367 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96439, &mem_96439_cached_sizze_98367, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96444_cached_sizze_98368 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96444, &mem_96444_cached_sizze_98368, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95325 = 0; i_95325 < (int64_t) 16; i_95325++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95321 = 0; i_95321 < (int64_t) 16; i_95321++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_84933 = ((double *) mem_96423)[i_95325 * (int64_t) 16 + i_95321];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_84934 = ((double *) mem_96150)[i_95325 * (int64_t) 16 + i_95321];
            
            // futhark/microgpt.fut:182:46-84
            
            double zp_res_84935 = zp_lhs_84933 + zp_rhs_84934;
            
            ((double *) mem_96444)[i_95321] = zp_res_84935;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96439, i_95325 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96455_cached_sizze_98369 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96455, &mem_96455_cached_sizze_98369, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96460_cached_sizze_98370 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96460, &mem_96460_cached_sizze_98370, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96467_cached_sizze_98371 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96467, &mem_96467_cached_sizze_98371, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95337 = 0; i_95337 < (int64_t) 16; i_95337++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95329 = 0; i_95329 < (int64_t) 16; i_95329++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84950 = ((double *) mem_96439)[i_95337 * (int64_t) 16 + i_95329];
            
            // futhark/microgpt.fut:183:78-117
            
            double zt_res_84951 = zt_lhs_84950 * zt_lhs_84950;
            
            ((double *) mem_96460)[i_95329] = zt_res_84951;
        }
        // futhark/microgpt.fut:71:13-49
        
        double defunc_0_lifted_lambda_res_84953;
        double r_84955 = 0.0;
        
        for (int64_t i_84954 = 0; i_84954 < (int64_t) 16; i_84954++) {
            // futhark/microgpt.fut:184:37-47
            
            double lifted_lambda_res_84956 = ((double *) mem_96460)[i_84954];
            
            // futhark/microgpt.fut:71:40-49
            
            double zp_res_84957 = r_84955 + lifted_lambda_res_84956;
            double r_tmp_98029 = zp_res_84957;
            
            r_84955 = r_tmp_98029;
        }
        defunc_0_lifted_lambda_res_84953 = r_84955;
        // futhark/microgpt.fut:184:17-64
        
        double zs_res_84958 = defunc_0_lifted_lambda_res_84953 / 16.0;
        
        // futhark/microgpt.fut:185:24-55
        
        double zp_res_84959 = 1.0e-5 + zs_res_84958;
        
        // futhark/microgpt.fut:185:16-55
        
        double sqrt_res_84960 = futrts_sqrt64(zp_res_84959);
        
        // futhark/microgpt.fut:186:28-39
        
        double zs_res_84961 = 1.0 / sqrt_res_84960;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95333 = 0; i_95333 < (int64_t) 16; i_95333++) {
            // futhark/microgpt.fut:4:11-25
            
            double zt_lhs_84968 = ((double *) mem_96439)[i_95337 * (int64_t) 16 + i_95333];
            
            // futhark/microgpt.fut:186:5-39
            
            double zt_res_84969 = zs_res_84961 * zt_lhs_84968;
            
            ((double *) mem_96467)[i_95333] = zt_res_84969;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96455, i_95337 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96467, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96478_cached_sizze_98372 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96478, &mem_96478_cached_sizze_98372, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96483_cached_sizze_98373 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96483, &mem_96483_cached_sizze_98373, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95345 = 0; i_95345 < (int64_t) 16; i_95345++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95341 = 0; i_95341 < (int64_t) 64; i_95341++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_84985;
            double r_84987 = 0.0;
            
            for (int64_t i_84986 = 0; i_84986 < (int64_t) 16; i_84986++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_84988 = ((double *) wup_mem_96113.mem)[i_95341 * (int64_t) 16 + i_84986];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_84989 = ((double *) mem_96455)[i_95345 * (int64_t) 16 + i_84986];
                
                // futhark/microgpt.fut:187:67-106
                
                double zt_res_84990 = zt_lhs_84988 * zt_rhs_84989;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_84991 = r_84987 + zt_res_84990;
                double r_tmp_98033 = zp_res_84991;
                
                r_84987 = r_tmp_98033;
            }
            defunc_0_lifted_lambda_res_84985 = r_84987;
            ((double *) mem_96483)[i_95341] = defunc_0_lifted_lambda_res_84985;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96478, i_95345 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96483, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96494_cached_sizze_98374 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96494, &mem_96494_cached_sizze_98374, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96499_cached_sizze_98375 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96499, &mem_96499_cached_sizze_98375, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95353 = 0; i_95353 < (int64_t) 16; i_95353++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95349 = 0; i_95349 < (int64_t) 64; i_95349++) {
            // futhark/microgpt.fut:4:11-25
            
            double max_arg0_85006 = ((double *) mem_96478)[i_95353 * (int64_t) 64 + i_95349];
            
            // futhark/microgpt.fut:188:45-73
            
            double max_res_85007 = fmax64(0.0, max_arg0_85006);
            
            ((double *) mem_96499)[i_95349] = max_res_85007;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96494, i_95353 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96499, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96510_cached_sizze_98376 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96510, &mem_96510_cached_sizze_98376, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96515_cached_sizze_98377 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96515, &mem_96515_cached_sizze_98377, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95361 = 0; i_95361 < (int64_t) 16; i_95361++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95357 = 0; i_95357 < (int64_t) 16; i_95357++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85022;
            double r_85024 = 0.0;
            
            for (int64_t i_85023 = 0; i_85023 < (int64_t) 64; i_85023++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85025 = ((double *) wdown_mem_96107.mem)[i_95357 * (int64_t) 64 + i_85023];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85026 = ((double *) mem_96494)[i_95361 * (int64_t) 64 + i_85023];
                
                // futhark/microgpt.fut:189:67-108
                
                double zt_res_85027 = zt_lhs_85025 * zt_rhs_85026;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85028 = r_85024 + zt_res_85027;
                double r_tmp_98038 = zp_res_85028;
                
                r_85024 = r_tmp_98038;
            }
            defunc_0_lifted_lambda_res_85022 = r_85024;
            ((double *) mem_96515)[i_95357] = defunc_0_lifted_lambda_res_85022;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96510, i_95361 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96515, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96526_cached_sizze_98378 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96526, &mem_96526_cached_sizze_98378, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96531_cached_sizze_98379 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96531, &mem_96531_cached_sizze_98379, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95369 = 0; i_95369 < (int64_t) 16; i_95369++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95365 = 0; i_95365 < (int64_t) 16; i_95365++) {
            // futhark/microgpt.fut:4:11-25
            
            double zp_lhs_85043 = ((double *) mem_96510)[i_95369 * (int64_t) 16 + i_95365];
            
            // futhark/microgpt.fut:4:11-25
            
            double zp_rhs_85044 = ((double *) mem_96439)[i_95369 * (int64_t) 16 + i_95365];
            
            // futhark/microgpt.fut:190:46-85
            
            double zp_res_85045 = zp_lhs_85043 + zp_rhs_85044;
            
            ((double *) mem_96531)[i_95365] = zp_res_85045;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96526, i_95369 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96531, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_96542, (int64_t) 3456, "mem_96542")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96547_cached_sizze_98380 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96547, &mem_96547_cached_sizze_98380, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_95377 = 0; i_95377 < (int64_t) 16; i_95377++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95373 = 0; i_95373 < (int64_t) 27; i_95373++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_85061;
            double r_85063 = 0.0;
            
            for (int64_t i_85062 = 0; i_85062 < (int64_t) 16; i_85062++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_85064 = ((double *) wvoc_mem_96115.mem)[i_95373 * (int64_t) 16 + i_85062];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_85065 = ((double *) mem_96526)[i_95377 * (int64_t) 16 + i_85062];
                
                // futhark/microgpt.fut:191:56-96
                
                double zt_res_85066 = zt_lhs_85064 * zt_rhs_85065;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_85067 = r_85063 + zt_res_85066;
                double r_tmp_98043 = zp_res_85067;
                
                r_85063 = r_tmp_98043;
            }
            defunc_0_lifted_lambda_res_85061 = r_85063;
            ((double *) mem_96547)[i_95373] = defunc_0_lifted_lambda_res_85061;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_96542.mem, i_95377 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96547, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_97974, &mem_96542, "mem_96542") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98327, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_96118);
        free(mem_96123);
        free(mem_96134);
        free(mem_96139);
        free(mem_96150);
        free(mem_96155);
        free(mem_96162);
        free(mem_96173);
        free(mem_96178);
        free(mem_96185);
        free(mem_96196);
        free(mem_96197);
        free(mem_96198);
        free(mem_96211);
        free(mem_96212);
        free(mem_96213);
        free(mem_96244);
        free(mem_96245);
        free(mem_96246);
        free(mem_96262);
        free(mem_96263);
        free(mem_96264);
        free(mem_96277);
        free(mem_96278);
        free(mem_96279);
        free(mem_96325);
        free(mem_96331);
        free(mem_96336);
        free(mem_96347);
        free(mem_96352);
        free(mem_96363);
        free(mem_96368);
        free(mem_96375);
        free(mem_96386);
        free(mem_96391);
        free(mem_96407);
        free(mem_96412);
        free(mem_96423);
        free(mem_96428);
        free(mem_96439);
        free(mem_96444);
        free(mem_96455);
        free(mem_96460);
        free(mem_96467);
        free(mem_96478);
        free(mem_96483);
        free(mem_96494);
        free(mem_96499);
        free(mem_96510);
        free(mem_96515);
        free(mem_96526);
        free(mem_96531);
        free(mem_96547);
        if (memblock_unref(ctx, &mem_96542, "mem_96542") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_98381, struct memblock *mem_out_p_98382, struct memblock *mem_out_p_98383, struct memblock *mem_out_p_98384, struct memblock *mem_out_p_98385, struct memblock *mem_out_p_98386, struct memblock *mem_out_p_98387, struct memblock *mem_out_p_98388, struct memblock *mem_out_p_98389, struct memblock wte_mem_96107, struct memblock wpe_mem_96108, struct memblock wqry_mem_96109, struct memblock wkey_mem_96110, struct memblock wval_mem_96111, struct memblock wout_mem_96112, struct memblock wup_mem_96113, struct memblock wdown_mem_96114, struct memblock wvoc_mem_96115)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    if (memblock_set(ctx, &mem_out_97974, &wdown_mem_96114, "wdown_mem_96114") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97975, &wkey_mem_96110, "wkey_mem_96110") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97976, &wout_mem_96112, "wout_mem_96112") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97977, &wpe_mem_96108, "wpe_mem_96108") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97978, &wqry_mem_96109, "wqry_mem_96109") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97979, &wte_mem_96107, "wte_mem_96107") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97980, &wup_mem_96113, "wup_mem_96113") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97981, &wval_mem_96111, "wval_mem_96111") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97982, &wvoc_mem_96115, "wvoc_mem_96115") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98381, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98382, &mem_out_97975, "mem_out_97975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98383, &mem_out_97976, "mem_out_97976") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98384, &mem_out_97977, "mem_out_97977") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98385, &mem_out_97978, "mem_out_97978") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98386, &mem_out_97979, "mem_out_97979") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98387, &mem_out_97980, "mem_out_97980") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98388, &mem_out_97981, "mem_out_97981") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98389, &mem_out_97982, "mem_out_97982") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_97982, "mem_out_97982") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97981, "mem_out_97981") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97980, "mem_out_97980") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97979, "mem_out_97979") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97978, "mem_out_97978") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97977, "mem_out_97977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97976, "mem_out_97976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97975, "mem_out_97975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_98390, struct memblock *mem_out_p_98391, struct memblock *mem_out_p_98392, struct memblock *mem_out_p_98393, struct memblock *mem_out_p_98394, struct memblock *mem_out_p_98395, struct memblock *mem_out_p_98396, struct memblock *mem_out_p_98397, struct memblock *mem_out_p_98398, struct memblock *mem_out_p_98399, struct memblock *mem_out_p_98400, struct memblock *mem_out_p_98401, struct memblock *mem_out_p_98402, struct memblock *mem_out_p_98403, struct memblock *mem_out_p_98404, struct memblock *mem_out_p_98405, struct memblock *mem_out_p_98406, struct memblock *mem_out_p_98407, struct memblock *mem_out_p_98408, struct memblock *mem_out_p_98409, struct memblock *mem_out_p_98410, struct memblock *mem_out_p_98411, struct memblock *mem_out_p_98412, struct memblock *mem_out_p_98413, struct memblock *mem_out_p_98414, struct memblock *mem_out_p_98415, struct memblock *mem_out_p_98416, struct memblock wdown_mem_96107, struct memblock wkey_mem_96108, struct memblock wout_mem_96109, struct memblock wpe_mem_96110, struct memblock wqry_mem_96111, struct memblock wte_mem_96112, struct memblock wup_mem_96113, struct memblock wval_mem_96114, struct memblock wvoc_mem_96115, struct memblock wdown_mem_96116, struct memblock wkey_mem_96117, struct memblock wout_mem_96118, struct memblock wpe_mem_96119, struct memblock wqry_mem_96120, struct memblock wte_mem_96121, struct memblock wup_mem_96122, struct memblock wval_mem_96123, struct memblock wvoc_mem_96124, struct memblock wdown_mem_96125, struct memblock wkey_mem_96126, struct memblock wout_mem_96127, struct memblock wpe_mem_96128, struct memblock wqry_mem_96129, struct memblock wte_mem_96130, struct memblock wup_mem_96131, struct memblock wval_mem_96132, struct memblock wvoc_mem_96133, struct memblock masks_mem_96134, struct memblock dls_mem_96135, struct memblock seqs_mem_96136)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_96245_cached_sizze_98417 = 0;
    unsigned char *mem_96245 = NULL;
    int64_t mem_96246_cached_sizze_98418 = 0;
    unsigned char *mem_96246 = NULL;
    int64_t mem_96255_cached_sizze_98419 = 0;
    unsigned char *mem_96255 = NULL;
    int64_t mem_96262_cached_sizze_98420 = 0;
    unsigned char *mem_96262 = NULL;
    int64_t mem_96277_cached_sizze_98421 = 0;
    unsigned char *mem_96277 = NULL;
    int64_t mem_96278_cached_sizze_98422 = 0;
    unsigned char *mem_96278 = NULL;
    int64_t mem_96287_cached_sizze_98423 = 0;
    unsigned char *mem_96287 = NULL;
    int64_t mem_96294_cached_sizze_98424 = 0;
    unsigned char *mem_96294 = NULL;
    int64_t mem_96309_cached_sizze_98425 = 0;
    unsigned char *mem_96309 = NULL;
    int64_t mem_96310_cached_sizze_98426 = 0;
    unsigned char *mem_96310 = NULL;
    int64_t mem_96319_cached_sizze_98427 = 0;
    unsigned char *mem_96319 = NULL;
    int64_t mem_96320_cached_sizze_98428 = 0;
    unsigned char *mem_96320 = NULL;
    int64_t mem_96341_cached_sizze_98429 = 0;
    unsigned char *mem_96341 = NULL;
    int64_t mem_96342_cached_sizze_98430 = 0;
    unsigned char *mem_96342 = NULL;
    int64_t mem_96343_cached_sizze_98431 = 0;
    unsigned char *mem_96343 = NULL;
    int64_t mem_96355_cached_sizze_98432 = 0;
    unsigned char *mem_96355 = NULL;
    int64_t mem_96356_cached_sizze_98433 = 0;
    unsigned char *mem_96356 = NULL;
    int64_t mem_96380_cached_sizze_98434 = 0;
    unsigned char *mem_96380 = NULL;
    int64_t mem_96381_cached_sizze_98435 = 0;
    unsigned char *mem_96381 = NULL;
    int64_t mem_96382_cached_sizze_98436 = 0;
    unsigned char *mem_96382 = NULL;
    int64_t mem_96383_cached_sizze_98437 = 0;
    unsigned char *mem_96383 = NULL;
    int64_t mem_96384_cached_sizze_98438 = 0;
    unsigned char *mem_96384 = NULL;
    int64_t mem_96403_cached_sizze_98439 = 0;
    unsigned char *mem_96403 = NULL;
    int64_t mem_96404_cached_sizze_98440 = 0;
    unsigned char *mem_96404 = NULL;
    int64_t mem_96405_cached_sizze_98441 = 0;
    unsigned char *mem_96405 = NULL;
    int64_t mem_96442_cached_sizze_98442 = 0;
    unsigned char *mem_96442 = NULL;
    int64_t mem_96443_cached_sizze_98443 = 0;
    unsigned char *mem_96443 = NULL;
    int64_t mem_96444_cached_sizze_98444 = 0;
    unsigned char *mem_96444 = NULL;
    int64_t mem_96460_cached_sizze_98445 = 0;
    unsigned char *mem_96460 = NULL;
    int64_t mem_96461_cached_sizze_98446 = 0;
    unsigned char *mem_96461 = NULL;
    int64_t mem_96462_cached_sizze_98447 = 0;
    unsigned char *mem_96462 = NULL;
    int64_t mem_96475_cached_sizze_98448 = 0;
    unsigned char *mem_96475 = NULL;
    int64_t mem_96476_cached_sizze_98449 = 0;
    unsigned char *mem_96476 = NULL;
    int64_t mem_96477_cached_sizze_98450 = 0;
    unsigned char *mem_96477 = NULL;
    int64_t mem_96523_cached_sizze_98451 = 0;
    unsigned char *mem_96523 = NULL;
    int64_t mem_96524_cached_sizze_98452 = 0;
    unsigned char *mem_96524 = NULL;
    int64_t mem_96535_cached_sizze_98453 = 0;
    unsigned char *mem_96535 = NULL;
    int64_t mem_96536_cached_sizze_98454 = 0;
    unsigned char *mem_96536 = NULL;
    int64_t mem_96545_cached_sizze_98455 = 0;
    unsigned char *mem_96545 = NULL;
    int64_t mem_96546_cached_sizze_98456 = 0;
    unsigned char *mem_96546 = NULL;
    int64_t mem_96567_cached_sizze_98457 = 0;
    unsigned char *mem_96567 = NULL;
    int64_t mem_96572_cached_sizze_98458 = 0;
    unsigned char *mem_96572 = NULL;
    int64_t mem_96583_cached_sizze_98459 = 0;
    unsigned char *mem_96583 = NULL;
    int64_t mem_96588_cached_sizze_98460 = 0;
    unsigned char *mem_96588 = NULL;
    int64_t mem_96595_cached_sizze_98461 = 0;
    unsigned char *mem_96595 = NULL;
    int64_t mem_96606_cached_sizze_98462 = 0;
    unsigned char *mem_96606 = NULL;
    int64_t mem_96611_cached_sizze_98463 = 0;
    unsigned char *mem_96611 = NULL;
    int64_t mem_96632_cached_sizze_98464 = 0;
    unsigned char *mem_96632 = NULL;
    int64_t mem_96633_cached_sizze_98465 = 0;
    unsigned char *mem_96633 = NULL;
    int64_t mem_96641_cached_sizze_98466 = 0;
    unsigned char *mem_96641 = NULL;
    int64_t mem_96655_cached_sizze_98467 = 0;
    unsigned char *mem_96655 = NULL;
    int64_t mem_96660_cached_sizze_98468 = 0;
    unsigned char *mem_96660 = NULL;
    int64_t mem_96671_cached_sizze_98469 = 0;
    unsigned char *mem_96671 = NULL;
    int64_t mem_96676_cached_sizze_98470 = 0;
    unsigned char *mem_96676 = NULL;
    int64_t mem_96687_cached_sizze_98471 = 0;
    unsigned char *mem_96687 = NULL;
    int64_t mem_96688_cached_sizze_98472 = 0;
    unsigned char *mem_96688 = NULL;
    int64_t mem_96697_cached_sizze_98473 = 0;
    unsigned char *mem_96697 = NULL;
    int64_t mem_96698_cached_sizze_98474 = 0;
    unsigned char *mem_96698 = NULL;
    int64_t mem_96719_cached_sizze_98475 = 0;
    unsigned char *mem_96719 = NULL;
    int64_t mem_96720_cached_sizze_98476 = 0;
    unsigned char *mem_96720 = NULL;
    int64_t mem_96728_cached_sizze_98477 = 0;
    unsigned char *mem_96728 = NULL;
    int64_t mem_96742_cached_sizze_98478 = 0;
    unsigned char *mem_96742 = NULL;
    int64_t mem_96743_cached_sizze_98479 = 0;
    unsigned char *mem_96743 = NULL;
    int64_t mem_96751_cached_sizze_98480 = 0;
    unsigned char *mem_96751 = NULL;
    int64_t mem_96765_cached_sizze_98481 = 0;
    unsigned char *mem_96765 = NULL;
    int64_t mem_96770_cached_sizze_98482 = 0;
    unsigned char *mem_96770 = NULL;
    int64_t mem_96781_cached_sizze_98483 = 0;
    unsigned char *mem_96781 = NULL;
    int64_t mem_96786_cached_sizze_98484 = 0;
    unsigned char *mem_96786 = NULL;
    int64_t mem_96797_cached_sizze_98485 = 0;
    unsigned char *mem_96797 = NULL;
    int64_t mem_96802_cached_sizze_98486 = 0;
    unsigned char *mem_96802 = NULL;
    int64_t mem_96813_cached_sizze_98487 = 0;
    unsigned char *mem_96813 = NULL;
    int64_t mem_96820_cached_sizze_98488 = 0;
    unsigned char *mem_96820 = NULL;
    int64_t mem_96825_cached_sizze_98489 = 0;
    unsigned char *mem_96825 = NULL;
    int64_t mem_96836_cached_sizze_98490 = 0;
    unsigned char *mem_96836 = NULL;
    int64_t mem_96843_cached_sizze_98491 = 0;
    unsigned char *mem_96843 = NULL;
    int64_t mem_96847_cached_sizze_98492 = 0;
    unsigned char *mem_96847 = NULL;
    int64_t mem_96857_cached_sizze_98493 = 0;
    unsigned char *mem_96857 = NULL;
    int64_t mem_96862_cached_sizze_98494 = 0;
    unsigned char *mem_96862 = NULL;
    int64_t mem_96869_cached_sizze_98495 = 0;
    unsigned char *mem_96869 = NULL;
    int64_t mem_96880_cached_sizze_98496 = 0;
    unsigned char *mem_96880 = NULL;
    int64_t mem_96887_cached_sizze_98497 = 0;
    unsigned char *mem_96887 = NULL;
    int64_t mem_96892_cached_sizze_98498 = 0;
    unsigned char *mem_96892 = NULL;
    int64_t mem_96903_cached_sizze_98499 = 0;
    unsigned char *mem_96903 = NULL;
    int64_t mem_96908_cached_sizze_98500 = 0;
    unsigned char *mem_96908 = NULL;
    int64_t mem_96919_cached_sizze_98501 = 0;
    unsigned char *mem_96919 = NULL;
    int64_t mem_96920_cached_sizze_98502 = 0;
    unsigned char *mem_96920 = NULL;
    int64_t mem_96929_cached_sizze_98503 = 0;
    unsigned char *mem_96929 = NULL;
    int64_t mem_96930_cached_sizze_98504 = 0;
    unsigned char *mem_96930 = NULL;
    int64_t mem_96951_cached_sizze_98505 = 0;
    unsigned char *mem_96951 = NULL;
    int64_t mem_96956_cached_sizze_98506 = 0;
    unsigned char *mem_96956 = NULL;
    int64_t mem_96967_cached_sizze_98507 = 0;
    unsigned char *mem_96967 = NULL;
    int64_t mem_96972_cached_sizze_98508 = 0;
    unsigned char *mem_96972 = NULL;
    int64_t mem_96983_cached_sizze_98509 = 0;
    unsigned char *mem_96983 = NULL;
    int64_t mem_96990_cached_sizze_98510 = 0;
    unsigned char *mem_96990 = NULL;
    int64_t mem_96997_cached_sizze_98511 = 0;
    unsigned char *mem_96997 = NULL;
    int64_t mem_97007_cached_sizze_98512 = 0;
    unsigned char *mem_97007 = NULL;
    int64_t mem_97012_cached_sizze_98513 = 0;
    unsigned char *mem_97012 = NULL;
    int64_t mem_97023_cached_sizze_98514 = 0;
    unsigned char *mem_97023 = NULL;
    int64_t mem_97024_cached_sizze_98515 = 0;
    unsigned char *mem_97024 = NULL;
    int64_t mem_97033_cached_sizze_98516 = 0;
    unsigned char *mem_97033 = NULL;
    int64_t mem_97034_cached_sizze_98517 = 0;
    unsigned char *mem_97034 = NULL;
    int64_t mem_97055_cached_sizze_98518 = 0;
    unsigned char *mem_97055 = NULL;
    int64_t mem_97056_cached_sizze_98519 = 0;
    unsigned char *mem_97056 = NULL;
    int64_t mem_97067_cached_sizze_98520 = 0;
    unsigned char *mem_97067 = NULL;
    int64_t mem_97068_cached_sizze_98521 = 0;
    unsigned char *mem_97068 = NULL;
    int64_t mem_97077_cached_sizze_98522 = 0;
    unsigned char *mem_97077 = NULL;
    int64_t mem_97084_cached_sizze_98523 = 0;
    unsigned char *mem_97084 = NULL;
    int64_t mem_97109_cached_sizze_98524 = 0;
    unsigned char *mem_97109 = NULL;
    int64_t mem_97110_cached_sizze_98525 = 0;
    unsigned char *mem_97110 = NULL;
    int64_t mem_97111_cached_sizze_98526 = 0;
    unsigned char *mem_97111 = NULL;
    int64_t mem_97126_cached_sizze_98527 = 0;
    unsigned char *mem_97126 = NULL;
    int64_t mem_97127_cached_sizze_98528 = 0;
    unsigned char *mem_97127 = NULL;
    int64_t mem_97128_cached_sizze_98529 = 0;
    unsigned char *mem_97128 = NULL;
    int64_t mem_97140_cached_sizze_98530 = 0;
    unsigned char *mem_97140 = NULL;
    int64_t mem_97147_cached_sizze_98531 = 0;
    unsigned char *mem_97147 = NULL;
    int64_t mem_97154_cached_sizze_98532 = 0;
    unsigned char *mem_97154 = NULL;
    int64_t mem_97186_cached_sizze_98533 = 0;
    unsigned char *mem_97186 = NULL;
    int64_t mem_97187_cached_sizze_98534 = 0;
    unsigned char *mem_97187 = NULL;
    int64_t mem_97198_cached_sizze_98535 = 0;
    unsigned char *mem_97198 = NULL;
    int64_t mem_97199_cached_sizze_98536 = 0;
    unsigned char *mem_97199 = NULL;
    int64_t mem_97208_cached_sizze_98537 = 0;
    unsigned char *mem_97208 = NULL;
    int64_t mem_97215_cached_sizze_98538 = 0;
    unsigned char *mem_97215 = NULL;
    int64_t mem_97240_cached_sizze_98539 = 0;
    unsigned char *mem_97240 = NULL;
    int64_t mem_97245_cached_sizze_98540 = 0;
    unsigned char *mem_97245 = NULL;
    int64_t mem_97256_cached_sizze_98541 = 0;
    unsigned char *mem_97256 = NULL;
    int64_t mem_97261_cached_sizze_98542 = 0;
    unsigned char *mem_97261 = NULL;
    int64_t mem_97272_cached_sizze_98543 = 0;
    unsigned char *mem_97272 = NULL;
    int64_t mem_97278_cached_sizze_98544 = 0;
    unsigned char *mem_97278 = NULL;
    int64_t mem_97283_cached_sizze_98545 = 0;
    unsigned char *mem_97283 = NULL;
    int64_t mem_97299_cached_sizze_98546 = 0;
    unsigned char *mem_97299 = NULL;
    int64_t mem_97304_cached_sizze_98547 = 0;
    unsigned char *mem_97304 = NULL;
    int64_t mem_97315_cached_sizze_98548 = 0;
    unsigned char *mem_97315 = NULL;
    int64_t mem_97321_cached_sizze_98549 = 0;
    unsigned char *mem_97321 = NULL;
    int64_t mem_97326_cached_sizze_98550 = 0;
    unsigned char *mem_97326 = NULL;
    int64_t mem_97342_cached_sizze_98551 = 0;
    unsigned char *mem_97342 = NULL;
    int64_t mem_97348_cached_sizze_98552 = 0;
    unsigned char *mem_97348 = NULL;
    int64_t mem_97353_cached_sizze_98553 = 0;
    unsigned char *mem_97353 = NULL;
    int64_t mem_97369_cached_sizze_98554 = 0;
    unsigned char *mem_97369 = NULL;
    int64_t mem_97370_cached_sizze_98555 = 0;
    unsigned char *mem_97370 = NULL;
    int64_t mem_97381_cached_sizze_98556 = 0;
    unsigned char *mem_97381 = NULL;
    int64_t mem_97382_cached_sizze_98557 = 0;
    unsigned char *mem_97382 = NULL;
    int64_t mem_97391_cached_sizze_98558 = 0;
    unsigned char *mem_97391 = NULL;
    int64_t mem_97392_cached_sizze_98559 = 0;
    unsigned char *mem_97392 = NULL;
    int64_t mem_97423_cached_sizze_98560 = 0;
    unsigned char *mem_97423 = NULL;
    int64_t mem_97424_cached_sizze_98561 = 0;
    unsigned char *mem_97424 = NULL;
    int64_t mem_97425_cached_sizze_98562 = 0;
    unsigned char *mem_97425 = NULL;
    int64_t mem_97438_cached_sizze_98563 = 0;
    unsigned char *mem_97438 = NULL;
    int64_t mem_97439_cached_sizze_98564 = 0;
    unsigned char *mem_97439 = NULL;
    int64_t mem_97440_cached_sizze_98565 = 0;
    unsigned char *mem_97440 = NULL;
    int64_t mem_97471_cached_sizze_98566 = 0;
    unsigned char *mem_97471 = NULL;
    int64_t mem_97472_cached_sizze_98567 = 0;
    unsigned char *mem_97472 = NULL;
    int64_t mem_97473_cached_sizze_98568 = 0;
    unsigned char *mem_97473 = NULL;
    int64_t mem_97474_cached_sizze_98569 = 0;
    unsigned char *mem_97474 = NULL;
    int64_t mem_97491_cached_sizze_98570 = 0;
    unsigned char *mem_97491 = NULL;
    int64_t mem_97492_cached_sizze_98571 = 0;
    unsigned char *mem_97492 = NULL;
    int64_t mem_97493_cached_sizze_98572 = 0;
    unsigned char *mem_97493 = NULL;
    int64_t mem_97494_cached_sizze_98573 = 0;
    unsigned char *mem_97494 = NULL;
    int64_t mem_97535_cached_sizze_98574 = 0;
    unsigned char *mem_97535 = NULL;
    int64_t mem_97542_cached_sizze_98575 = 0;
    unsigned char *mem_97542 = NULL;
    int64_t mem_97549_cached_sizze_98576 = 0;
    unsigned char *mem_97549 = NULL;
    int64_t mem_97559_cached_sizze_98577 = 0;
    unsigned char *mem_97559 = NULL;
    int64_t mem_97564_cached_sizze_98578 = 0;
    unsigned char *mem_97564 = NULL;
    int64_t mem_97575_cached_sizze_98579 = 0;
    unsigned char *mem_97575 = NULL;
    int64_t mem_97582_cached_sizze_98580 = 0;
    unsigned char *mem_97582 = NULL;
    int64_t mem_97589_cached_sizze_98581 = 0;
    unsigned char *mem_97589 = NULL;
    int64_t mem_97599_cached_sizze_98582 = 0;
    unsigned char *mem_97599 = NULL;
    int64_t mem_97604_cached_sizze_98583 = 0;
    unsigned char *mem_97604 = NULL;
    int64_t mem_97615_cached_sizze_98584 = 0;
    unsigned char *mem_97615 = NULL;
    int64_t mem_97616_cached_sizze_98585 = 0;
    unsigned char *mem_97616 = NULL;
    int64_t mem_97625_cached_sizze_98586 = 0;
    unsigned char *mem_97625 = NULL;
    int64_t mem_97626_cached_sizze_98587 = 0;
    unsigned char *mem_97626 = NULL;
    int64_t mem_97647_cached_sizze_98588 = 0;
    unsigned char *mem_97647 = NULL;
    int64_t mem_97652_cached_sizze_98589 = 0;
    unsigned char *mem_97652 = NULL;
    int64_t mem_97663_cached_sizze_98590 = 0;
    unsigned char *mem_97663 = NULL;
    int64_t mem_97664_cached_sizze_98591 = 0;
    unsigned char *mem_97664 = NULL;
    int64_t mem_97673_cached_sizze_98592 = 0;
    unsigned char *mem_97673 = NULL;
    int64_t mem_97674_cached_sizze_98593 = 0;
    unsigned char *mem_97674 = NULL;
    struct memblock mem_param_tmp_98027;
    
    mem_param_tmp_98027.references = NULL;
    
    struct memblock mem_param_tmp_98026;
    
    mem_param_tmp_98026.references = NULL;
    
    struct memblock mem_param_tmp_98025;
    
    mem_param_tmp_98025.references = NULL;
    
    struct memblock mem_param_tmp_98024;
    
    mem_param_tmp_98024.references = NULL;
    
    struct memblock mem_param_tmp_98023;
    
    mem_param_tmp_98023.references = NULL;
    
    struct memblock mem_param_tmp_98022;
    
    mem_param_tmp_98022.references = NULL;
    
    struct memblock mem_param_tmp_98021;
    
    mem_param_tmp_98021.references = NULL;
    
    struct memblock mem_param_tmp_98020;
    
    mem_param_tmp_98020.references = NULL;
    
    struct memblock mem_param_tmp_98019;
    
    mem_param_tmp_98019.references = NULL;
    
    struct memblock mem_param_tmp_98018;
    
    mem_param_tmp_98018.references = NULL;
    
    struct memblock mem_param_tmp_98017;
    
    mem_param_tmp_98017.references = NULL;
    
    struct memblock mem_param_tmp_98016;
    
    mem_param_tmp_98016.references = NULL;
    
    struct memblock mem_param_tmp_98015;
    
    mem_param_tmp_98015.references = NULL;
    
    struct memblock mem_param_tmp_98014;
    
    mem_param_tmp_98014.references = NULL;
    
    struct memblock mem_param_tmp_98013;
    
    mem_param_tmp_98013.references = NULL;
    
    struct memblock mem_param_tmp_98012;
    
    mem_param_tmp_98012.references = NULL;
    
    struct memblock mem_param_tmp_98011;
    
    mem_param_tmp_98011.references = NULL;
    
    struct memblock mem_param_tmp_98010;
    
    mem_param_tmp_98010.references = NULL;
    
    struct memblock mem_param_tmp_98009;
    
    mem_param_tmp_98009.references = NULL;
    
    struct memblock mem_param_tmp_98008;
    
    mem_param_tmp_98008.references = NULL;
    
    struct memblock mem_param_tmp_98007;
    
    mem_param_tmp_98007.references = NULL;
    
    struct memblock mem_param_tmp_98006;
    
    mem_param_tmp_98006.references = NULL;
    
    struct memblock mem_param_tmp_98005;
    
    mem_param_tmp_98005.references = NULL;
    
    struct memblock mem_param_tmp_98004;
    
    mem_param_tmp_98004.references = NULL;
    
    struct memblock mem_param_tmp_98003;
    
    mem_param_tmp_98003.references = NULL;
    
    struct memblock mem_param_tmp_98002;
    
    mem_param_tmp_98002.references = NULL;
    
    struct memblock mem_param_tmp_98001;
    
    mem_param_tmp_98001.references = NULL;
    
    struct memblock ext_mem_97791;
    
    ext_mem_97791.references = NULL;
    
    struct memblock ext_mem_97792;
    
    ext_mem_97792.references = NULL;
    
    struct memblock ext_mem_97793;
    
    ext_mem_97793.references = NULL;
    
    struct memblock mem_97789;
    
    mem_97789.references = NULL;
    
    struct memblock mem_97787;
    
    mem_97787.references = NULL;
    
    struct memblock mem_97785;
    
    mem_97785.references = NULL;
    
    struct memblock mem_97783;
    
    mem_97783.references = NULL;
    
    struct memblock ext_mem_97780;
    
    ext_mem_97780.references = NULL;
    
    struct memblock ext_mem_97781;
    
    ext_mem_97781.references = NULL;
    
    struct memblock ext_mem_97782;
    
    ext_mem_97782.references = NULL;
    
    struct memblock mem_97778;
    
    mem_97778.references = NULL;
    
    struct memblock mem_97776;
    
    mem_97776.references = NULL;
    
    struct memblock mem_97774;
    
    mem_97774.references = NULL;
    
    struct memblock mem_97772;
    
    mem_97772.references = NULL;
    
    struct memblock ext_mem_97769;
    
    ext_mem_97769.references = NULL;
    
    struct memblock ext_mem_97770;
    
    ext_mem_97770.references = NULL;
    
    struct memblock ext_mem_97771;
    
    ext_mem_97771.references = NULL;
    
    struct memblock mem_97767;
    
    mem_97767.references = NULL;
    
    struct memblock mem_97765;
    
    mem_97765.references = NULL;
    
    struct memblock mem_97763;
    
    mem_97763.references = NULL;
    
    struct memblock mem_97761;
    
    mem_97761.references = NULL;
    
    struct memblock ext_mem_97758;
    
    ext_mem_97758.references = NULL;
    
    struct memblock ext_mem_97759;
    
    ext_mem_97759.references = NULL;
    
    struct memblock ext_mem_97760;
    
    ext_mem_97760.references = NULL;
    
    struct memblock mem_97756;
    
    mem_97756.references = NULL;
    
    struct memblock mem_97754;
    
    mem_97754.references = NULL;
    
    struct memblock mem_97752;
    
    mem_97752.references = NULL;
    
    struct memblock mem_97750;
    
    mem_97750.references = NULL;
    
    struct memblock ext_mem_97747;
    
    ext_mem_97747.references = NULL;
    
    struct memblock ext_mem_97748;
    
    ext_mem_97748.references = NULL;
    
    struct memblock ext_mem_97749;
    
    ext_mem_97749.references = NULL;
    
    struct memblock mem_97745;
    
    mem_97745.references = NULL;
    
    struct memblock mem_97743;
    
    mem_97743.references = NULL;
    
    struct memblock mem_97741;
    
    mem_97741.references = NULL;
    
    struct memblock mem_97739;
    
    mem_97739.references = NULL;
    
    struct memblock ext_mem_97736;
    
    ext_mem_97736.references = NULL;
    
    struct memblock ext_mem_97737;
    
    ext_mem_97737.references = NULL;
    
    struct memblock ext_mem_97738;
    
    ext_mem_97738.references = NULL;
    
    struct memblock mem_97734;
    
    mem_97734.references = NULL;
    
    struct memblock mem_97732;
    
    mem_97732.references = NULL;
    
    struct memblock mem_97730;
    
    mem_97730.references = NULL;
    
    struct memblock mem_97728;
    
    mem_97728.references = NULL;
    
    struct memblock ext_mem_97725;
    
    ext_mem_97725.references = NULL;
    
    struct memblock ext_mem_97726;
    
    ext_mem_97726.references = NULL;
    
    struct memblock ext_mem_97727;
    
    ext_mem_97727.references = NULL;
    
    struct memblock mem_97723;
    
    mem_97723.references = NULL;
    
    struct memblock mem_97721;
    
    mem_97721.references = NULL;
    
    struct memblock mem_97719;
    
    mem_97719.references = NULL;
    
    struct memblock mem_97717;
    
    mem_97717.references = NULL;
    
    struct memblock ext_mem_97714;
    
    ext_mem_97714.references = NULL;
    
    struct memblock ext_mem_97715;
    
    ext_mem_97715.references = NULL;
    
    struct memblock ext_mem_97716;
    
    ext_mem_97716.references = NULL;
    
    struct memblock mem_97712;
    
    mem_97712.references = NULL;
    
    struct memblock mem_97710;
    
    mem_97710.references = NULL;
    
    struct memblock mem_97708;
    
    mem_97708.references = NULL;
    
    struct memblock mem_97706;
    
    mem_97706.references = NULL;
    
    struct memblock ext_mem_97703;
    
    ext_mem_97703.references = NULL;
    
    struct memblock ext_mem_97704;
    
    ext_mem_97704.references = NULL;
    
    struct memblock ext_mem_97705;
    
    ext_mem_97705.references = NULL;
    
    struct memblock mem_97701;
    
    mem_97701.references = NULL;
    
    struct memblock mem_97699;
    
    mem_97699.references = NULL;
    
    struct memblock mem_97697;
    
    mem_97697.references = NULL;
    
    struct memblock mem_97695;
    
    mem_97695.references = NULL;
    
    struct memblock mem_param_96244;
    
    mem_param_96244.references = NULL;
    
    struct memblock mem_param_96240;
    
    mem_param_96240.references = NULL;
    
    struct memblock mem_param_96236;
    
    mem_param_96236.references = NULL;
    
    struct memblock mem_param_96232;
    
    mem_param_96232.references = NULL;
    
    struct memblock mem_param_96228;
    
    mem_param_96228.references = NULL;
    
    struct memblock mem_param_96224;
    
    mem_param_96224.references = NULL;
    
    struct memblock mem_param_96220;
    
    mem_param_96220.references = NULL;
    
    struct memblock mem_param_96216;
    
    mem_param_96216.references = NULL;
    
    struct memblock mem_param_96212;
    
    mem_param_96212.references = NULL;
    
    struct memblock mem_param_96208;
    
    mem_param_96208.references = NULL;
    
    struct memblock mem_param_96204;
    
    mem_param_96204.references = NULL;
    
    struct memblock mem_param_96200;
    
    mem_param_96200.references = NULL;
    
    struct memblock mem_param_96196;
    
    mem_param_96196.references = NULL;
    
    struct memblock mem_param_96192;
    
    mem_param_96192.references = NULL;
    
    struct memblock mem_param_96188;
    
    mem_param_96188.references = NULL;
    
    struct memblock mem_param_96184;
    
    mem_param_96184.references = NULL;
    
    struct memblock mem_param_96180;
    
    mem_param_96180.references = NULL;
    
    struct memblock mem_param_96176;
    
    mem_param_96176.references = NULL;
    
    struct memblock mem_param_96172;
    
    mem_param_96172.references = NULL;
    
    struct memblock mem_param_96168;
    
    mem_param_96168.references = NULL;
    
    struct memblock mem_param_96164;
    
    mem_param_96164.references = NULL;
    
    struct memblock mem_param_96160;
    
    mem_param_96160.references = NULL;
    
    struct memblock mem_param_96156;
    
    mem_param_96156.references = NULL;
    
    struct memblock mem_param_96152;
    
    mem_param_96152.references = NULL;
    
    struct memblock mem_param_96148;
    
    mem_param_96148.references = NULL;
    
    struct memblock mem_param_96144;
    
    mem_param_96144.references = NULL;
    
    struct memblock mem_param_96140;
    
    mem_param_96140.references = NULL;
    
    struct memblock ext_mem_97875;
    
    ext_mem_97875.references = NULL;
    
    struct memblock ext_mem_97876;
    
    ext_mem_97876.references = NULL;
    
    struct memblock ext_mem_97877;
    
    ext_mem_97877.references = NULL;
    
    struct memblock ext_mem_97878;
    
    ext_mem_97878.references = NULL;
    
    struct memblock ext_mem_97879;
    
    ext_mem_97879.references = NULL;
    
    struct memblock ext_mem_97880;
    
    ext_mem_97880.references = NULL;
    
    struct memblock ext_mem_97881;
    
    ext_mem_97881.references = NULL;
    
    struct memblock ext_mem_97882;
    
    ext_mem_97882.references = NULL;
    
    struct memblock ext_mem_97883;
    
    ext_mem_97883.references = NULL;
    
    struct memblock ext_mem_97884;
    
    ext_mem_97884.references = NULL;
    
    struct memblock ext_mem_97885;
    
    ext_mem_97885.references = NULL;
    
    struct memblock ext_mem_97886;
    
    ext_mem_97886.references = NULL;
    
    struct memblock ext_mem_97887;
    
    ext_mem_97887.references = NULL;
    
    struct memblock ext_mem_97888;
    
    ext_mem_97888.references = NULL;
    
    struct memblock ext_mem_97889;
    
    ext_mem_97889.references = NULL;
    
    struct memblock ext_mem_97890;
    
    ext_mem_97890.references = NULL;
    
    struct memblock ext_mem_97891;
    
    ext_mem_97891.references = NULL;
    
    struct memblock ext_mem_97892;
    
    ext_mem_97892.references = NULL;
    
    struct memblock ext_mem_97893;
    
    ext_mem_97893.references = NULL;
    
    struct memblock ext_mem_97894;
    
    ext_mem_97894.references = NULL;
    
    struct memblock ext_mem_97895;
    
    ext_mem_97895.references = NULL;
    
    struct memblock ext_mem_97896;
    
    ext_mem_97896.references = NULL;
    
    struct memblock ext_mem_97897;
    
    ext_mem_97897.references = NULL;
    
    struct memblock ext_mem_97898;
    
    ext_mem_97898.references = NULL;
    
    struct memblock ext_mem_97899;
    
    ext_mem_97899.references = NULL;
    
    struct memblock ext_mem_97900;
    
    ext_mem_97900.references = NULL;
    
    struct memblock ext_mem_97901;
    
    ext_mem_97901.references = NULL;
    
    struct memblock mem_out_98000;
    
    mem_out_98000.references = NULL;
    
    struct memblock mem_out_97999;
    
    mem_out_97999.references = NULL;
    
    struct memblock mem_out_97998;
    
    mem_out_97998.references = NULL;
    
    struct memblock mem_out_97997;
    
    mem_out_97997.references = NULL;
    
    struct memblock mem_out_97996;
    
    mem_out_97996.references = NULL;
    
    struct memblock mem_out_97995;
    
    mem_out_97995.references = NULL;
    
    struct memblock mem_out_97994;
    
    mem_out_97994.references = NULL;
    
    struct memblock mem_out_97993;
    
    mem_out_97993.references = NULL;
    
    struct memblock mem_out_97992;
    
    mem_out_97992.references = NULL;
    
    struct memblock mem_out_97991;
    
    mem_out_97991.references = NULL;
    
    struct memblock mem_out_97990;
    
    mem_out_97990.references = NULL;
    
    struct memblock mem_out_97989;
    
    mem_out_97989.references = NULL;
    
    struct memblock mem_out_97988;
    
    mem_out_97988.references = NULL;
    
    struct memblock mem_out_97987;
    
    mem_out_97987.references = NULL;
    
    struct memblock mem_out_97986;
    
    mem_out_97986.references = NULL;
    
    struct memblock mem_out_97985;
    
    mem_out_97985.references = NULL;
    
    struct memblock mem_out_97984;
    
    mem_out_97984.references = NULL;
    
    struct memblock mem_out_97983;
    
    mem_out_97983.references = NULL;
    
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_96245_cached_sizze_98417 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96245, &mem_96245_cached_sizze_98417, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96246_cached_sizze_98418 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96246, &mem_96246_cached_sizze_98418, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96255_cached_sizze_98419 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96255, &mem_96255_cached_sizze_98419, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96262_cached_sizze_98420 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96262, &mem_96262_cached_sizze_98420, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96277_cached_sizze_98421 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96277, &mem_96277_cached_sizze_98421, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96278_cached_sizze_98422 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96278, &mem_96278_cached_sizze_98422, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96287_cached_sizze_98423 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96287, &mem_96287_cached_sizze_98423, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96294_cached_sizze_98424 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96294, &mem_96294_cached_sizze_98424, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96309_cached_sizze_98425 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96309, &mem_96309_cached_sizze_98425, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96310_cached_sizze_98426 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96310, &mem_96310_cached_sizze_98426, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96319_cached_sizze_98427 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96319, &mem_96319_cached_sizze_98427, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96320_cached_sizze_98428 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96320, &mem_96320_cached_sizze_98428, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96341_cached_sizze_98429 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96341, &mem_96341_cached_sizze_98429, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96342_cached_sizze_98430 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96342, &mem_96342_cached_sizze_98430, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96343_cached_sizze_98431 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96343, &mem_96343_cached_sizze_98431, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96355_cached_sizze_98432 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96355, &mem_96355_cached_sizze_98432, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96356_cached_sizze_98433 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96356, &mem_96356_cached_sizze_98433, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96380_cached_sizze_98434 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96380, &mem_96380_cached_sizze_98434, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96381_cached_sizze_98435 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96381, &mem_96381_cached_sizze_98435, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96382_cached_sizze_98436 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96382, &mem_96382_cached_sizze_98436, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96383_cached_sizze_98437 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96383, &mem_96383_cached_sizze_98437, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96384_cached_sizze_98438 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96384, &mem_96384_cached_sizze_98438, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96403_cached_sizze_98439 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96403, &mem_96403_cached_sizze_98439, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96404_cached_sizze_98440 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96404, &mem_96404_cached_sizze_98440, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96405_cached_sizze_98441 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96405, &mem_96405_cached_sizze_98441, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96442_cached_sizze_98442 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96442, &mem_96442_cached_sizze_98442, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96443_cached_sizze_98443 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96443, &mem_96443_cached_sizze_98443, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96444_cached_sizze_98444 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96444, &mem_96444_cached_sizze_98444, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96460_cached_sizze_98445 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96460, &mem_96460_cached_sizze_98445, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96461_cached_sizze_98446 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96461, &mem_96461_cached_sizze_98446, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96462_cached_sizze_98447 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96462, &mem_96462_cached_sizze_98447, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96475_cached_sizze_98448 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96475, &mem_96475_cached_sizze_98448, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96476_cached_sizze_98449 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96476, &mem_96476_cached_sizze_98449, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96477_cached_sizze_98450 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96477, &mem_96477_cached_sizze_98450, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96523_cached_sizze_98451 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96523, &mem_96523_cached_sizze_98451, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96524_cached_sizze_98452 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96524, &mem_96524_cached_sizze_98452, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96535_cached_sizze_98453 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96535, &mem_96535_cached_sizze_98453, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96536_cached_sizze_98454 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96536, &mem_96536_cached_sizze_98454, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96545_cached_sizze_98455 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96545, &mem_96545_cached_sizze_98455, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96546_cached_sizze_98456 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96546, &mem_96546_cached_sizze_98456, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96567_cached_sizze_98457 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96567, &mem_96567_cached_sizze_98457, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96572_cached_sizze_98458 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96572, &mem_96572_cached_sizze_98458, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96583_cached_sizze_98459 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96583, &mem_96583_cached_sizze_98459, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96588_cached_sizze_98460 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96588, &mem_96588_cached_sizze_98460, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96595_cached_sizze_98461 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96595, &mem_96595_cached_sizze_98461, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96606_cached_sizze_98462 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96606, &mem_96606_cached_sizze_98462, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96611_cached_sizze_98463 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_96611, &mem_96611_cached_sizze_98463, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96632_cached_sizze_98464 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96632, &mem_96632_cached_sizze_98464, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96633_cached_sizze_98465 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96633, &mem_96633_cached_sizze_98465, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96641_cached_sizze_98466 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96641, &mem_96641_cached_sizze_98466, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96655_cached_sizze_98467 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96655, &mem_96655_cached_sizze_98467, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96660_cached_sizze_98468 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96660, &mem_96660_cached_sizze_98468, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96671_cached_sizze_98469 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96671, &mem_96671_cached_sizze_98469, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96676_cached_sizze_98470 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96676, &mem_96676_cached_sizze_98470, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96687_cached_sizze_98471 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96687, &mem_96687_cached_sizze_98471, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96688_cached_sizze_98472 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96688, &mem_96688_cached_sizze_98472, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96697_cached_sizze_98473 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96697, &mem_96697_cached_sizze_98473, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96698_cached_sizze_98474 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96698, &mem_96698_cached_sizze_98474, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96719_cached_sizze_98475 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96719, &mem_96719_cached_sizze_98475, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96720_cached_sizze_98476 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96720, &mem_96720_cached_sizze_98476, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96728_cached_sizze_98477 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96728, &mem_96728_cached_sizze_98477, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96742_cached_sizze_98478 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96742, &mem_96742_cached_sizze_98478, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96743_cached_sizze_98479 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96743, &mem_96743_cached_sizze_98479, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96751_cached_sizze_98480 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96751, &mem_96751_cached_sizze_98480, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96765_cached_sizze_98481 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96765, &mem_96765_cached_sizze_98481, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96770_cached_sizze_98482 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96770, &mem_96770_cached_sizze_98482, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96781_cached_sizze_98483 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96781, &mem_96781_cached_sizze_98483, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96786_cached_sizze_98484 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96786, &mem_96786_cached_sizze_98484, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96797_cached_sizze_98485 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96797, &mem_96797_cached_sizze_98485, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96802_cached_sizze_98486 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96802, &mem_96802_cached_sizze_98486, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96813_cached_sizze_98487 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96813, &mem_96813_cached_sizze_98487, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96820_cached_sizze_98488 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96820, &mem_96820_cached_sizze_98488, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96825_cached_sizze_98489 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96825, &mem_96825_cached_sizze_98489, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96836_cached_sizze_98490 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96836, &mem_96836_cached_sizze_98490, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96843_cached_sizze_98491 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96843, &mem_96843_cached_sizze_98491, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96847_cached_sizze_98492 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96847, &mem_96847_cached_sizze_98492, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96857_cached_sizze_98493 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96857, &mem_96857_cached_sizze_98493, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96862_cached_sizze_98494 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96862, &mem_96862_cached_sizze_98494, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96869_cached_sizze_98495 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96869, &mem_96869_cached_sizze_98495, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96880_cached_sizze_98496 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96880, &mem_96880_cached_sizze_98496, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96887_cached_sizze_98497 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_96887, &mem_96887_cached_sizze_98497, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96892_cached_sizze_98498 < (int64_t) 216) {
        err = lexical_realloc(ctx, &mem_96892, &mem_96892_cached_sizze_98498, (int64_t) 216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96903_cached_sizze_98499 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96903, &mem_96903_cached_sizze_98499, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96908_cached_sizze_98500 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96908, &mem_96908_cached_sizze_98500, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96919_cached_sizze_98501 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96919, &mem_96919_cached_sizze_98501, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96920_cached_sizze_98502 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96920, &mem_96920_cached_sizze_98502, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96929_cached_sizze_98503 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96929, &mem_96929_cached_sizze_98503, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96930_cached_sizze_98504 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96930, &mem_96930_cached_sizze_98504, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96951_cached_sizze_98505 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_96951, &mem_96951_cached_sizze_98505, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96956_cached_sizze_98506 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_96956, &mem_96956_cached_sizze_98506, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96967_cached_sizze_98507 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96967, &mem_96967_cached_sizze_98507, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96972_cached_sizze_98508 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96972, &mem_96972_cached_sizze_98508, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96983_cached_sizze_98509 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96983, &mem_96983_cached_sizze_98509, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96990_cached_sizze_98510 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_96990, &mem_96990_cached_sizze_98510, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_96997_cached_sizze_98511 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_96997, &mem_96997_cached_sizze_98511, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97007_cached_sizze_98512 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97007, &mem_97007_cached_sizze_98512, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97012_cached_sizze_98513 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97012, &mem_97012_cached_sizze_98513, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97023_cached_sizze_98514 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97023, &mem_97023_cached_sizze_98514, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97024_cached_sizze_98515 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97024, &mem_97024_cached_sizze_98515, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97033_cached_sizze_98516 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97033, &mem_97033_cached_sizze_98516, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97034_cached_sizze_98517 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97034, &mem_97034_cached_sizze_98517, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97055_cached_sizze_98518 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97055, &mem_97055_cached_sizze_98518, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97056_cached_sizze_98519 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97056, &mem_97056_cached_sizze_98519, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97067_cached_sizze_98520 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97067, &mem_97067_cached_sizze_98520, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97068_cached_sizze_98521 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97068, &mem_97068_cached_sizze_98521, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97077_cached_sizze_98522 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_97077, &mem_97077_cached_sizze_98522, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97084_cached_sizze_98523 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97084, &mem_97084_cached_sizze_98523, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97109_cached_sizze_98524 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97109, &mem_97109_cached_sizze_98524, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97110_cached_sizze_98525 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97110, &mem_97110_cached_sizze_98525, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97111_cached_sizze_98526 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97111, &mem_97111_cached_sizze_98526, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97126_cached_sizze_98527 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97126, &mem_97126_cached_sizze_98527, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97127_cached_sizze_98528 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97127, &mem_97127_cached_sizze_98528, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97128_cached_sizze_98529 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97128, &mem_97128_cached_sizze_98529, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:115:5-117:48
    if (mem_97140_cached_sizze_98530 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97140, &mem_97140_cached_sizze_98530, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97147_cached_sizze_98531 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97147, &mem_97147_cached_sizze_98531, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97154_cached_sizze_98532 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97154, &mem_97154_cached_sizze_98532, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97186_cached_sizze_98533 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97186, &mem_97186_cached_sizze_98533, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97187_cached_sizze_98534 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97187, &mem_97187_cached_sizze_98534, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97198_cached_sizze_98535 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97198, &mem_97198_cached_sizze_98535, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97199_cached_sizze_98536 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97199, &mem_97199_cached_sizze_98536, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97208_cached_sizze_98537 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97208, &mem_97208_cached_sizze_98537, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97215_cached_sizze_98538 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_97215, &mem_97215_cached_sizze_98538, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97240_cached_sizze_98539 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97240, &mem_97240_cached_sizze_98539, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97245_cached_sizze_98540 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97245, &mem_97245_cached_sizze_98540, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97256_cached_sizze_98541 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97256, &mem_97256_cached_sizze_98541, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97261_cached_sizze_98542 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97261, &mem_97261_cached_sizze_98542, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97272_cached_sizze_98543 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97272, &mem_97272_cached_sizze_98543, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97278_cached_sizze_98544 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97278, &mem_97278_cached_sizze_98544, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97283_cached_sizze_98545 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97283, &mem_97283_cached_sizze_98545, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97299_cached_sizze_98546 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97299, &mem_97299_cached_sizze_98546, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97304_cached_sizze_98547 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97304, &mem_97304_cached_sizze_98547, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97315_cached_sizze_98548 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97315, &mem_97315_cached_sizze_98548, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97321_cached_sizze_98549 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97321, &mem_97321_cached_sizze_98549, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97326_cached_sizze_98550 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97326, &mem_97326_cached_sizze_98550, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97342_cached_sizze_98551 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97342, &mem_97342_cached_sizze_98551, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97348_cached_sizze_98552 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97348, &mem_97348_cached_sizze_98552, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97353_cached_sizze_98553 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97353, &mem_97353_cached_sizze_98553, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97369_cached_sizze_98554 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97369, &mem_97369_cached_sizze_98554, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97370_cached_sizze_98555 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97370, &mem_97370_cached_sizze_98555, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97381_cached_sizze_98556 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97381, &mem_97381_cached_sizze_98556, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97382_cached_sizze_98557 < (int64_t) 512) {
        err = lexical_realloc(ctx, &mem_97382, &mem_97382_cached_sizze_98557, (int64_t) 512);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97391_cached_sizze_98558 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_97391, &mem_97391_cached_sizze_98558, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97392_cached_sizze_98559 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_97392, &mem_97392_cached_sizze_98559, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97423_cached_sizze_98560 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97423, &mem_97423_cached_sizze_98560, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97424_cached_sizze_98561 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97424, &mem_97424_cached_sizze_98561, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97425_cached_sizze_98562 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97425, &mem_97425_cached_sizze_98562, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97438_cached_sizze_98563 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97438, &mem_97438_cached_sizze_98563, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97439_cached_sizze_98564 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97439, &mem_97439_cached_sizze_98564, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97440_cached_sizze_98565 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97440, &mem_97440_cached_sizze_98565, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97471_cached_sizze_98566 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97471, &mem_97471_cached_sizze_98566, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97472_cached_sizze_98567 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97472, &mem_97472_cached_sizze_98567, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97473_cached_sizze_98568 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97473, &mem_97473_cached_sizze_98568, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97474_cached_sizze_98569 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97474, &mem_97474_cached_sizze_98569, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97491_cached_sizze_98570 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97491, &mem_97491_cached_sizze_98570, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97492_cached_sizze_98571 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97492, &mem_97492_cached_sizze_98571, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97493_cached_sizze_98572 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97493, &mem_97493_cached_sizze_98572, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97494_cached_sizze_98573 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97494, &mem_97494_cached_sizze_98573, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97535_cached_sizze_98574 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97535, &mem_97535_cached_sizze_98574, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97542_cached_sizze_98575 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97542, &mem_97542_cached_sizze_98575, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97549_cached_sizze_98576 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97549, &mem_97549_cached_sizze_98576, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97559_cached_sizze_98577 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97559, &mem_97559_cached_sizze_98577, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97564_cached_sizze_98578 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97564, &mem_97564_cached_sizze_98578, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97575_cached_sizze_98579 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97575, &mem_97575_cached_sizze_98579, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97582_cached_sizze_98580 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97582, &mem_97582_cached_sizze_98580, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97589_cached_sizze_98581 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97589, &mem_97589_cached_sizze_98581, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97599_cached_sizze_98582 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97599, &mem_97599_cached_sizze_98582, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97604_cached_sizze_98583 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97604, &mem_97604_cached_sizze_98583, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97615_cached_sizze_98584 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97615, &mem_97615_cached_sizze_98584, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97616_cached_sizze_98585 < (int64_t) 2048) {
        err = lexical_realloc(ctx, &mem_97616, &mem_97616_cached_sizze_98585, (int64_t) 2048);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97625_cached_sizze_98586 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97625, &mem_97625_cached_sizze_98586, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97626_cached_sizze_98587 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97626, &mem_97626_cached_sizze_98587, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97647_cached_sizze_98588 < (int64_t) 8192) {
        err = lexical_realloc(ctx, &mem_97647, &mem_97647_cached_sizze_98588, (int64_t) 8192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97652_cached_sizze_98589 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97652, &mem_97652_cached_sizze_98589, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97663_cached_sizze_98590 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_97663, &mem_97663_cached_sizze_98590, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97664_cached_sizze_98591 < (int64_t) 3456) {
        err = lexical_realloc(ctx, &mem_97664, &mem_97664_cached_sizze_98591, (int64_t) 3456);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97673_cached_sizze_98592 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97673, &mem_97673_cached_sizze_98592, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_97674_cached_sizze_98593 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_97674, &mem_97674_cached_sizze_98593, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:491:5-496:51
    if (memblock_set(ctx, &mem_param_96140, &wdown_mem_96107, "wdown_mem_96107") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96144, &wkey_mem_96108, "wkey_mem_96108") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96148, &wout_mem_96109, "wout_mem_96109") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96152, &wpe_mem_96110, "wpe_mem_96110") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96156, &wqry_mem_96111, "wqry_mem_96111") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96160, &wte_mem_96112, "wte_mem_96112") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96164, &wup_mem_96113, "wup_mem_96113") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96168, &wval_mem_96114, "wval_mem_96114") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96172, &wvoc_mem_96115, "wvoc_mem_96115") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96176, &wdown_mem_96116, "wdown_mem_96116") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96180, &wkey_mem_96117, "wkey_mem_96117") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96184, &wout_mem_96118, "wout_mem_96118") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96188, &wpe_mem_96119, "wpe_mem_96119") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96192, &wqry_mem_96120, "wqry_mem_96120") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96196, &wte_mem_96121, "wte_mem_96121") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96200, &wup_mem_96122, "wup_mem_96122") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96204, &wval_mem_96123, "wval_mem_96123") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96208, &wvoc_mem_96124, "wvoc_mem_96124") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96212, &wdown_mem_96125, "wdown_mem_96125") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96216, &wkey_mem_96126, "wkey_mem_96126") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96220, &wout_mem_96127, "wout_mem_96127") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96224, &wpe_mem_96128, "wpe_mem_96128") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96228, &wqry_mem_96129, "wqry_mem_96129") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96232, &wte_mem_96130, "wte_mem_96130") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96236, &wup_mem_96131, "wup_mem_96131") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96240, &wval_mem_96132, "wval_mem_96132") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_96244, &wvoc_mem_96133, "wvoc_mem_96133") != 0)
        return 1;
    for (int64_t step_88717 = 0; step_88717 < (int64_t) 500; step_88717++) {
        // futhark/microgpt.fut:493:16-25
        
        int64_t dl_88745 = ((int64_t *) dls_mem_96135.mem)[step_88717];
        
        // futhark/microgpt.fut:406:37-40
        
        int64_t zl_rhs_88750 = sub64(dl_88745, (int64_t) 1);
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95182 = 0; i_95182 < (int64_t) 16; i_95182++) {
            // futhark/microgpt.fut:406:25-81
            
            bool cond_90839 = slt64(i_95182, zl_rhs_88750);
            
            // futhark/microgpt.fut:406:56-59
            
            int64_t zeze_lhs_90840 = add64((int64_t) 1, i_95182);
            
            // futhark/microgpt.fut:406:47-60
            
            bool x_90841 = sle64((int64_t) 0, zeze_lhs_90840);
            
            // futhark/microgpt.fut:406:47-60
            
            bool y_90842 = slt64(zeze_lhs_90840, (int64_t) 16);
            
            // futhark/microgpt.fut:406:47-60
            
            bool bounds_check_90843 = x_90841 && y_90842;
            
            // futhark/microgpt.fut:9:27-39
            
            bool loop_not_taken_90844 = !cond_90839;
            
            // futhark/microgpt.fut:9:27-39
            
            bool protect_assert_disj_90845 = bounds_check_90843 || loop_not_taken_90844;
            
            // futhark/microgpt.fut:406:47-60
            
            bool index_certs_90846;
            
            if (!protect_assert_disj_90845) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_90840, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:406:47-60\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:406:3-83\n   #6  futhark/microgpt.fut:464:18-38\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            int64_t tmp_90861 = ((int64_t *) seqs_mem_96136.mem)[step_88717 * (int64_t) 16 + i_95182];
            
            // futhark/microgpt.fut:466:37-51
            
            bool x_90862 = sle64((int64_t) 0, tmp_90861);
            
            // futhark/microgpt.fut:466:37-51
            
            bool y_90863 = slt64(tmp_90861, (int64_t) 27);
            
            // futhark/microgpt.fut:466:37-51
            
            bool bounds_check_90864 = x_90862 && y_90863;
            
            // futhark/microgpt.fut:466:37-51
            
            bool index_certs_90865;
            
            if (!bounds_check_90864) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_90861, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:466:37-51\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:466:16-55\n   #6  futhark/microgpt.fut:474:26-480:31\n   #7  futhark/microgpt.fut:496:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:406:47-60
            
            int64_t zeze_lhs_90847;
            
            if (cond_90839) {
                int64_t x_94956 = ((int64_t *) seqs_mem_96136.mem)[step_88717 * (int64_t) 16 + zeze_lhs_90840];
                
                zeze_lhs_90847 = x_94956;
            } else {
                zeze_lhs_90847 = (int64_t) 0;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95172 = 0; i_95172 < (int64_t) 27; i_95172++) {
                // futhark/microgpt.fut:406:61-65
                
                bool cond_t_res_90851 = zeze_lhs_90847 == i_95172;
                
                // futhark/microgpt.fut:9:27-39
                
                bool x_90852 = cond_90839 && cond_t_res_90851;
                
                // futhark/microgpt.fut:406:25-81
                
                double lifted_lambda_res_90853;
                
                if (x_90852) {
                    lifted_lambda_res_90853 = 1.0;
                } else {
                    lifted_lambda_res_90853 = 0.0;
                }
                ((double *) mem_96255)[i_95172] = lifted_lambda_res_90853;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95176 = 0; i_95176 < (int64_t) 16; i_95176++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_90872 = ((double *) mem_param_96160.mem)[tmp_90861 * (int64_t) 16 + i_95176];
                
                ((double *) mem_96262)[i_95176] = lifted_lambda_res_90872;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96245, i_95182 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96262, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96246, i_95182 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96255, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95197 = 0; i_95197 < (int64_t) 16; i_95197++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95187 = 0; i_95187 < (int64_t) 16; i_95187++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90897 = ((double *) mem_param_96152.mem)[i_95197 * (int64_t) 16 + i_95187];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_90898 = ((double *) mem_96245)[i_95197 * (int64_t) 16 + i_95187];
                
                // futhark/microgpt.fut:239:41-77
                
                double zp_res_90899 = zp_lhs_90897 + zp_rhs_90898;
                
                ((double *) mem_96287)[i_95187] = zp_res_90899;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95191 = 0; i_95191 < (int64_t) 27; i_95191++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90913 = ((double *) mem_96246)[i_95197 * (int64_t) 27 + i_95191];
                
                // futhark/microgpt.fut:275:43-85
                
                double zt_res_90914 = -6.25e-2 * zt_rhs_90913;
                
                ((double *) mem_96294)[i_95191] = zt_res_90914;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96277, i_95197 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96294, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96278, i_95197 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96287, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95211 = 0; i_95211 < (int64_t) 16; i_95211++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90933;
            double r_90935 = 0.0;
            
            for (int64_t i_90934 = 0; i_90934 < (int64_t) 16; i_90934++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_90936 = ((double *) mem_96278)[i_95211 * (int64_t) 16 + i_90934];
                
                // futhark/microgpt.fut:240:70-103
                
                double zt_res_90937 = zt_lhs_90936 * zt_lhs_90936;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90938 = r_90935 + zt_res_90937;
                double r_tmp_98065 = zp_res_90938;
                
                r_90935 = r_tmp_98065;
            }
            defunc_0_lifted_lambda_res_90933 = r_90935;
            // futhark/microgpt.fut:240:50-121
            
            double zs_res_90939 = defunc_0_lifted_lambda_res_90933 / 16.0;
            
            // futhark/microgpt.fut:241:23-53
            
            double zp_res_90940 = 1.0e-5 + zs_res_90939;
            
            // futhark/microgpt.fut:241:15-53
            
            double sqrt_res_90941 = futrts_sqrt64(zp_res_90940);
            
            // futhark/microgpt.fut:242:25-35
            
            double zs_res_90942 = 1.0 / sqrt_res_90941;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95204 = 0; i_95204 < (int64_t) 16; i_95204++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_92980 = ((double *) mem_96278)[i_95211 * (int64_t) 16 + i_95204];
                
                // futhark/microgpt.fut:242:5-35
                
                double zt_res_92981 = zs_res_90942 * zt_lhs_92980;
                
                // futhark/microgpt.fut:335:45-86
                
                double zt_res_92989 = zt_lhs_92980 * zt_lhs_92980;
                
                ((double *) mem_96319)[i_95204] = zt_res_92989;
                ((double *) mem_96320)[i_95204] = zt_res_92981;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96309, i_95211 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96319, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96310, i_95211 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96320, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95227 = 0; i_95227 < (int64_t) 16; i_95227++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91041;
            double r_91043 = 0.0;
            
            for (int64_t i_91042 = 0; i_91042 < (int64_t) 16; i_91042++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91044 = ((double *) mem_96310)[i_95227 * (int64_t) 16 + i_91042];
                
                // futhark/microgpt.fut:243:71-106
                
                double zt_res_91045 = zt_lhs_91044 * zt_lhs_91044;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91046 = r_91043 + zt_res_91045;
                double r_tmp_98071 = zp_res_91046;
                
                r_91043 = r_tmp_98071;
            }
            defunc_0_lifted_lambda_res_91041 = r_91043;
            // futhark/microgpt.fut:243:50-124
            
            double zs_res_91047 = defunc_0_lifted_lambda_res_91041 / 16.0;
            
            // futhark/microgpt.fut:244:24-54
            
            double zp_res_91048 = 1.0e-5 + zs_res_91047;
            
            // futhark/microgpt.fut:244:16-54
            
            double sqrt_res_91049 = futrts_sqrt64(zp_res_91048);
            
            // futhark/microgpt.fut:245:25-36
            
            double zs_res_91050 = 1.0 / sqrt_res_91049;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95218 = 0; i_95218 < (int64_t) 16; i_95218++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_93009 = ((double *) mem_96310)[i_95227 * (int64_t) 16 + i_95218];
                
                // futhark/microgpt.fut:245:5-36
                
                double zt_res_93010 = zs_res_91050 * zt_lhs_93009;
                
                // futhark/microgpt.fut:327:45-86
                
                double zt_res_93018 = zt_lhs_93009 * zt_lhs_93009;
                
                ((double *) mem_96355)[i_95218] = zt_res_93018;
                ((double *) mem_96356)[i_95218] = zt_res_93010;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91084;
            double r_91086 = 0.0;
            
            for (int64_t i_91085 = 0; i_91085 < (int64_t) 16; i_91085++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_91087 = ((double *) mem_96309)[i_95227 * (int64_t) 16 + i_91085];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91088 = r_91086 + lifted_lambda_res_91087;
                double r_tmp_98074 = zp_res_91088;
                
                r_91086 = r_tmp_98074;
            }
            defunc_0_lifted_lambda_res_91084 = r_91086;
            // futhark/microgpt.fut:336:36-94
            
            double zs_res_91089 = defunc_0_lifted_lambda_res_91084 / 16.0;
            
            ((double *) mem_96341)[i_95227] = zs_res_91089;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96342, i_95227 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96355, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96343, i_95227 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96356, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95251 = 0; i_95251 < (int64_t) 16; i_95251++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95237 = 0; i_95237 < (int64_t) 16; i_95237++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93081;
                double r_93083 = 0.0;
                
                for (int64_t i_93082 = 0; i_93082 < (int64_t) 16; i_93082++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93084 = ((double *) mem_param_96156.mem)[i_95237 * (int64_t) 16 + i_93082];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93085 = ((double *) mem_96343)[i_95251 * (int64_t) 16 + i_93082];
                    
                    // futhark/microgpt.fut:246:63-102
                    
                    double zt_res_93086 = zt_lhs_93084 * zt_rhs_93085;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93087 = r_93083 + zt_res_93086;
                    double r_tmp_98083 = zp_res_93087;
                    
                    r_93083 = r_tmp_98083;
                }
                defunc_0_lifted_lambda_res_93081 = r_93083;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93094;
                double r_93096 = 0.0;
                
                for (int64_t i_93095 = 0; i_93095 < (int64_t) 16; i_93095++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93097 = ((double *) mem_param_96144.mem)[i_95237 * (int64_t) 16 + i_93095];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93098 = ((double *) mem_96343)[i_95251 * (int64_t) 16 + i_93095];
                    
                    // futhark/microgpt.fut:247:63-102
                    
                    double zt_res_93099 = zt_lhs_93097 * zt_rhs_93098;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93100 = r_93096 + zt_res_93099;
                    double r_tmp_98084 = zp_res_93100;
                    
                    r_93096 = r_tmp_98084;
                }
                defunc_0_lifted_lambda_res_93094 = r_93096;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93110;
                double r_93112 = 0.0;
                
                for (int64_t i_93111 = 0; i_93111 < (int64_t) 16; i_93111++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93113 = ((double *) mem_param_96168.mem)[i_95237 * (int64_t) 16 + i_93111];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93114 = ((double *) mem_96343)[i_95251 * (int64_t) 16 + i_93111];
                    
                    // futhark/microgpt.fut:248:63-102
                    
                    double zt_res_93115 = zt_lhs_93113 * zt_rhs_93114;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93116 = r_93112 + zt_res_93115;
                    double r_tmp_98085 = zp_res_93116;
                    
                    r_93112 = r_tmp_98085;
                }
                defunc_0_lifted_lambda_res_93110 = r_93112;
                ((double *) mem_96403)[i_95237] = defunc_0_lifted_lambda_res_93110;
                ((double *) mem_96404)[i_95237] = defunc_0_lifted_lambda_res_93094;
                ((double *) mem_96405)[i_95237] = defunc_0_lifted_lambda_res_93081;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91431;
            double r_91433 = 0.0;
            
            for (int64_t i_91432 = 0; i_91432 < (int64_t) 16; i_91432++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_91434 = ((double *) mem_96342)[i_95251 * (int64_t) 16 + i_91432];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91435 = r_91433 + lifted_lambda_res_91434;
                double r_tmp_98086 = zp_res_91435;
                
                r_91433 = r_tmp_98086;
            }
            defunc_0_lifted_lambda_res_91431 = r_91433;
            // futhark/microgpt.fut:328:36-94
            
            double zs_res_91436 = defunc_0_lifted_lambda_res_91431 / 16.0;
            
            // futhark/microgpt.fut:337:43-55
            
            double zp_lhs_91450 = ((double *) mem_96341)[i_95251];
            
            // futhark/microgpt.fut:337:43-83
            
            double zp_res_91451 = 1.0e-5 + zp_lhs_91450;
            
            // futhark/microgpt.fut:337:35-83
            
            double sqrt_res_91452 = futrts_sqrt64(zp_res_91451);
            
            ((double *) mem_96380)[i_95251] = sqrt_res_91452;
            ((double *) mem_96381)[i_95251] = zs_res_91436;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96382, i_95251 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96403, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96383, i_95251 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96404, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96384, i_95251 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95283 = 0; i_95283 < (int64_t) 4; i_95283++) {
            // futhark/microgpt.fut:249:67-70
            
            int64_t zp_lhs_91524 = mul64((int64_t) 4, i_95283);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95273 = 0; i_95273 < (int64_t) 16; i_95273++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95263 = 0; i_95263 < (int64_t) 4; i_95263++) {
                    // futhark/microgpt.fut:249:72-79
                    
                    int64_t tmp_93274 = add64(zp_lhs_91524, i_95263);
                    
                    // futhark/microgpt.fut:249:48-81
                    
                    bool x_93275 = sle64((int64_t) 0, tmp_93274);
                    
                    // futhark/microgpt.fut:249:48-81
                    
                    bool y_93276 = slt64(tmp_93274, (int64_t) 16);
                    
                    // futhark/microgpt.fut:249:48-81
                    
                    bool bounds_check_93277 = x_93275 && y_93276;
                    
                    // futhark/microgpt.fut:249:48-81
                    
                    bool index_certs_93278;
                    
                    if (!bounds_check_93277) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93274, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:249:48-81\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:249:12-82\n   #9  futhark/microgpt.fut:469:5-76\n   #10 futhark/microgpt.fut:474:26-480:31\n   #11 futhark/microgpt.fut:496:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_93279 = ((double *) mem_96384)[i_95273 * (int64_t) 16 + tmp_93274];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_93287 = ((double *) mem_96383)[i_95273 * (int64_t) 16 + tmp_93274];
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_93298 = ((double *) mem_96382)[i_95273 * (int64_t) 16 + tmp_93274];
                    
                    ((double *) mem_96475)[i_95263] = lifted_lambda_res_93298;
                    ((double *) mem_96476)[i_95263] = lifted_lambda_res_93287;
                    ((double *) mem_96477)[i_95263] = lifted_lambda_res_93279;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96460, i_95273 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96475, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96461, i_95273 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96476, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96462, i_95273 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96477, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_96442, i_95283 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96460, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_96443, i_95283 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96461, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_96444, i_95283 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96462, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95336 = 0; i_95336 < (int64_t) 4; i_95336++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95298 = 0; i_95298 < (int64_t) 16; i_95298++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95291 = 0; i_95291 < (int64_t) 16; i_95291++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_93377;
                    double r_93379 = 0.0;
                    
                    for (int64_t i_93378 = 0; i_93378 < (int64_t) 4; i_93378++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_93380 = ((double *) mem_96444)[i_95336 * (int64_t) 64 + i_95298 * (int64_t) 4 + i_93378];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_93381 = ((double *) mem_96443)[i_95336 * (int64_t) 64 + i_95291 * (int64_t) 4 + i_93378];
                        
                        // futhark/microgpt.fut:252:110-163
                        
                        double zt_res_93382 = zt_lhs_93380 * zt_rhs_93381;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_93383 = r_93379 + zt_res_93382;
                        double r_tmp_98102 = zp_res_93383;
                        
                        r_93379 = r_tmp_98102;
                    }
                    defunc_0_lifted_lambda_res_93377 = r_93379;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_93390;
                    double r_93392 = 0.0;
                    
                    for (int64_t i_93391 = 0; i_93391 < (int64_t) 4; i_93391++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_93393 = ((double *) mem_96444)[i_95336 * (int64_t) 64 + i_95298 * (int64_t) 4 + i_93391];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_93394 = ((double *) mem_96443)[i_95336 * (int64_t) 64 + i_95291 * (int64_t) 4 + i_93391];
                        
                        // futhark/microgpt.fut:304:75-134
                        
                        double zt_res_93395 = zt_lhs_93393 * zt_rhs_93394;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_93396 = r_93392 + zt_res_93395;
                        double r_tmp_98103 = zp_res_93396;
                        
                        r_93392 = r_tmp_98103;
                    }
                    defunc_0_lifted_lambda_res_93390 = r_93392;
                    ((double *) mem_96545)[i_95291] = defunc_0_lifted_lambda_res_93390;
                    ((double *) mem_96546)[i_95291] = defunc_0_lifted_lambda_res_93377;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96535, i_95298 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96545, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96536, i_95298 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96546, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95307 = 0; i_95307 < (int64_t) 16; i_95307++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95303 = 0; i_95303 < (int64_t) 16; i_95303++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_91633 = ((double *) mem_96536)[i_95307 * (int64_t) 16 + i_95303];
                    
                    // futhark/microgpt.fut:253:47-78
                    
                    double zs_res_91634 = zs_lhs_91633 / 2.0;
                    double zp_rhs_91635 = ((double *) masks_mem_96134.mem)[step_88717 * (int64_t) 256 + i_95307 * (int64_t) 16 + i_95303];
                    
                    // futhark/microgpt.fut:253:65-102
                    
                    double zp_res_91636 = zs_res_91634 + zp_rhs_91635;
                    
                    ((double *) mem_96572)[i_95303] = zp_res_91636;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96567, i_95307 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96572, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95322 = 0; i_95322 < (int64_t) 16; i_95322++) {
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_94977;
                int64_t defunc_0_reduce_res_94978;
                double redout_95309;
                int64_t redout_95310;
                
                redout_95309 = -INFINITY;
                redout_95310 = (int64_t) 16;
                for (int64_t i_95311 = 0; i_95311 < (int64_t) 16; i_95311++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_93417 = ((double *) mem_96567)[i_95322 * (int64_t) 16 + i_95311];
                    
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_91661 = lifted_lambda_res_93417 < redout_95309;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_91662;
                    
                    if (zg_res_91661) {
                        lifted_lambda_res_91662 = redout_95309;
                    } else {
                        lifted_lambda_res_91662 = lifted_lambda_res_93417;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_91663;
                    
                    if (zg_res_91661) {
                        lifted_lambda_res_91663 = redout_95310;
                    } else {
                        lifted_lambda_res_91663 = i_95311;
                    }
                    
                    double redout_tmp_98107 = lifted_lambda_res_91662;
                    int64_t redout_tmp_98108 = lifted_lambda_res_91663;
                    
                    redout_95309 = redout_tmp_98107;
                    redout_95310 = redout_tmp_98108;
                }
                defunc_0_reduce_res_94977 = redout_95309;
                defunc_0_reduce_res_94978 = redout_95310;
                // futhark/microgpt.fut:254:56-112
                
                bool x_91664 = sle64((int64_t) 0, defunc_0_reduce_res_94978);
                
                // futhark/microgpt.fut:254:56-112
                
                bool y_91665 = slt64(defunc_0_reduce_res_94978, (int64_t) 16);
                
                // futhark/microgpt.fut:254:56-112
                
                bool bounds_check_91666 = x_91664 && y_91665;
                
                // futhark/microgpt.fut:254:56-112
                
                bool index_certs_91667;
                
                if (!bounds_check_91666) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_94978, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:254:56-112\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:254:16-257:38\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:27-39\n   #8  futhark/microgpt.fut:4:11-25\n   #9  futhark/microgpt.fut:9:13-40\n   #10 futhark/microgpt.fut:15:29-44\n   #11 futhark/microgpt.fut:4:11-25\n   #12 futhark/microgpt.fut:15:15-45\n   #13 futhark/microgpt.fut:252:12-258:79\n   #14 futhark/microgpt.fut:469:5-76\n   #15 futhark/microgpt.fut:474:26-480:31\n   #16 futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double x36_91668 = ((double *) mem_96567)[i_95322 * (int64_t) 16 + defunc_0_reduce_res_94978];
                
                // futhark/microgpt.fut:255:67-76
                
                double neg_res_91669 = -x36_91668;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95314 = 0; i_95314 < (int64_t) 16; i_95314++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_91676 = ((double *) mem_96567)[i_95322 * (int64_t) 16 + i_95314];
                    
                    // futhark/microgpt.fut:255:44-76
                    
                    double zp_res_91677 = neg_res_91669 + zp_lhs_91676;
                    
                    // futhark/microgpt.fut:255:37-76
                    
                    double exp_res_91678 = futrts_exp64(zp_res_91677);
                    
                    ((double *) mem_96588)[i_95314] = exp_res_91678;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91680;
                double r_91682 = 0.0;
                
                for (int64_t i_91681 = 0; i_91681 < (int64_t) 16; i_91681++) {
                    // futhark/microgpt.fut:256:36-46
                    
                    double lifted_lambda_res_91683 = ((double *) mem_96588)[i_91681];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91684 = r_91682 + lifted_lambda_res_91683;
                    double r_tmp_98110 = zp_res_91684;
                    
                    r_91682 = r_tmp_98110;
                }
                defunc_0_lifted_lambda_res_91680 = r_91682;
                // futhark/microgpt.fut:257:21-32
                
                double zs_res_91685 = 1.0 / defunc_0_lifted_lambda_res_91680;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95318 = 0; i_95318 < (int64_t) 16; i_95318++) {
                    // futhark/microgpt.fut:257:5-15
                    
                    double zt_lhs_91692 = ((double *) mem_96588)[i_95318];
                    
                    // futhark/microgpt.fut:257:5-32
                    
                    double zt_res_91693 = zs_res_91685 * zt_lhs_91692;
                    
                    ((double *) mem_96595)[i_95318] = zt_res_91693;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96583, i_95322 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96595, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95330 = 0; i_95330 < (int64_t) 16; i_95330++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95326 = 0; i_95326 < (int64_t) 4; i_95326++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_91708;
                    double r_91710 = 0.0;
                    
                    for (int64_t i_91709 = 0; i_91709 < (int64_t) 16; i_91709++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_91711 = ((double *) mem_96583)[i_95330 * (int64_t) 16 + i_91709];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_91712 = ((double *) mem_96442)[i_95336 * (int64_t) 64 + i_91709 * (int64_t) 4 + i_95326];
                        
                        // futhark/microgpt.fut:258:26-72
                        
                        double zt_res_91713 = zt_lhs_91711 * zt_rhs_91712;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_91714 = r_91710 + zt_res_91713;
                        double r_tmp_98114 = zp_res_91714;
                        
                        r_91710 = r_tmp_98114;
                    }
                    defunc_0_lifted_lambda_res_91708 = r_91710;
                    ((double *) mem_96611)[i_95326] = defunc_0_lifted_lambda_res_91708;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_96606, i_95330 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96611, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_96523, i_95336 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_96535, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_96524, i_95336 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_96606, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95347 = 0; i_95347 < (int64_t) 16; i_95347++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95341 = 0; i_95341 < (int64_t) 16; i_95341++) {
                // futhark/microgpt.fut:259:52-55
                
                int64_t tmp_91763 = sdiv64(i_95341, (int64_t) 4);
                
                // futhark/microgpt.fut:259:41-57
                
                bool x_91764 = sle64((int64_t) 0, tmp_91763);
                
                // futhark/microgpt.fut:259:41-57
                
                bool y_91765 = slt64(tmp_91763, (int64_t) 4);
                
                // futhark/microgpt.fut:259:41-57
                
                bool bounds_check_91766 = x_91764 && y_91765;
                
                // futhark/microgpt.fut:259:41-57
                
                bool index_certs_91767;
                
                if (!bounds_check_91766) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_91763, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:259:41-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:259:12-78\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:259:72-75
                
                int64_t tmp_91768 = smod64(i_95341, (int64_t) 4);
                
                // futhark/microgpt.fut:259:41-77
                
                bool x_91769 = sle64((int64_t) 0, tmp_91768);
                
                // futhark/microgpt.fut:259:41-77
                
                bool y_91770 = slt64(tmp_91768, (int64_t) 4);
                
                // futhark/microgpt.fut:259:41-77
                
                bool bounds_check_91771 = x_91769 && y_91770;
                
                // futhark/microgpt.fut:259:41-77
                
                bool index_certs_91772;
                
                if (!bounds_check_91771) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_91768, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:259:41-77\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:259:12-78\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_91773 = ((double *) mem_96524)[tmp_91763 * (int64_t) 64 + i_95347 * (int64_t) 4 + tmp_91768];
                
                ((double *) mem_96641)[i_95341] = lifted_lambda_res_91773;
            }
            // futhark/microgpt.fut:329:43-55
            
            double zp_lhs_91781 = ((double *) mem_96381)[i_95347];
            
            // futhark/microgpt.fut:329:43-83
            
            double zp_res_91782 = 1.0e-5 + zp_lhs_91781;
            
            // futhark/microgpt.fut:329:35-83
            
            double sqrt_res_91783 = futrts_sqrt64(zp_res_91782);
            
            ((double *) mem_96632)[i_95347] = sqrt_res_91783;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96633, i_95347 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96641, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95356 = 0; i_95356 < (int64_t) 16; i_95356++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95352 = 0; i_95352 < (int64_t) 16; i_95352++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89141;
                double r_89143 = 0.0;
                
                for (int64_t i_89142 = 0; i_89142 < (int64_t) 16; i_89142++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_89144 = ((double *) mem_param_96148.mem)[i_95352 * (int64_t) 16 + i_89142];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_89145 = ((double *) mem_96633)[i_95356 * (int64_t) 16 + i_89142];
                    
                    // futhark/microgpt.fut:260:63-103
                    
                    double zt_res_89146 = zt_lhs_89144 * zt_rhs_89145;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89147 = r_89143 + zt_res_89146;
                    double r_tmp_98120 = zp_res_89147;
                    
                    r_89143 = r_tmp_98120;
                }
                defunc_0_lifted_lambda_res_89141 = r_89143;
                ((double *) mem_96660)[i_95352] = defunc_0_lifted_lambda_res_89141;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96655, i_95356 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96660, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95364 = 0; i_95364 < (int64_t) 16; i_95364++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95360 = 0; i_95360 < (int64_t) 16; i_95360++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89162 = ((double *) mem_96655)[i_95364 * (int64_t) 16 + i_95360];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_89163 = ((double *) mem_96310)[i_95364 * (int64_t) 16 + i_95360];
                
                // futhark/microgpt.fut:261:42-80
                
                double zp_res_89164 = zp_lhs_89162 + zp_rhs_89163;
                
                ((double *) mem_96676)[i_95360] = zp_res_89164;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96671, i_95364 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96676, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95377 = 0; i_95377 < (int64_t) 16; i_95377++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91801;
            double r_91803 = 0.0;
            
            for (int64_t i_91802 = 0; i_91802 < (int64_t) 16; i_91802++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_91804 = ((double *) mem_96671)[i_95377 * (int64_t) 16 + i_91802];
                
                // futhark/microgpt.fut:262:75-114
                
                double zt_res_91805 = zt_lhs_91804 * zt_lhs_91804;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91806 = r_91803 + zt_res_91805;
                double r_tmp_98125 = zp_res_91806;
                
                r_91803 = r_tmp_98125;
            }
            defunc_0_lifted_lambda_res_91801 = r_91803;
            // futhark/microgpt.fut:262:54-132
            
            double zs_res_91807 = defunc_0_lifted_lambda_res_91801 / 16.0;
            
            // futhark/microgpt.fut:263:24-55
            
            double zp_res_91808 = 1.0e-5 + zs_res_91807;
            
            // futhark/microgpt.fut:263:16-55
            
            double sqrt_res_91809 = futrts_sqrt64(zp_res_91808);
            
            // futhark/microgpt.fut:264:28-39
            
            double zs_res_91810 = 1.0 / sqrt_res_91809;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95370 = 0; i_95370 < (int64_t) 16; i_95370++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_93459 = ((double *) mem_96671)[i_95377 * (int64_t) 16 + i_95370];
                
                // futhark/microgpt.fut:264:5-39
                
                double zt_res_93460 = zs_res_91810 * zt_lhs_93459;
                
                // futhark/microgpt.fut:294:45-88
                
                double zt_res_93468 = zt_lhs_93459 * zt_lhs_93459;
                
                ((double *) mem_96697)[i_95370] = zt_res_93468;
                ((double *) mem_96698)[i_95370] = zt_res_93460;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96687, i_95377 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96697, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96688, i_95377 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96698, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95388 = 0; i_95388 < (int64_t) 16; i_95388++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95382 = 0; i_95382 < (int64_t) 64; i_95382++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_91858;
                double r_91860 = 0.0;
                
                for (int64_t i_91859 = 0; i_91859 < (int64_t) 16; i_91859++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_91861 = ((double *) mem_param_96164.mem)[i_95382 * (int64_t) 16 + i_91859];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_91862 = ((double *) mem_96688)[i_95388 * (int64_t) 16 + i_91859];
                    
                    // futhark/microgpt.fut:265:63-102
                    
                    double zt_res_91863 = zt_lhs_91861 * zt_rhs_91862;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_91864 = r_91860 + zt_res_91863;
                    double r_tmp_98131 = zp_res_91864;
                    
                    r_91860 = r_tmp_98131;
                }
                defunc_0_lifted_lambda_res_91858 = r_91860;
                ((double *) mem_96728)[i_95382] = defunc_0_lifted_lambda_res_91858;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_91872;
            double r_91874 = 0.0;
            
            for (int64_t i_91873 = 0; i_91873 < (int64_t) 16; i_91873++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_91875 = ((double *) mem_96687)[i_95388 * (int64_t) 16 + i_91873];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_91876 = r_91874 + lifted_lambda_res_91875;
                double r_tmp_98132 = zp_res_91876;
                
                r_91874 = r_tmp_98132;
            }
            defunc_0_lifted_lambda_res_91872 = r_91874;
            // futhark/microgpt.fut:295:36-94
            
            double zs_res_91877 = defunc_0_lifted_lambda_res_91872 / 16.0;
            
            ((double *) mem_96719)[i_95388] = zs_res_91877;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96720, i_95388 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96728, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95399 = 0; i_95399 < (int64_t) 16; i_95399++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95393 = 0; i_95393 < (int64_t) 64; i_95393++) {
                // futhark/microgpt.fut:4:11-25
                
                double max_arg0_91901 = ((double *) mem_96720)[i_95399 * (int64_t) 64 + i_95393];
                
                // futhark/microgpt.fut:266:41-69
                
                double max_res_91902 = fmax64(0.0, max_arg0_91901);
                
                ((double *) mem_96751)[i_95393] = max_res_91902;
            }
            // futhark/microgpt.fut:296:43-55
            
            double zp_lhs_91910 = ((double *) mem_96719)[i_95399];
            
            // futhark/microgpt.fut:296:43-83
            
            double zp_res_91911 = 1.0e-5 + zp_lhs_91910;
            
            // futhark/microgpt.fut:296:35-83
            
            double sqrt_res_91912 = futrts_sqrt64(zp_res_91911);
            
            ((double *) mem_96742)[i_95399] = sqrt_res_91912;
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96743, i_95399 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96751, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95408 = 0; i_95408 < (int64_t) 16; i_95408++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95404 = 0; i_95404 < (int64_t) 16; i_95404++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89243;
                double r_89245 = 0.0;
                
                for (int64_t i_89244 = 0; i_89244 < (int64_t) 64; i_89244++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_89246 = ((double *) mem_param_96140.mem)[i_95404 * (int64_t) 64 + i_89244];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_89247 = ((double *) mem_96743)[i_95408 * (int64_t) 64 + i_89244];
                    
                    // futhark/microgpt.fut:267:63-104
                    
                    double zt_res_89248 = zt_lhs_89246 * zt_rhs_89247;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89249 = r_89245 + zt_res_89248;
                    double r_tmp_98138 = zp_res_89249;
                    
                    r_89245 = r_tmp_98138;
                }
                defunc_0_lifted_lambda_res_89243 = r_89245;
                ((double *) mem_96770)[i_95404] = defunc_0_lifted_lambda_res_89243;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96765, i_95408 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96770, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95416 = 0; i_95416 < (int64_t) 16; i_95416++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95412 = 0; i_95412 < (int64_t) 16; i_95412++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89264 = ((double *) mem_96765)[i_95416 * (int64_t) 16 + i_95412];
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_89265 = ((double *) mem_96671)[i_95416 * (int64_t) 16 + i_95412];
                
                // futhark/microgpt.fut:268:42-81
                
                double zp_res_89266 = zp_lhs_89264 + zp_rhs_89265;
                
                ((double *) mem_96786)[i_95412] = zp_res_89266;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96781, i_95416 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96786, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95424 = 0; i_95424 < (int64_t) 16; i_95424++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95420 = 0; i_95420 < (int64_t) 27; i_95420++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89282;
                double r_89284 = 0.0;
                
                for (int64_t i_89283 = 0; i_89283 < (int64_t) 16; i_89283++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_89285 = ((double *) mem_param_96172.mem)[i_95420 * (int64_t) 16 + i_89283];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_89286 = ((double *) mem_96781)[i_95424 * (int64_t) 16 + i_89283];
                    
                    // futhark/microgpt.fut:269:63-103
                    
                    double zt_res_89287 = zt_lhs_89285 * zt_rhs_89286;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89288 = r_89284 + zt_res_89287;
                    double r_tmp_98143 = zp_res_89288;
                    
                    r_89284 = r_tmp_98143;
                }
                defunc_0_lifted_lambda_res_89282 = r_89284;
                ((double *) mem_96802)[i_95420] = defunc_0_lifted_lambda_res_89282;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96797, i_95424 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96802, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95431 = 0; i_95431 < (int64_t) 16; i_95431++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_94997;
            int64_t defunc_0_reduce_res_94998;
            double redout_95426;
            int64_t redout_95427;
            
            redout_95426 = -INFINITY;
            redout_95427 = (int64_t) 27;
            for (int64_t i_95428 = 0; i_95428 < (int64_t) 27; i_95428++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93493 = ((double *) mem_96797)[i_95431 * (int64_t) 27 + i_95428];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_89329 = lifted_lambda_res_93493 < redout_95426;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_89330;
                
                if (zg_res_89329) {
                    lifted_lambda_res_89330 = redout_95426;
                } else {
                    lifted_lambda_res_89330 = lifted_lambda_res_93493;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_89331;
                
                if (zg_res_89329) {
                    lifted_lambda_res_89331 = redout_95427;
                } else {
                    lifted_lambda_res_89331 = i_95428;
                }
                
                double redout_tmp_98145 = lifted_lambda_res_89330;
                int64_t redout_tmp_98146 = lifted_lambda_res_89331;
                
                redout_95426 = redout_tmp_98145;
                redout_95427 = redout_tmp_98146;
            }
            defunc_0_reduce_res_94997 = redout_95426;
            defunc_0_reduce_res_94998 = redout_95427;
            // futhark/microgpt.fut:276:32-88
            
            bool x_89332 = sle64((int64_t) 0, defunc_0_reduce_res_94998);
            
            // futhark/microgpt.fut:276:32-88
            
            bool y_89333 = slt64(defunc_0_reduce_res_94998, (int64_t) 27);
            
            // futhark/microgpt.fut:276:32-88
            
            bool bounds_check_89334 = x_89332 && y_89333;
            
            // futhark/microgpt.fut:276:32-88
            
            bool index_certs_89335;
            
            if (!bounds_check_89334) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_94998, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:276:32-88\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:276:12-89\n   #4  futhark/microgpt.fut:469:5-76\n   #5  futhark/microgpt.fut:474:26-480:31\n   #6  futhark/microgpt.fut:496:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double lifted_lambda_res_89336 = ((double *) mem_96797)[i_95431 * (int64_t) 27 + defunc_0_reduce_res_94998];
            
            ((double *) mem_96813)[i_95431] = lifted_lambda_res_89336;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95439 = 0; i_95439 < (int64_t) 16; i_95439++) {
            // futhark/microgpt.fut:277:78-88
            
            double neg_arg0_89344 = ((double *) mem_96813)[i_95439];
            
            // futhark/microgpt.fut:277:72-88
            
            double neg_res_89345 = -neg_arg0_89344;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95435 = 0; i_95435 < (int64_t) 27; i_95435++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89352 = ((double *) mem_96797)[i_95439 * (int64_t) 27 + i_95435];
                
                // futhark/microgpt.fut:277:49-88
                
                double zp_res_89353 = neg_res_89345 + zp_lhs_89352;
                
                // futhark/microgpt.fut:277:42-88
                
                double exp_res_89354 = futrts_exp64(zp_res_89353);
                
                ((double *) mem_96825)[i_95435] = exp_res_89354;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96820, i_95439 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96825, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95443 = 0; i_95443 < (int64_t) 16; i_95443++) {
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_89363;
            double r_89365 = 0.0;
            
            for (int64_t i_89364 = 0; i_89364 < (int64_t) 27; i_89364++) {
                // futhark/microgpt.fut:71:46-49
                
                double lifted_lambda_res_89366 = ((double *) mem_96820)[i_95443 * (int64_t) 27 + i_89364];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_89367 = r_89365 + lifted_lambda_res_89366;
                double r_tmp_98150 = zp_res_89367;
                
                r_89365 = r_tmp_98150;
            }
            defunc_0_lifted_lambda_res_89363 = r_89365;
            ((double *) mem_96836)[i_95443] = defunc_0_lifted_lambda_res_89363;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95454 = 0; i_95454 < (int64_t) 16; i_95454++) {
            // futhark/microgpt.fut:279:65-75
            
            double zt_lhs_89375 = ((double *) mem_96836)[i_95454];
            
            // futhark/microgpt.fut:279:65-90
            
            double zt_res_89376 = zt_lhs_89375 * zt_lhs_89375;
            
            // futhark/microgpt.fut:283:99-117
            
            double zs_res_89377 = 1.0 / zt_res_89376;
            double x_95001;
            int64_t x_95002;
            double redout_95445;
            int64_t redout_95446;
            
            redout_95445 = -INFINITY;
            redout_95446 = (int64_t) 27;
            for (int64_t i_95447 = 0; i_95447 < (int64_t) 27; i_95447++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93517 = ((double *) mem_96797)[i_95454 * (int64_t) 27 + i_95447];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_89397 = lifted_lambda_res_93517 < redout_95445;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_89398;
                
                if (zg_res_89397) {
                    lifted_lambda_res_89398 = redout_95445;
                } else {
                    lifted_lambda_res_89398 = lifted_lambda_res_93517;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_89399;
                
                if (zg_res_89397) {
                    lifted_lambda_res_89399 = redout_95446;
                } else {
                    lifted_lambda_res_89399 = i_95447;
                }
                
                double redout_tmp_98152 = lifted_lambda_res_89398;
                int64_t redout_tmp_98153 = lifted_lambda_res_89399;
                
                redout_95445 = redout_tmp_98152;
                redout_95446 = redout_tmp_98153;
            }
            x_95001 = redout_95445;
            x_95002 = redout_95446;
            
            double x_93536 = ((double *) mem_96797)[i_95454 * (int64_t) 27 + x_95002];
            
            // futhark/microgpt.fut:281:67-76
            
            double neg_res_89405 = -x_93536;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_89378;
            double r_89380 = 0.0;
            
            for (int64_t i_89379 = 0; i_89379 < (int64_t) 27; i_89379++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95450 = 0; i_95450 < (int64_t) 27; i_95450++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_89412 = ((double *) mem_96797)[i_95454 * (int64_t) 27 + i_95450];
                    
                    // futhark/microgpt.fut:281:44-76
                    
                    double zp_res_89413 = neg_res_89405 + zp_lhs_89412;
                    
                    // futhark/microgpt.fut:281:37-76
                    
                    double exp_res_89414 = futrts_exp64(zp_res_89413);
                    
                    ((double *) mem_96847)[i_95450] = exp_res_89414;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89416;
                double r_89418 = 0.0;
                
                for (int64_t i_89417 = 0; i_89417 < (int64_t) 27; i_89417++) {
                    // futhark/microgpt.fut:282:36-46
                    
                    double lifted_lambda_res_89419 = ((double *) mem_96847)[i_89417];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89420 = r_89418 + lifted_lambda_res_89419;
                    double r_tmp_98156 = zp_res_89420;
                    
                    r_89418 = r_tmp_98156;
                }
                defunc_0_lifted_lambda_res_89416 = r_89418;
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_89421 = ((double *) mem_96277)[i_95454 * (int64_t) 27 + i_89379];
                
                // futhark/microgpt.fut:283:39-49
                
                double zt_lhs_89422 = ((double *) mem_96847)[i_89379];
                
                // futhark/microgpt.fut:283:55-66
                
                double zs_res_89423 = 1.0 / defunc_0_lifted_lambda_res_89416;
                
                // futhark/microgpt.fut:283:39-66
                
                double zt_res_89424 = zt_lhs_89422 * zs_res_89423;
                
                // futhark/microgpt.fut:283:30-66
                
                double zs_res_89425 = 1.0 / zt_res_89424;
                
                // futhark/microgpt.fut:283:7-66
                
                double zt_res_89426 = zt_lhs_89421 * zs_res_89425;
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_89427 = ((double *) mem_96820)[i_95454 * (int64_t) 27 + i_89379];
                
                // futhark/microgpt.fut:283:25-92
                
                double zt_res_89428 = zt_res_89426 * zt_rhs_89427;
                
                // futhark/microgpt.fut:283:71-117
                
                double zt_res_89429 = zs_res_89377 * zt_res_89428;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_89430 = r_89380 + zt_res_89429;
                double r_tmp_98154 = zp_res_89430;
                
                r_89380 = r_tmp_98154;
            }
            defunc_0_lifted_lambda_res_89378 = r_89380;
            // futhark/microgpt.fut:280:5-283:123
            
            double neg_res_89431 = -defunc_0_lifted_lambda_res_89378;
            
            ((double *) mem_96843)[i_95454] = neg_res_89431;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95469 = 0; i_95469 < (int64_t) 16; i_95469++) {
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_95003;
            int64_t defunc_0_reduce_res_95004;
            double redout_95456;
            int64_t redout_95457;
            
            redout_95456 = -INFINITY;
            redout_95457 = (int64_t) 27;
            for (int64_t i_95458 = 0; i_95458 < (int64_t) 27; i_95458++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93553 = ((double *) mem_96797)[i_95469 * (int64_t) 27 + i_95458];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_89455 = lifted_lambda_res_93553 < redout_95456;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_89456;
                
                if (zg_res_89455) {
                    lifted_lambda_res_89456 = redout_95456;
                } else {
                    lifted_lambda_res_89456 = lifted_lambda_res_93553;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_89457;
                
                if (zg_res_89455) {
                    lifted_lambda_res_89457 = redout_95457;
                } else {
                    lifted_lambda_res_89457 = i_95458;
                }
                
                double redout_tmp_98158 = lifted_lambda_res_89456;
                int64_t redout_tmp_98159 = lifted_lambda_res_89457;
                
                redout_95456 = redout_tmp_98158;
                redout_95457 = redout_tmp_98159;
            }
            defunc_0_reduce_res_95003 = redout_95456;
            defunc_0_reduce_res_95004 = redout_95457;
            // futhark/microgpt.fut:284:55-115
            
            bool x_89458 = sle64((int64_t) 0, defunc_0_reduce_res_95004);
            
            // futhark/microgpt.fut:284:55-115
            
            bool y_89459 = slt64(defunc_0_reduce_res_95004, (int64_t) 27);
            
            // futhark/microgpt.fut:284:55-115
            
            bool bounds_check_89460 = x_89458 && y_89459;
            
            // futhark/microgpt.fut:284:55-115
            
            bool index_certs_89461;
            
            if (!bounds_check_89460) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_95004, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:284:55-115\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:284:12-287:123\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            
            double x101_89462 = ((double *) mem_96797)[i_95469 * (int64_t) 27 + defunc_0_reduce_res_95004];
            
            // futhark/microgpt.fut:285:71-81
            
            double neg_res_89463 = -x101_89462;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95461 = 0; i_95461 < (int64_t) 27; i_95461++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89470 = ((double *) mem_96797)[i_95469 * (int64_t) 27 + i_95461];
                
                // futhark/microgpt.fut:285:46-81
                
                double zp_res_89471 = neg_res_89463 + zp_lhs_89470;
                
                // futhark/microgpt.fut:285:39-81
                
                double exp_res_89472 = futrts_exp64(zp_res_89471);
                
                ((double *) mem_96862)[i_95461] = exp_res_89472;
            }
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_89474;
            double r_89476 = 0.0;
            
            for (int64_t i_89475 = 0; i_89475 < (int64_t) 27; i_89475++) {
                // futhark/microgpt.fut:286:38-50
                
                double lifted_lambda_res_89477 = ((double *) mem_96862)[i_89475];
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_89478 = r_89476 + lifted_lambda_res_89477;
                double r_tmp_98161 = zp_res_89478;
                
                r_89476 = r_tmp_98161;
            }
            defunc_0_lifted_lambda_res_89474 = r_89476;
            // futhark/microgpt.fut:287:59-71
            
            double zs_res_89479 = 1.0 / defunc_0_lifted_lambda_res_89474;
            
            // futhark/microgpt.fut:287:89-100
            
            double zs_rhs_89480 = ((double *) mem_96836)[i_95469];
            
            // futhark/microgpt.fut:287:81-100
            
            double zs_res_89481 = 1.0 / zs_rhs_89480;
            
            // futhark/microgpt.fut:287:107-118
            
            double zp_rhs_89482 = ((double *) mem_96843)[i_95469];
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95465 = 0; i_95465 < (int64_t) 27; i_95465++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_89489 = ((double *) mem_96277)[i_95469 * (int64_t) 27 + i_95465];
                
                // futhark/microgpt.fut:287:41-53
                
                double zt_lhs_89490 = ((double *) mem_96862)[i_95465];
                
                // futhark/microgpt.fut:287:41-71
                
                double zt_res_89491 = zs_res_89479 * zt_lhs_89490;
                
                // futhark/microgpt.fut:287:32-71
                
                double zs_res_89492 = 1.0 / zt_res_89491;
                
                // futhark/microgpt.fut:287:7-71
                
                double zt_res_89493 = zt_lhs_89489 * zs_res_89492;
                
                // futhark/microgpt.fut:287:27-100
                
                double zt_res_89494 = zs_res_89481 * zt_res_89493;
                
                // futhark/microgpt.fut:287:76-118
                
                double zp_res_89495 = zp_rhs_89482 + zt_res_89494;
                
                ((double *) mem_96869)[i_95465] = zp_res_89495;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96857, i_95469 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96869, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95473 = 0; i_95473 < (int64_t) 16; i_95473++) {
            double eta_p_elem_89500 = ((double *) mem_96813)[i_95473];
            
            // futhark/microgpt.fut:288:97-114
            
            double neg_res_89505 = -eta_p_elem_89500;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_89506;
            double r_89508 = 0.0;
            
            for (int64_t i_89507 = 0; i_89507 < (int64_t) 27; i_89507++) {
                // futhark/microgpt.fut:71:46-49
                
                double zp_lhs_89509 = ((double *) mem_96797)[i_95473 * (int64_t) 27 + i_89507];
                
                // futhark/microgpt.fut:288:72-114
                
                double zp_res_89510 = neg_res_89505 + zp_lhs_89509;
                
                // futhark/microgpt.fut:288:65-114
                
                double exp_res_89511 = futrts_exp64(zp_res_89510);
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_89512 = ((double *) mem_96857)[i_95473 * (int64_t) 27 + i_89507];
                
                // futhark/microgpt.fut:288:65-141
                
                double zt_res_89513 = exp_res_89511 * zt_rhs_89512;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_89514 = r_89508 + zt_res_89513;
                double r_tmp_98164 = zp_res_89514;
                
                r_89508 = r_tmp_98164;
            }
            defunc_0_lifted_lambda_res_89506 = r_89508;
            // futhark/microgpt.fut:288:35-143
            
            double neg_res_89515 = -defunc_0_lifted_lambda_res_89506;
            
            ((double *) mem_96880)[i_95473] = neg_res_89515;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95484 = 0; i_95484 < (int64_t) 16; i_95484++) {
            // futhark/microgpt.fut:289:85-96
            
            double neg_arg0_89523 = ((double *) mem_96813)[i_95484];
            
            // futhark/microgpt.fut:289:79-96
            
            double neg_res_89524 = -neg_arg0_89523;
            
            // futhark/microgpt.fut:115:5-117:48
            
            double defunc_0_reduce_res_95012;
            int64_t defunc_0_reduce_res_95013;
            double redout_95475;
            int64_t redout_95476;
            
            redout_95475 = -INFINITY;
            redout_95476 = (int64_t) 27;
            for (int64_t i_95477 = 0; i_95477 < (int64_t) 27; i_95477++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93578 = ((double *) mem_96797)[i_95484 * (int64_t) 27 + i_95477];
                
                // futhark/microgpt.fut:116:31-71
                
                bool zg_res_89541 = lifted_lambda_res_93578 < redout_95475;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double lifted_lambda_res_89542;
                
                if (zg_res_89541) {
                    lifted_lambda_res_89542 = redout_95475;
                } else {
                    lifted_lambda_res_89542 = lifted_lambda_res_93578;
                }
                // futhark/microgpt.fut:115:5-117:48
                
                int64_t lifted_lambda_res_89543;
                
                if (zg_res_89541) {
                    lifted_lambda_res_89543 = redout_95476;
                } else {
                    lifted_lambda_res_89543 = i_95477;
                }
                
                double redout_tmp_98166 = lifted_lambda_res_89542;
                int64_t redout_tmp_98167 = lifted_lambda_res_89543;
                
                redout_95475 = redout_tmp_98166;
                redout_95476 = redout_tmp_98167;
            }
            defunc_0_reduce_res_95012 = redout_95475;
            defunc_0_reduce_res_95013 = redout_95476;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95480 = 0; i_95480 < (int64_t) 27; i_95480++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89550 = ((double *) mem_96797)[i_95484 * (int64_t) 27 + i_95480];
                
                // futhark/microgpt.fut:289:54-96
                
                double zp_res_89551 = neg_res_89524 + zp_lhs_89550;
                
                // futhark/microgpt.fut:289:47-96
                
                double exp_res_89552 = futrts_exp64(zp_res_89551);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_89553 = ((double *) mem_96857)[i_95484 * (int64_t) 27 + i_95480];
                
                // futhark/microgpt.fut:289:47-123
                
                double zt_res_89554 = exp_res_89552 * zt_rhs_89553;
                
                // futhark/microgpt.fut:289:130-222
                
                bool cond_89555 = i_95480 == defunc_0_reduce_res_95013;
                
                // futhark/microgpt.fut:289:130-222
                
                double zp_rhs_89556;
                
                if (cond_89555) {
                    // futhark/microgpt.fut:289:200-212
                    
                    double zp_rhs_t_res_95011 = ((double *) mem_96880)[i_95484];
                    
                    zp_rhs_89556 = zp_rhs_t_res_95011;
                } else {
                    zp_rhs_89556 = 0.0;
                }
                // futhark/microgpt.fut:289:100-222
                
                double zp_res_89562 = zt_res_89554 + zp_rhs_89556;
                
                ((double *) mem_96892)[i_95480] = zp_res_89562;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96887, i_95484 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96892, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95492 = 0; i_95492 < (int64_t) 16; i_95492++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95488 = 0; i_95488 < (int64_t) 16; i_95488++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89577;
                double r_89579 = 0.0;
                
                for (int64_t i_89578 = 0; i_89578 < (int64_t) 27; i_89578++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_89580 = ((double *) mem_96887)[i_95492 * (int64_t) 27 + i_89578];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_89581 = ((double *) mem_param_96172.mem)[i_89578 * (int64_t) 16 + i_95488];
                    
                    // futhark/microgpt.fut:290:67-112
                    
                    double zt_res_89582 = zt_lhs_89580 * zt_rhs_89581;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89583 = r_89579 + zt_res_89582;
                    double r_tmp_98171 = zp_res_89583;
                    
                    r_89579 = r_tmp_98171;
                }
                defunc_0_lifted_lambda_res_89577 = r_89579;
                ((double *) mem_96908)[i_95488] = defunc_0_lifted_lambda_res_89577;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96903, i_95492 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96908, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95505 = 0; i_95505 < (int64_t) 16; i_95505++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95498 = 0; i_95498 < (int64_t) 64; i_95498++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93613;
                double r_93615 = 0.0;
                
                for (int64_t i_93614 = 0; i_93614 < (int64_t) 16; i_93614++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93616 = ((double *) mem_96903)[i_95505 * (int64_t) 16 + i_93614];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93617 = ((double *) mem_param_96140.mem)[i_93614 * (int64_t) 64 + i_95498];
                    
                    // futhark/microgpt.fut:291:67-113
                    
                    double zt_res_93618 = zt_lhs_93616 * zt_rhs_93617;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93619 = r_93615 + zt_res_93618;
                    double r_tmp_98176 = zp_res_93619;
                    
                    r_93615 = r_tmp_98176;
                }
                defunc_0_lifted_lambda_res_93613 = r_93615;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93626;
                double r_93628 = 0.0;
                
                for (int64_t i_93627 = 0; i_93627 < (int64_t) 16; i_93627++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93629 = ((double *) mem_96903)[i_93627 * (int64_t) 16 + i_95505];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93630 = ((double *) mem_96743)[i_93627 * (int64_t) 64 + i_95498];
                    
                    // futhark/microgpt.fut:351:69-113
                    
                    double zt_res_93631 = zt_lhs_93629 * zt_rhs_93630;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93632 = r_93628 + zt_res_93631;
                    double r_tmp_98177 = zp_res_93632;
                    
                    r_93628 = r_tmp_98177;
                }
                defunc_0_lifted_lambda_res_93626 = r_93628;
                ((double *) mem_96929)[i_95498] = defunc_0_lifted_lambda_res_93626;
                ((double *) mem_96930)[i_95498] = defunc_0_lifted_lambda_res_93613;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96919, i_95505 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96929, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96920, i_95505 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96930, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95514 = 0; i_95514 < (int64_t) 16; i_95514++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95510 = 0; i_95510 < (int64_t) 64; i_95510++) {
                // futhark/microgpt.fut:4:11-25
                
                double indicatorp_arg0_89619 = ((double *) mem_96720)[i_95514 * (int64_t) 64 + i_95510];
                
                // futhark/microgpt.fut:110:42-54
                
                double max_res_89620 = fmax64(0.0, indicatorp_arg0_89619);
                
                // futhark/microgpt.fut:110:35-54
                
                double sgn_res_89621 = fsignum64(max_res_89620);
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_89622 = ((double *) mem_96920)[i_95514 * (int64_t) 64 + i_95510];
                
                // futhark/microgpt.fut:292:46-102
                
                double zt_res_89623 = sgn_res_89621 * zt_rhs_89622;
                
                ((double *) mem_96956)[i_95510] = zt_res_89623;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96951, i_95514 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96956, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95522 = 0; i_95522 < (int64_t) 16; i_95522++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95518 = 0; i_95518 < (int64_t) 16; i_95518++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_89638;
                double r_89640 = 0.0;
                
                for (int64_t i_89639 = 0; i_89639 < (int64_t) 64; i_89639++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_89641 = ((double *) mem_96951)[i_95522 * (int64_t) 64 + i_89639];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_89642 = ((double *) mem_param_96164.mem)[i_89639 * (int64_t) 16 + i_95518];
                    
                    // futhark/microgpt.fut:293:67-111
                    
                    double zt_res_89643 = zt_lhs_89641 * zt_rhs_89642;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_89644 = r_89640 + zt_res_89643;
                    double r_tmp_98182 = zp_res_89644;
                    
                    r_89640 = r_tmp_98182;
                }
                defunc_0_lifted_lambda_res_89638 = r_89640;
                ((double *) mem_96972)[i_95518] = defunc_0_lifted_lambda_res_89638;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_96967, i_95522 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_96972, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95526 = 0; i_95526 < (int64_t) 16; i_95526++) {
            // futhark/microgpt.fut:297:69-81
            
            double zt_lhs_89692 = ((double *) mem_96742)[i_95526];
            
            // futhark/microgpt.fut:297:69-98
            
            double zt_res_89693 = zt_lhs_89692 * zt_lhs_89692;
            
            // futhark/microgpt.fut:298:86-106
            
            double zs_res_89694 = 1.0 / zt_res_89693;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_89695;
            double r_89697 = 0.0;
            
            for (int64_t i_89696 = 0; i_89696 < (int64_t) 16; i_89696++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_89698 = ((double *) mem_96967)[i_95526 * (int64_t) 16 + i_89696];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_89699 = ((double *) mem_96671)[i_95526 * (int64_t) 16 + i_89696];
                
                // futhark/microgpt.fut:298:35-79
                
                double zt_res_89700 = zt_lhs_89698 * zt_rhs_89699;
                
                // futhark/microgpt.fut:298:56-106
                
                double zt_res_89701 = zs_res_89694 * zt_res_89700;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_89702 = r_89697 + zt_res_89701;
                double r_tmp_98184 = zp_res_89702;
                
                r_89697 = r_tmp_98184;
            }
            defunc_0_lifted_lambda_res_89695 = r_89697;
            // futhark/microgpt.fut:298:5-109
            
            double neg_res_89703 = -defunc_0_lifted_lambda_res_89695;
            
            ((double *) mem_96983)[i_95526] = neg_res_89703;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95530 = 0; i_95530 < (int64_t) 16; i_95530++) {
            // futhark/microgpt.fut:299:35-47
            
            double zt_lhs_89711 = ((double *) mem_96983)[i_95530];
            
            // futhark/microgpt.fut:299:89-101
            
            double zp_lhs_89712 = ((double *) mem_96719)[i_95530];
            
            // futhark/microgpt.fut:299:89-129
            
            double zp_res_89713 = 1.0e-5 + zp_lhs_89712;
            
            // futhark/microgpt.fut:299:81-129
            
            double sqrt_res_89714 = futrts_sqrt64(zp_res_89713);
            
            // futhark/microgpt.fut:299:67-131
            
            double zt_res_89715 = 2.0 * sqrt_res_89714;
            
            // futhark/microgpt.fut:299:53-131
            
            double zs_res_89716 = 1.0 / zt_res_89715;
            
            // futhark/microgpt.fut:299:35-131
            
            double zt_res_89717 = zt_lhs_89711 * zs_res_89716;
            
            ((double *) mem_96990)[i_95530] = zt_res_89717;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95534 = 0; i_95534 < (int64_t) 16; i_95534++) {
            // futhark/microgpt.fut:300:45-57
            
            double zs_lhs_89725 = ((double *) mem_96990)[i_95534];
            
            // futhark/microgpt.fut:300:45-72
            
            double zs_res_89726 = zs_lhs_89725 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_98187 = 0; nest_i_98187 < (int64_t) 16; nest_i_98187++) {
                ((double *) mem_96997)[i_95534 * (int64_t) 16 + nest_i_98187] = zs_res_89726;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95542 = 0; i_95542 < (int64_t) 16; i_95542++) {
            // futhark/microgpt.fut:301:107-119
            
            double zs_rhs_89735 = ((double *) mem_96742)[i_95542];
            
            // futhark/microgpt.fut:301:99-119
            
            double zs_res_89736 = 1.0 / zs_rhs_89735;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95538 = 0; i_95538 < (int64_t) 16; i_95538++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_89743 = ((double *) mem_96903)[i_95542 * (int64_t) 16 + i_95538];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_89744 = ((double *) mem_96967)[i_95542 * (int64_t) 16 + i_95538];
                
                // futhark/microgpt.fut:301:73-119
                
                double zt_res_89745 = zs_res_89736 * zt_lhs_89744;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_89746 = ((double *) mem_96997)[i_95542 * (int64_t) 16 + i_95538];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_89747 = ((double *) mem_96671)[i_95542 * (int64_t) 16 + i_95538];
                
                // futhark/microgpt.fut:301:127-171
                
                double zt_res_89748 = zt_lhs_89746 * zt_rhs_89747;
                
                // futhark/microgpt.fut:301:94-171
                
                double zp_res_89749 = zt_res_89745 + zt_res_89748;
                
                // futhark/microgpt.fut:301:122-223
                
                double zp_res_89750 = zt_res_89748 + zp_res_89749;
                
                // futhark/microgpt.fut:301:45-223
                
                double zp_res_89751 = zp_lhs_89743 + zp_res_89750;
                
                ((double *) mem_97012)[i_95538] = zp_res_89751;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97007, i_95542 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97012, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95555 = 0; i_95555 < (int64_t) 16; i_95555++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95548 = 0; i_95548 < (int64_t) 16; i_95548++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93655;
                double r_93657 = 0.0;
                
                for (int64_t i_93656 = 0; i_93656 < (int64_t) 16; i_93656++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93658 = ((double *) mem_97007)[i_95555 * (int64_t) 16 + i_93656];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93659 = ((double *) mem_param_96148.mem)[i_93656 * (int64_t) 16 + i_95548];
                    
                    // futhark/microgpt.fut:302:67-112
                    
                    double zt_res_93660 = zt_lhs_93658 * zt_rhs_93659;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93661 = r_93657 + zt_res_93660;
                    double r_tmp_98194 = zp_res_93661;
                    
                    r_93657 = r_tmp_98194;
                }
                defunc_0_lifted_lambda_res_93655 = r_93657;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93668;
                double r_93670 = 0.0;
                
                for (int64_t i_93669 = 0; i_93669 < (int64_t) 16; i_93669++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_93671 = ((double *) mem_97007)[i_93669 * (int64_t) 16 + i_95555];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_93672 = ((double *) mem_96633)[i_93669 * (int64_t) 16 + i_95548];
                    
                    // futhark/microgpt.fut:349:68-112
                    
                    double zt_res_93673 = zt_lhs_93671 * zt_rhs_93672;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93674 = r_93670 + zt_res_93673;
                    double r_tmp_98195 = zp_res_93674;
                    
                    r_93670 = r_tmp_98195;
                }
                defunc_0_lifted_lambda_res_93668 = r_93670;
                ((double *) mem_97033)[i_95548] = defunc_0_lifted_lambda_res_93668;
                ((double *) mem_97034)[i_95548] = defunc_0_lifted_lambda_res_93655;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97023, i_95555 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97033, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97024, i_95555 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97034, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95577 = 0; i_95577 < (int64_t) 4; i_95577++) {
            // futhark/microgpt.fut:303:74-77
            
            int64_t zp_lhs_92028 = mul64((int64_t) 4, i_95577);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95570 = 0; i_95570 < (int64_t) 16; i_95570++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95560 = 0; i_95560 < (int64_t) 4; i_95560++) {
                    // futhark/microgpt.fut:303:79-87
                    
                    int64_t tmp_93696 = add64(zp_lhs_92028, i_95560);
                    
                    // futhark/microgpt.fut:303:52-89
                    
                    bool x_93697 = sle64((int64_t) 0, tmp_93696);
                    
                    // futhark/microgpt.fut:303:52-89
                    
                    bool y_93698 = slt64(tmp_93696, (int64_t) 16);
                    
                    // futhark/microgpt.fut:303:52-89
                    
                    bool bounds_check_93699 = x_93697 && y_93698;
                    
                    // futhark/microgpt.fut:303:52-89
                    
                    bool index_certs_93700;
                    
                    if (!bounds_check_93699) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_93696, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:303:52-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:303:13-90\n   #9  futhark/microgpt.fut:469:5-76\n   #10 futhark/microgpt.fut:474:26-480:31\n   #11 futhark/microgpt.fut:496:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_93701 = ((double *) mem_97024)[i_95570 * (int64_t) 16 + tmp_93696];
                    
                    ((double *) mem_97077)[i_95560] = lifted_lambda_res_93701;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95564 = 0; i_95564 < (int64_t) 16; i_95564++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_93715 = ((double *) mem_96523)[i_95577 * (int64_t) 256 + i_95570 * (int64_t) 16 + i_95564];
                    
                    // futhark/microgpt.fut:305:55-97
                    
                    double zs_res_93716 = zs_lhs_93715 / 2.0;
                    double zp_rhs_93717 = ((double *) masks_mem_96134.mem)[step_88717 * (int64_t) 256 + i_95570 * (int64_t) 16 + i_95564];
                    
                    // futhark/microgpt.fut:305:84-123
                    
                    double zp_res_93718 = zs_res_93716 + zp_rhs_93717;
                    
                    ((double *) mem_97084)[i_95564] = zp_res_93718;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97067, i_95570 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97084, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97068, i_95570 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97077, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97055, i_95577 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97067, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97056, i_95577 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_97068, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95612 = 0; i_95612 < (int64_t) 4; i_95612++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95602 = 0; i_95602 < (int64_t) 16; i_95602++) {
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_95027;
                int64_t defunc_0_reduce_res_95028;
                double defunc_0_reduce_res_95029;
                int64_t defunc_0_reduce_res_95030;
                double redout_95581;
                int64_t redout_95582;
                double redout_95583;
                int64_t redout_95584;
                
                redout_95581 = -INFINITY;
                redout_95582 = (int64_t) 16;
                redout_95583 = -INFINITY;
                redout_95584 = (int64_t) 16;
                for (int64_t i_95586 = 0; i_95586 < (int64_t) 16; i_95586++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_94057 = ((double *) mem_97055)[i_95612 * (int64_t) 256 + i_95602 * (int64_t) 16 + i_95586];
                    
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_94069;
                    double r_94071 = 0.0;
                    
                    for (int64_t i_94070 = 0; i_94070 < (int64_t) 4; i_94070++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_94072 = ((double *) mem_97056)[i_95612 * (int64_t) 64 + i_95602 * (int64_t) 4 + i_94070];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_94073 = ((double *) mem_96442)[i_95612 * (int64_t) 64 + i_95586 * (int64_t) 4 + i_94070];
                        
                        // futhark/microgpt.fut:310:75-135
                        
                        double zt_res_94074 = zt_lhs_94072 * zt_rhs_94073;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_94075 = r_94071 + zt_res_94074;
                        double r_tmp_98213 = zp_res_94075;
                        
                        r_94071 = r_tmp_98213;
                    }
                    defunc_0_lifted_lambda_res_94069 = r_94071;
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_93852 = lifted_lambda_res_94057 < redout_95581;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_93853;
                    
                    if (zg_res_93852) {
                        lifted_lambda_res_93853 = redout_95581;
                    } else {
                        lifted_lambda_res_93853 = lifted_lambda_res_94057;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_93854;
                    
                    if (zg_res_93852) {
                        lifted_lambda_res_93854 = redout_95582;
                    } else {
                        lifted_lambda_res_93854 = i_95586;
                    }
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_93931 = lifted_lambda_res_94057 < redout_95583;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_93932;
                    
                    if (zg_res_93931) {
                        lifted_lambda_res_93932 = redout_95583;
                    } else {
                        lifted_lambda_res_93932 = lifted_lambda_res_94057;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_93933;
                    
                    if (zg_res_93931) {
                        lifted_lambda_res_93933 = redout_95584;
                    } else {
                        lifted_lambda_res_93933 = i_95586;
                    }
                    ((double *) mem_97140)[i_95586] = defunc_0_lifted_lambda_res_94069;
                    
                    double redout_tmp_98208 = lifted_lambda_res_93853;
                    int64_t redout_tmp_98209 = lifted_lambda_res_93854;
                    double redout_tmp_98210 = lifted_lambda_res_93932;
                    int64_t redout_tmp_98211 = lifted_lambda_res_93933;
                    
                    redout_95581 = redout_tmp_98208;
                    redout_95582 = redout_tmp_98209;
                    redout_95583 = redout_tmp_98210;
                    redout_95584 = redout_tmp_98211;
                }
                defunc_0_reduce_res_95027 = redout_95581;
                defunc_0_reduce_res_95028 = redout_95582;
                defunc_0_reduce_res_95029 = redout_95583;
                defunc_0_reduce_res_95030 = redout_95584;
                // futhark/microgpt.fut:306:65-143
                
                bool x_93855 = sle64((int64_t) 0, defunc_0_reduce_res_95028);
                
                // futhark/microgpt.fut:306:65-143
                
                bool y_93856 = slt64(defunc_0_reduce_res_95028, (int64_t) 16);
                
                // futhark/microgpt.fut:306:65-143
                
                bool bounds_check_93857 = x_93855 && y_93856;
                
                // futhark/microgpt.fut:306:65-143
                
                bool index_certs_93858;
                
                if (!bounds_check_93857) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_95028, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:306:65-143\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:306:13-309:41\n   #9  futhark/microgpt.fut:469:5-76\n   #10 futhark/microgpt.fut:474:26-480:31\n   #11 futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:311:43-121
                
                bool x_93934 = sle64((int64_t) 0, defunc_0_reduce_res_95030);
                
                // futhark/microgpt.fut:311:43-121
                
                bool y_93935 = slt64(defunc_0_reduce_res_95030, (int64_t) 16);
                
                // futhark/microgpt.fut:311:43-121
                
                bool bounds_check_93936 = x_93934 && y_93935;
                
                // futhark/microgpt.fut:311:43-121
                
                bool index_certs_93937;
                
                if (!bounds_check_93936) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) defunc_0_reduce_res_95030, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:311:43-121\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:311:13-122\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double x154_93859 = ((double *) mem_97055)[i_95612 * (int64_t) 256 + i_95602 * (int64_t) 16 + defunc_0_reduce_res_95028];
                
                // futhark/microgpt.fut:307:80-90
                
                double neg_res_93860 = -x154_93859;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95590 = 0; i_95590 < (int64_t) 16; i_95590++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_93867 = ((double *) mem_97055)[i_95612 * (int64_t) 256 + i_95602 * (int64_t) 16 + i_95590];
                    
                    // futhark/microgpt.fut:307:46-90
                    
                    double zp_res_93868 = neg_res_93860 + zp_lhs_93867;
                    
                    // futhark/microgpt.fut:307:39-90
                    
                    double exp_res_93869 = futrts_exp64(zp_res_93868);
                    
                    ((double *) mem_97147)[i_95590] = exp_res_93869;
                }
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_93871;
                double r_93873 = 0.0;
                
                for (int64_t i_93872 = 0; i_93872 < (int64_t) 16; i_93872++) {
                    // futhark/microgpt.fut:308:38-50
                    
                    double lifted_lambda_res_93874 = ((double *) mem_97147)[i_93872];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_93875 = r_93873 + lifted_lambda_res_93874;
                    double r_tmp_98215 = zp_res_93875;
                    
                    r_93873 = r_tmp_98215;
                }
                defunc_0_lifted_lambda_res_93871 = r_93873;
                // futhark/microgpt.fut:309:23-35
                
                double zs_res_93876 = 1.0 / defunc_0_lifted_lambda_res_93871;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95594 = 0; i_95594 < (int64_t) 16; i_95594++) {
                    // futhark/microgpt.fut:309:5-17
                    
                    double zt_lhs_93883 = ((double *) mem_97147)[i_95594];
                    
                    // futhark/microgpt.fut:309:5-35
                    
                    double zt_res_93884 = zs_res_93876 * zt_lhs_93883;
                    
                    ((double *) mem_97154)[i_95594] = zt_res_93884;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_93938 = ((double *) mem_97055)[i_95612 * (int64_t) 256 + i_95602 * (int64_t) 16 + defunc_0_reduce_res_95030];
                
                ((double *) mem_97126)[i_95602] = lifted_lambda_res_93938;
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97127, i_95602 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97140, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97128, i_95602 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97154, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97109, i_95612 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97126, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97110, i_95612 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97127, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97111, i_95612 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97128, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95635 = 0; i_95635 < (int64_t) 4; i_95635++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95628 = 0; i_95628 < (int64_t) 16; i_95628++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_94133 = ((double *) mem_97109)[i_95635 * (int64_t) 16 + i_95628];
                
                // futhark/microgpt.fut:312:95-121
                
                double neg_res_94134 = -neg_arg0_94133;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95618 = 0; i_95618 < (int64_t) 16; i_95618++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_94141 = ((double *) mem_97055)[i_95635 * (int64_t) 256 + i_95628 * (int64_t) 16 + i_95618];
                    
                    // futhark/microgpt.fut:312:61-121
                    
                    double zp_res_94142 = neg_res_94134 + zp_lhs_94141;
                    
                    // futhark/microgpt.fut:312:54-121
                    
                    double exp_res_94143 = futrts_exp64(zp_res_94142);
                    
                    ((double *) mem_97208)[i_95618] = exp_res_94143;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95622 = 0; i_95622 < (int64_t) 4; i_95622++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_94157;
                    double r_94159 = 0.0;
                    
                    for (int64_t i_94158 = 0; i_94158 < (int64_t) 16; i_94158++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_94160 = ((double *) mem_97056)[i_95635 * (int64_t) 64 + i_94158 * (int64_t) 4 + i_95622];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_94161 = ((double *) mem_97111)[i_95635 * (int64_t) 256 + i_94158 * (int64_t) 16 + i_95628];
                        
                        // futhark/microgpt.fut:320:75-136
                        
                        double zt_res_94162 = zt_lhs_94160 * zt_rhs_94161;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_94163 = r_94159 + zt_res_94162;
                        double r_tmp_98223 = zp_res_94163;
                        
                        r_94159 = r_tmp_98223;
                    }
                    defunc_0_lifted_lambda_res_94157 = r_94159;
                    ((double *) mem_97215)[i_95622] = defunc_0_lifted_lambda_res_94157;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97198, i_95628 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97215, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97199, i_95628 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97208, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97186, i_95635 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_97198, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97187, i_95635 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97199, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95644 = 0; i_95644 < (int64_t) 4; i_95644++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95640 = 0; i_95640 < (int64_t) 16; i_95640++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90016;
                double r_90018 = 0.0;
                
                for (int64_t i_90017 = 0; i_90017 < (int64_t) 16; i_90017++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double lifted_lambda_res_90019 = ((double *) mem_97187)[i_95644 * (int64_t) 256 + i_95640 * (int64_t) 16 + i_90017];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90020 = r_90018 + lifted_lambda_res_90019;
                    double r_tmp_98226 = zp_res_90020;
                    
                    r_90018 = r_tmp_98226;
                }
                defunc_0_lifted_lambda_res_90016 = r_90018;
                ((double *) mem_97245)[i_95640] = defunc_0_lifted_lambda_res_90016;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97240, i_95644 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97245, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95652 = 0; i_95652 < (int64_t) 4; i_95652++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95648 = 0; i_95648 < (int64_t) 16; i_95648++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90035 = ((double *) mem_97240)[i_95652 * (int64_t) 16 + i_95648];
                
                // futhark/microgpt.fut:314:78-123
                
                double zt_res_90036 = zt_lhs_90035 * zt_lhs_90035;
                
                // futhark/microgpt.fut:315:103-123
                
                double zs_res_90037 = 1.0 / zt_res_90036;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90038;
                double r_90040 = 0.0;
                
                for (int64_t i_90039 = 0; i_90039 < (int64_t) 16; i_90039++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90041 = ((double *) mem_97110)[i_95652 * (int64_t) 256 + i_95648 * (int64_t) 16 + i_90039];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90042 = ((double *) mem_97187)[i_95652 * (int64_t) 256 + i_95648 * (int64_t) 16 + i_90039];
                    
                    // futhark/microgpt.fut:315:35-96
                    
                    double zt_res_90043 = zt_lhs_90041 * zt_rhs_90042;
                    
                    // futhark/microgpt.fut:315:64-123
                    
                    double zt_res_90044 = zs_res_90037 * zt_res_90043;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90045 = r_90040 + zt_res_90044;
                    double r_tmp_98229 = zp_res_90045;
                    
                    r_90040 = r_tmp_98229;
                }
                defunc_0_lifted_lambda_res_90038 = r_90040;
                // futhark/microgpt.fut:315:5-126
                
                double neg_res_90046 = -defunc_0_lifted_lambda_res_90038;
                
                ((double *) mem_97261)[i_95648] = neg_res_90046;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97256, i_95652 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97261, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95664 = 0; i_95664 < (int64_t) 4; i_95664++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95660 = 0; i_95660 < (int64_t) 16; i_95660++) {
                // futhark/microgpt.fut:4:11-25
                
                double zs_rhs_90061 = ((double *) mem_97240)[i_95664 * (int64_t) 16 + i_95660];
                
                // futhark/microgpt.fut:316:89-117
                
                double zs_res_90062 = 1.0 / zs_rhs_90061;
                
                // futhark/microgpt.fut:4:11-25
                
                double zp_rhs_90063 = ((double *) mem_97256)[i_95664 * (int64_t) 16 + i_95660];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95656 = 0; i_95656 < (int64_t) 16; i_95656++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_lhs_90070 = ((double *) mem_97110)[i_95664 * (int64_t) 256 + i_95660 * (int64_t) 16 + i_95656];
                    
                    // futhark/microgpt.fut:316:55-117
                    
                    double zt_res_90071 = zs_res_90062 * zt_lhs_90070;
                    
                    // futhark/microgpt.fut:316:84-144
                    
                    double zp_res_90072 = zp_rhs_90063 + zt_res_90071;
                    
                    ((double *) mem_97283)[i_95656] = zp_res_90072;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97278, i_95660 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97283, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97272, i_95664 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97278, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95672 = 0; i_95672 < (int64_t) 4; i_95672++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95668 = 0; i_95668 < (int64_t) 16; i_95668++) {
                double f_elem_90085 = ((double *) mem_97109)[i_95672 * (int64_t) 16 + i_95668];
                
                // futhark/microgpt.fut:317:115-141
                
                double neg_res_90090 = -f_elem_90085;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90091;
                double r_90093 = 0.0;
                
                for (int64_t i_90092 = 0; i_90092 < (int64_t) 16; i_90092++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zp_lhs_90094 = ((double *) mem_97055)[i_95672 * (int64_t) 256 + i_95668 * (int64_t) 16 + i_90092];
                    
                    // futhark/microgpt.fut:317:81-141
                    
                    double zp_res_90095 = neg_res_90090 + zp_lhs_90094;
                    
                    // futhark/microgpt.fut:317:74-141
                    
                    double exp_res_90096 = futrts_exp64(zp_res_90095);
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90097 = ((double *) mem_97272)[i_95672 * (int64_t) 256 + i_95668 * (int64_t) 16 + i_90092];
                    
                    // futhark/microgpt.fut:317:74-177
                    
                    double zt_res_90098 = exp_res_90096 * zt_rhs_90097;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90099 = r_90093 + zt_res_90098;
                    double r_tmp_98235 = zp_res_90099;
                    
                    r_90093 = r_tmp_98235;
                }
                defunc_0_lifted_lambda_res_90091 = r_90093;
                // futhark/microgpt.fut:317:44-179
                
                double neg_res_90100 = -defunc_0_lifted_lambda_res_90091;
                
                ((double *) mem_97304)[i_95668] = neg_res_90100;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97299, i_95672 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97304, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95687 = 0; i_95687 < (int64_t) 4; i_95687++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95683 = 0; i_95683 < (int64_t) 16; i_95683++) {
                // futhark/microgpt.fut:4:11-25
                
                double neg_arg0_90115 = ((double *) mem_97109)[i_95687 * (int64_t) 16 + i_95683];
                
                // futhark/microgpt.fut:318:97-123
                
                double neg_res_90116 = -neg_arg0_90115;
                
                // futhark/microgpt.fut:115:5-117:48
                
                double defunc_0_reduce_res_95055;
                int64_t defunc_0_reduce_res_95056;
                double redout_95674;
                int64_t redout_95675;
                
                redout_95674 = -INFINITY;
                redout_95675 = (int64_t) 16;
                for (int64_t i_95676 = 0; i_95676 < (int64_t) 16; i_95676++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double lifted_lambda_res_94193 = ((double *) mem_97055)[i_95687 * (int64_t) 256 + i_95683 * (int64_t) 16 + i_95676];
                    
                    // futhark/microgpt.fut:116:31-71
                    
                    bool zg_res_90133 = lifted_lambda_res_94193 < redout_95674;
                    
                    // futhark/microgpt.fut:115:5-117:48
                    
                    double lifted_lambda_res_90134;
                    
                    if (zg_res_90133) {
                        lifted_lambda_res_90134 = redout_95674;
                    } else {
                        lifted_lambda_res_90134 = lifted_lambda_res_94193;
                    }
                    // futhark/microgpt.fut:115:5-117:48
                    
                    int64_t lifted_lambda_res_90135;
                    
                    if (zg_res_90133) {
                        lifted_lambda_res_90135 = redout_95675;
                    } else {
                        lifted_lambda_res_90135 = i_95676;
                    }
                    
                    double redout_tmp_98238 = lifted_lambda_res_90134;
                    int64_t redout_tmp_98239 = lifted_lambda_res_90135;
                    
                    redout_95674 = redout_tmp_98238;
                    redout_95675 = redout_tmp_98239;
                }
                defunc_0_reduce_res_95055 = redout_95674;
                defunc_0_reduce_res_95056 = redout_95675;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95679 = 0; i_95679 < (int64_t) 16; i_95679++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zp_lhs_90142 = ((double *) mem_97055)[i_95687 * (int64_t) 256 + i_95683 * (int64_t) 16 + i_95679];
                    
                    // futhark/microgpt.fut:318:63-123
                    
                    double zp_res_90143 = neg_res_90116 + zp_lhs_90142;
                    
                    // futhark/microgpt.fut:318:56-123
                    
                    double exp_res_90144 = futrts_exp64(zp_res_90143);
                    
                    // futhark/microgpt.fut:4:11-25
                    
                    double zt_rhs_90145 = ((double *) mem_97272)[i_95687 * (int64_t) 256 + i_95683 * (int64_t) 16 + i_95679];
                    
                    // futhark/microgpt.fut:318:56-159
                    
                    double zt_res_90146 = exp_res_90144 * zt_rhs_90145;
                    
                    // futhark/microgpt.fut:318:166-275
                    
                    bool cond_90147 = i_95679 == defunc_0_reduce_res_95056;
                    
                    // futhark/microgpt.fut:318:166-275
                    
                    double zp_rhs_90148;
                    
                    if (cond_90147) {
                        // futhark/microgpt.fut:4:11-25
                        
                        double zp_rhs_t_res_95054 = ((double *) mem_97299)[i_95687 * (int64_t) 16 + i_95683];
                        
                        zp_rhs_90148 = zp_rhs_t_res_95054;
                    } else {
                        zp_rhs_90148 = 0.0;
                    }
                    // futhark/microgpt.fut:318:127-275
                    
                    double zp_res_90158 = zt_res_90146 + zp_rhs_90148;
                    
                    ((double *) mem_97326)[i_95679] = zp_res_90158;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97321, i_95683 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97326, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97315, i_95687 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97321, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95699 = 0; i_95699 < (int64_t) 4; i_95699++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95695 = 0; i_95695 < (int64_t) 16; i_95695++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95691 = 0; i_95691 < (int64_t) 16; i_95691++) {
                    // futhark/microgpt.fut:4:11-25
                    
                    double zs_lhs_90180 = ((double *) mem_97315)[i_95699 * (int64_t) 256 + i_95695 * (int64_t) 16 + i_95691];
                    
                    // futhark/microgpt.fut:319:54-96
                    
                    double zs_res_90181 = zs_lhs_90180 / 2.0;
                    
                    ((double *) mem_97353)[i_95691] = zs_res_90181;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97348, i_95695 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97353, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97342, i_95699 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97348, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95719 = 0; i_95719 < (int64_t) 4; i_95719++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95712 = 0; i_95712 < (int64_t) 16; i_95712++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_95705 = 0; i_95705 < (int64_t) 4; i_95705++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_94301;
                    double r_94303 = 0.0;
                    
                    for (int64_t i_94302 = 0; i_94302 < (int64_t) 16; i_94302++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_94304 = ((double *) mem_97342)[i_95719 * (int64_t) 256 + i_94302 * (int64_t) 16 + i_95712];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_94305 = ((double *) mem_96444)[i_95719 * (int64_t) 64 + i_94302 * (int64_t) 4 + i_95705];
                        
                        // futhark/microgpt.fut:321:75-135
                        
                        double zt_res_94306 = zt_lhs_94304 * zt_rhs_94305;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_94307 = r_94303 + zt_res_94306;
                        double r_tmp_98250 = zp_res_94307;
                        
                        r_94303 = r_tmp_98250;
                    }
                    defunc_0_lifted_lambda_res_94301 = r_94303;
                    // futhark/microgpt.fut:71:13-49
                    
                    double defunc_0_lifted_lambda_res_94314;
                    double r_94316 = 0.0;
                    
                    for (int64_t i_94315 = 0; i_94315 < (int64_t) 16; i_94315++) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_lhs_94317 = ((double *) mem_97342)[i_95719 * (int64_t) 256 + i_95712 * (int64_t) 16 + i_94315];
                        
                        // futhark/microgpt.fut:71:46-49
                        
                        double zt_rhs_94318 = ((double *) mem_96443)[i_95719 * (int64_t) 64 + i_94315 * (int64_t) 4 + i_95705];
                        
                        // futhark/microgpt.fut:322:75-135
                        
                        double zt_res_94319 = zt_lhs_94317 * zt_rhs_94318;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        double zp_res_94320 = r_94316 + zt_res_94319;
                        double r_tmp_98251 = zp_res_94320;
                        
                        r_94316 = r_tmp_98251;
                    }
                    defunc_0_lifted_lambda_res_94314 = r_94316;
                    ((double *) mem_97391)[i_95705] = defunc_0_lifted_lambda_res_94314;
                    ((double *) mem_97392)[i_95705] = defunc_0_lifted_lambda_res_94301;
                }
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97381, i_95712 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97391, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_8b(ctx, 1, (uint64_t *) mem_97382, i_95712 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97392, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97369, i_95719 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_97381, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_8b(ctx, 2, (uint64_t *) mem_97370, i_95719 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint64_t *) mem_97382, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95738 = 0; i_95738 < (int64_t) 16; i_95738++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95728 = 0; i_95728 < (int64_t) 16; i_95728++) {
                // futhark/microgpt.fut:323:57-60
                
                int64_t tmp_94383 = sdiv64(i_95728, (int64_t) 4);
                
                // futhark/microgpt.fut:323:44-62
                
                bool x_94384 = sle64((int64_t) 0, tmp_94383);
                
                // futhark/microgpt.fut:323:44-62
                
                bool y_94385 = slt64(tmp_94383, (int64_t) 4);
                
                // futhark/microgpt.fut:323:44-62
                
                bool bounds_check_94386 = x_94384 && y_94385;
                
                // futhark/microgpt.fut:323:44-62
                
                bool index_certs_94387;
                
                if (!bounds_check_94386) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_94383, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:323:44-62\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:323:13-85\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:323:79-82
                
                int64_t tmp_94388 = smod64(i_95728, (int64_t) 4);
                
                // futhark/microgpt.fut:323:44-84
                
                bool x_94389 = sle64((int64_t) 0, tmp_94388);
                
                // futhark/microgpt.fut:323:44-84
                
                bool y_94390 = slt64(tmp_94388, (int64_t) 4);
                
                // futhark/microgpt.fut:323:44-84
                
                bool bounds_check_94391 = x_94389 && y_94390;
                
                // futhark/microgpt.fut:323:44-84
                
                bool index_certs_94392;
                
                if (!bounds_check_94391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_94388, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:323:44-84\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:323:13-85\n   #6  futhark/microgpt.fut:469:5-76\n   #7  futhark/microgpt.fut:474:26-480:31\n   #8  futhark/microgpt.fut:496:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94393 = ((double *) mem_97186)[tmp_94383 * (int64_t) 64 + i_95738 * (int64_t) 4 + tmp_94388];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94406 = ((double *) mem_97370)[tmp_94383 * (int64_t) 64 + i_95738 * (int64_t) 4 + tmp_94388];
                
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94422 = ((double *) mem_97369)[tmp_94383 * (int64_t) 64 + i_95738 * (int64_t) 4 + tmp_94388];
                
                ((double *) mem_97438)[i_95728] = lifted_lambda_res_94422;
                ((double *) mem_97439)[i_95728] = lifted_lambda_res_94406;
                ((double *) mem_97440)[i_95728] = lifted_lambda_res_94393;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97423, i_95738 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97438, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97424, i_95738 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97439, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97425, i_95738 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97440, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95763 = 0; i_95763 < (int64_t) 16; i_95763++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95750 = 0; i_95750 < (int64_t) 16; i_95750++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94585;
                double r_94587 = 0.0;
                
                for (int64_t i_94586 = 0; i_94586 < (int64_t) 16; i_94586++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94588 = ((double *) mem_97425)[i_95763 * (int64_t) 16 + i_94586];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94589 = ((double *) mem_param_96168.mem)[i_94586 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:326:69-114
                    
                    double zt_res_94590 = zt_lhs_94588 * zt_rhs_94589;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94591 = r_94587 + zt_res_94590;
                    double r_tmp_98266 = zp_res_94591;
                    
                    r_94587 = r_tmp_98266;
                }
                defunc_0_lifted_lambda_res_94585 = r_94587;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94592;
                double r_94594 = 0.0;
                
                for (int64_t i_94593 = 0; i_94593 < (int64_t) 16; i_94593++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94595 = ((double *) mem_97424)[i_95763 * (int64_t) 16 + i_94593];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94596 = ((double *) mem_param_96144.mem)[i_94593 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:326:145-190
                    
                    double zt_res_94597 = zt_lhs_94595 * zt_rhs_94596;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94598 = r_94594 + zt_res_94597;
                    double r_tmp_98267 = zp_res_94598;
                    
                    r_94594 = r_tmp_98267;
                }
                defunc_0_lifted_lambda_res_94592 = r_94594;
                // futhark/microgpt.fut:326:47-192
                
                double zp_res_94599 = defunc_0_lifted_lambda_res_94585 + defunc_0_lifted_lambda_res_94592;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94600;
                double r_94602 = 0.0;
                
                for (int64_t i_94601 = 0; i_94601 < (int64_t) 16; i_94601++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94603 = ((double *) mem_97423)[i_95763 * (int64_t) 16 + i_94601];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94604 = ((double *) mem_param_96156.mem)[i_94601 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:326:222-267
                    
                    double zt_res_94605 = zt_lhs_94603 * zt_rhs_94604;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94606 = r_94602 + zt_res_94605;
                    double r_tmp_98268 = zp_res_94606;
                    
                    r_94602 = r_tmp_98268;
                }
                defunc_0_lifted_lambda_res_94600 = r_94602;
                // futhark/microgpt.fut:326:118-269
                
                double zp_res_94607 = zp_res_94599 + defunc_0_lifted_lambda_res_94600;
                
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94614;
                double r_94616 = 0.0;
                
                for (int64_t i_94615 = 0; i_94615 < (int64_t) 16; i_94615++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94617 = ((double *) mem_97423)[i_94615 * (int64_t) 16 + i_95763];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94618 = ((double *) mem_96343)[i_94615 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:346:68-111
                    
                    double zt_res_94619 = zt_lhs_94617 * zt_rhs_94618;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94620 = r_94616 + zt_res_94619;
                    double r_tmp_98269 = zp_res_94620;
                    
                    r_94616 = r_tmp_98269;
                }
                defunc_0_lifted_lambda_res_94614 = r_94616;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94630;
                double r_94632 = 0.0;
                
                for (int64_t i_94631 = 0; i_94631 < (int64_t) 16; i_94631++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94633 = ((double *) mem_97424)[i_94631 * (int64_t) 16 + i_95763];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94634 = ((double *) mem_96343)[i_94631 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:347:68-111
                    
                    double zt_res_94635 = zt_lhs_94633 * zt_rhs_94634;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94636 = r_94632 + zt_res_94635;
                    double r_tmp_98270 = zp_res_94636;
                    
                    r_94632 = r_tmp_98270;
                }
                defunc_0_lifted_lambda_res_94630 = r_94632;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94648;
                double r_94650 = 0.0;
                
                for (int64_t i_94649 = 0; i_94649 < (int64_t) 16; i_94649++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94651 = ((double *) mem_97425)[i_94649 * (int64_t) 16 + i_95763];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94652 = ((double *) mem_96343)[i_94649 * (int64_t) 16 + i_95750];
                    
                    // futhark/microgpt.fut:348:68-111
                    
                    double zt_res_94653 = zt_lhs_94651 * zt_rhs_94652;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94654 = r_94650 + zt_res_94653;
                    double r_tmp_98271 = zp_res_94654;
                    
                    r_94650 = r_tmp_98271;
                }
                defunc_0_lifted_lambda_res_94648 = r_94650;
                ((double *) mem_97491)[i_95750] = defunc_0_lifted_lambda_res_94648;
                ((double *) mem_97492)[i_95750] = defunc_0_lifted_lambda_res_94630;
                ((double *) mem_97493)[i_95750] = defunc_0_lifted_lambda_res_94614;
                ((double *) mem_97494)[i_95750] = zp_res_94607;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97471, i_95763 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97491, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97472, i_95763 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97492, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97473, i_95763 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97493, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97474, i_95763 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97494, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95770 = 0; i_95770 < (int64_t) 16; i_95770++) {
            // futhark/microgpt.fut:330:69-81
            
            double zt_lhs_90414 = ((double *) mem_96632)[i_95770];
            
            // futhark/microgpt.fut:330:69-98
            
            double zt_res_90415 = zt_lhs_90414 * zt_lhs_90414;
            
            // futhark/microgpt.fut:331:85-105
            
            double zs_res_90416 = 1.0 / zt_res_90415;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90417;
            double r_90419 = 0.0;
            
            for (int64_t i_90418 = 0; i_90418 < (int64_t) 16; i_90418++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_90420 = ((double *) mem_97474)[i_95770 * (int64_t) 16 + i_90418];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_90421 = ((double *) mem_96310)[i_95770 * (int64_t) 16 + i_90418];
                
                // futhark/microgpt.fut:331:35-78
                
                double zt_res_90422 = zt_lhs_90420 * zt_rhs_90421;
                
                // futhark/microgpt.fut:331:56-105
                
                double zt_res_90423 = zs_res_90416 * zt_res_90422;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90424 = r_90419 + zt_res_90423;
                double r_tmp_98273 = zp_res_90424;
                
                r_90419 = r_tmp_98273;
            }
            defunc_0_lifted_lambda_res_90417 = r_90419;
            // futhark/microgpt.fut:331:5-108
            
            double neg_res_90425 = -defunc_0_lifted_lambda_res_90417;
            
            ((double *) mem_97535)[i_95770] = neg_res_90425;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95774 = 0; i_95774 < (int64_t) 16; i_95774++) {
            // futhark/microgpt.fut:332:35-47
            
            double zt_lhs_90433 = ((double *) mem_97535)[i_95774];
            
            // futhark/microgpt.fut:332:89-101
            
            double zp_lhs_90434 = ((double *) mem_96381)[i_95774];
            
            // futhark/microgpt.fut:332:89-129
            
            double zp_res_90435 = 1.0e-5 + zp_lhs_90434;
            
            // futhark/microgpt.fut:332:81-129
            
            double sqrt_res_90436 = futrts_sqrt64(zp_res_90435);
            
            // futhark/microgpt.fut:332:67-131
            
            double zt_res_90437 = 2.0 * sqrt_res_90436;
            
            // futhark/microgpt.fut:332:53-131
            
            double zs_res_90438 = 1.0 / zt_res_90437;
            
            // futhark/microgpt.fut:332:35-131
            
            double zt_res_90439 = zt_lhs_90433 * zs_res_90438;
            
            ((double *) mem_97542)[i_95774] = zt_res_90439;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95778 = 0; i_95778 < (int64_t) 16; i_95778++) {
            // futhark/microgpt.fut:333:45-57
            
            double zs_lhs_90447 = ((double *) mem_97542)[i_95778];
            
            // futhark/microgpt.fut:333:45-72
            
            double zs_res_90448 = zs_lhs_90447 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_98276 = 0; nest_i_98276 < (int64_t) 16; nest_i_98276++) {
                ((double *) mem_97549)[i_95778 * (int64_t) 16 + nest_i_98276] = zs_res_90448;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95786 = 0; i_95786 < (int64_t) 16; i_95786++) {
            // futhark/microgpt.fut:334:107-119
            
            double zs_rhs_90457 = ((double *) mem_96632)[i_95786];
            
            // futhark/microgpt.fut:334:99-119
            
            double zs_res_90458 = 1.0 / zs_rhs_90457;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95782 = 0; i_95782 < (int64_t) 16; i_95782++) {
                // futhark/microgpt.fut:4:11-25
                
                double zp_lhs_90465 = ((double *) mem_97007)[i_95786 * (int64_t) 16 + i_95782];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90466 = ((double *) mem_97474)[i_95786 * (int64_t) 16 + i_95782];
                
                // futhark/microgpt.fut:334:73-119
                
                double zt_res_90467 = zs_res_90458 * zt_lhs_90466;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90468 = ((double *) mem_97549)[i_95786 * (int64_t) 16 + i_95782];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90469 = ((double *) mem_96310)[i_95786 * (int64_t) 16 + i_95782];
                
                // futhark/microgpt.fut:334:127-170
                
                double zt_res_90470 = zt_lhs_90468 * zt_rhs_90469;
                
                // futhark/microgpt.fut:334:94-170
                
                double zp_res_90471 = zt_res_90467 + zt_res_90470;
                
                // futhark/microgpt.fut:334:122-221
                
                double zp_res_90472 = zt_res_90470 + zp_res_90471;
                
                // futhark/microgpt.fut:334:45-221
                
                double zp_res_90473 = zp_lhs_90465 + zp_res_90472;
                
                ((double *) mem_97564)[i_95782] = zp_res_90473;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97559, i_95786 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97564, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95790 = 0; i_95790 < (int64_t) 16; i_95790++) {
            // futhark/microgpt.fut:338:69-81
            
            double zt_lhs_90521 = ((double *) mem_96380)[i_95790];
            
            // futhark/microgpt.fut:338:69-98
            
            double zt_res_90522 = zt_lhs_90521 * zt_lhs_90521;
            
            // futhark/microgpt.fut:339:85-105
            
            double zs_res_90523 = 1.0 / zt_res_90522;
            
            // futhark/microgpt.fut:71:13-49
            
            double defunc_0_lifted_lambda_res_90524;
            double r_90526 = 0.0;
            
            for (int64_t i_90525 = 0; i_90525 < (int64_t) 16; i_90525++) {
                // futhark/microgpt.fut:71:46-49
                
                double zt_lhs_90527 = ((double *) mem_97559)[i_95790 * (int64_t) 16 + i_90525];
                
                // futhark/microgpt.fut:71:46-49
                
                double zt_rhs_90528 = ((double *) mem_96278)[i_95790 * (int64_t) 16 + i_90525];
                
                // futhark/microgpt.fut:339:35-78
                
                double zt_res_90529 = zt_lhs_90527 * zt_rhs_90528;
                
                // futhark/microgpt.fut:339:56-105
                
                double zt_res_90530 = zs_res_90523 * zt_res_90529;
                
                // futhark/microgpt.fut:71:40-49
                
                double zp_res_90531 = r_90526 + zt_res_90530;
                double r_tmp_98280 = zp_res_90531;
                
                r_90526 = r_tmp_98280;
            }
            defunc_0_lifted_lambda_res_90524 = r_90526;
            // futhark/microgpt.fut:339:5-108
            
            double neg_res_90532 = -defunc_0_lifted_lambda_res_90524;
            
            ((double *) mem_97575)[i_95790] = neg_res_90532;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95794 = 0; i_95794 < (int64_t) 16; i_95794++) {
            // futhark/microgpt.fut:340:35-47
            
            double zt_lhs_90540 = ((double *) mem_97575)[i_95794];
            
            // futhark/microgpt.fut:340:89-101
            
            double zp_lhs_90541 = ((double *) mem_96341)[i_95794];
            
            // futhark/microgpt.fut:340:89-129
            
            double zp_res_90542 = 1.0e-5 + zp_lhs_90541;
            
            // futhark/microgpt.fut:340:81-129
            
            double sqrt_res_90543 = futrts_sqrt64(zp_res_90542);
            
            // futhark/microgpt.fut:340:67-131
            
            double zt_res_90544 = 2.0 * sqrt_res_90543;
            
            // futhark/microgpt.fut:340:53-131
            
            double zs_res_90545 = 1.0 / zt_res_90544;
            
            // futhark/microgpt.fut:340:35-131
            
            double zt_res_90546 = zt_lhs_90540 * zs_res_90545;
            
            ((double *) mem_97582)[i_95794] = zt_res_90546;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95798 = 0; i_95798 < (int64_t) 16; i_95798++) {
            // futhark/microgpt.fut:341:45-57
            
            double zs_lhs_90554 = ((double *) mem_97582)[i_95798];
            
            // futhark/microgpt.fut:341:45-72
            
            double zs_res_90555 = zs_lhs_90554 / 16.0;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_98283 = 0; nest_i_98283 < (int64_t) 16; nest_i_98283++) {
                ((double *) mem_97589)[i_95798 * (int64_t) 16 + nest_i_98283] = zs_res_90555;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95806 = 0; i_95806 < (int64_t) 16; i_95806++) {
            // futhark/microgpt.fut:342:81-93
            
            double zs_rhs_90564 = ((double *) mem_96380)[i_95806];
            
            // futhark/microgpt.fut:342:73-93
            
            double zs_res_90565 = 1.0 / zs_rhs_90564;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95802 = 0; i_95802 < (int64_t) 16; i_95802++) {
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90572 = ((double *) mem_97559)[i_95806 * (int64_t) 16 + i_95802];
                
                // futhark/microgpt.fut:342:47-93
                
                double zt_res_90573 = zs_res_90565 * zt_lhs_90572;
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_lhs_90574 = ((double *) mem_97589)[i_95806 * (int64_t) 16 + i_95802];
                
                // futhark/microgpt.fut:4:11-25
                
                double zt_rhs_90575 = ((double *) mem_96278)[i_95806 * (int64_t) 16 + i_95802];
                
                // futhark/microgpt.fut:342:101-144
                
                double zt_res_90576 = zt_lhs_90574 * zt_rhs_90575;
                
                // futhark/microgpt.fut:342:68-144
                
                double zp_res_90577 = zt_res_90573 + zt_res_90576;
                
                // futhark/microgpt.fut:342:96-195
                
                double zp_res_90578 = zt_res_90576 + zp_res_90577;
                
                ((double *) mem_97604)[i_95802] = zp_res_90578;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97599, i_95806 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97604, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95819 = 0; i_95819 < (int64_t) 16; i_95819++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95812 = 0; i_95812 < (int64_t) 16; i_95812++) {
                // futhark/microgpt.fut:4:11-25
                
                double lifted_lambda_res_94680 = ((double *) mem_97599)[i_95819 * (int64_t) 16 + i_95812];
                
                ((double *) mem_97625)[i_95812] = lifted_lambda_res_94680;
                ((double *) mem_97626)[i_95812] = lifted_lambda_res_94680;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97615, i_95819 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97625, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97616, i_95819 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95828 = 0; i_95828 < (int64_t) 64; i_95828++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95824 = 0; i_95824 < (int64_t) 16; i_95824++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_90692;
                double r_90694 = 0.0;
                
                for (int64_t i_90693 = 0; i_90693 < (int64_t) 16; i_90693++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_90695 = ((double *) mem_96951)[i_90693 * (int64_t) 64 + i_95828];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_90696 = ((double *) mem_96688)[i_90693 * (int64_t) 16 + i_95824];
                    
                    // futhark/microgpt.fut:350:67-111
                    
                    double zt_res_90697 = zt_lhs_90695 * zt_rhs_90696;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_90698 = r_90694 + zt_res_90697;
                    double r_tmp_98292 = zp_res_90698;
                    
                    r_90694 = r_tmp_98292;
                }
                defunc_0_lifted_lambda_res_90692 = r_90694;
                ((double *) mem_97652)[i_95824] = defunc_0_lifted_lambda_res_90692;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97647, i_95828 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97652, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_95841 = 0; i_95841 < (int64_t) 27; i_95841++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_95834 = 0; i_95834 < (int64_t) 16; i_95834++) {
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94708;
                double r_94710 = 0.0;
                
                for (int64_t i_94709 = 0; i_94709 < (int64_t) 16; i_94709++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_lhs_94711 = ((double *) mem_96887)[i_94709 * (int64_t) 27 + i_95841];
                    
                    // futhark/microgpt.fut:71:46-49
                    
                    double zt_rhs_94712 = ((double *) mem_96781)[i_94709 * (int64_t) 16 + i_95834];
                    
                    // futhark/microgpt.fut:352:68-112
                    
                    double zt_res_94713 = zt_lhs_94711 * zt_rhs_94712;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94714 = r_94710 + zt_res_94713;
                    double r_tmp_98297 = zp_res_94714;
                    
                    r_94710 = r_tmp_98297;
                }
                defunc_0_lifted_lambda_res_94708 = r_94710;
                // futhark/microgpt.fut:71:13-49
                
                double defunc_0_lifted_lambda_res_94717;
                double r_94719 = 0.0;
                
                for (int64_t i_94718 = 0; i_94718 < (int64_t) 16; i_94718++) {
                    int64_t zeze_lhs_94720 = ((int64_t *) seqs_mem_96136.mem)[step_88717 * (int64_t) 16 + i_94718];
                    
                    // futhark/microgpt.fut:470:58-109
                    
                    bool cond_94721 = zeze_lhs_94720 == i_95841;
                    
                    // futhark/microgpt.fut:470:58-109
                    
                    double lifted_lambda_res_94722;
                    
                    if (cond_94721) {
                        // futhark/microgpt.fut:71:46-49
                        
                        double lifted_lambda_res_t_res_95081 = ((double *) mem_97615)[i_94718 * (int64_t) 16 + i_95834];
                        
                        lifted_lambda_res_94722 = lifted_lambda_res_t_res_95081;
                    } else {
                        lifted_lambda_res_94722 = 0.0;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    double zp_res_94728 = r_94719 + lifted_lambda_res_94722;
                    double r_tmp_98298 = zp_res_94728;
                    
                    r_94719 = r_tmp_98298;
                }
                defunc_0_lifted_lambda_res_94717 = r_94719;
                ((double *) mem_97673)[i_95834] = defunc_0_lifted_lambda_res_94717;
                ((double *) mem_97674)[i_95834] = defunc_0_lifted_lambda_res_94708;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97663, i_95841 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97673, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_97664, i_95841 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_97674, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        double i64_res_90797 = sitofp_i64_f64(step_88717);
        
        // futhark/microgpt.fut:426:46-65
        
        double zm_rhs_90798 = i64_res_90797 / 500.0;
        
        // futhark/microgpt.fut:426:24-65
        
        double zt_rhs_90799 = 1.0 - zm_rhs_90798;
        
        // futhark/microgpt.fut:426:19-65
        
        double lt_r_90800 = 1.0e-2 * zt_rhs_90799;
        
        // futhark/microgpt.fut:428:5-52
        if (memblock_alloc(ctx, &mem_97695, (int64_t) 3456, "mem_97695")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-52
        // futhark/microgpt.fut:428:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97695.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96160.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-52
        if (memblock_alloc(ctx, &mem_97697, (int64_t) 3456, "mem_97697")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-52
        // futhark/microgpt.fut:428:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97697.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96196.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-52
        if (memblock_alloc(ctx, &mem_97699, (int64_t) 3456, "mem_97699")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-52
        // futhark/microgpt.fut:428:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97699.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96232.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-52
        if (memblock_alloc(ctx, &mem_97701, (int64_t) 3456, "mem_97701")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:428:5-52
        // futhark/microgpt.fut:428:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97701.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97663, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:428:5-52
        if (futrts_adam_opt_w_10524(ctx, &ext_mem_97705, &ext_mem_97704, &ext_mem_97703, mem_97695, mem_97697, mem_97699, mem_97701, (int64_t) 27, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97695, "mem_97695") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97697, "mem_97697") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97699, "mem_97699") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97701, "mem_97701") != 0)
            return 1;
        // futhark/microgpt.fut:430:5-52
        if (memblock_alloc(ctx, &mem_97706, (int64_t) 2048, "mem_97706")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:430:5-52
        // futhark/microgpt.fut:430:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97706.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96152.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:430:5-52
        if (memblock_alloc(ctx, &mem_97708, (int64_t) 2048, "mem_97708")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:430:5-52
        // futhark/microgpt.fut:430:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97708.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96188.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:430:5-52
        if (memblock_alloc(ctx, &mem_97710, (int64_t) 2048, "mem_97710")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:430:5-52
        // futhark/microgpt.fut:430:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97710.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96224.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:430:5-52
        if (memblock_alloc(ctx, &mem_97712, (int64_t) 2048, "mem_97712")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:430:5-52
        // futhark/microgpt.fut:430:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97712.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97616, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:430:5-52
        if (futrts_adam_opt_w_10525(ctx, &ext_mem_97716, &ext_mem_97715, &ext_mem_97714, mem_97706, mem_97708, mem_97710, mem_97712, (int64_t) 16, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97706, "mem_97706") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97708, "mem_97708") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97710, "mem_97710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97712, "mem_97712") != 0)
            return 1;
        // futhark/microgpt.fut:432:5-56
        if (memblock_alloc(ctx, &mem_97717, (int64_t) 2048, "mem_97717")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:432:5-56
        // futhark/microgpt.fut:432:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97717.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96156.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:432:5-56
        if (memblock_alloc(ctx, &mem_97719, (int64_t) 2048, "mem_97719")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:432:5-56
        // futhark/microgpt.fut:432:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97719.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96192.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:432:5-56
        if (memblock_alloc(ctx, &mem_97721, (int64_t) 2048, "mem_97721")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:432:5-56
        // futhark/microgpt.fut:432:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97721.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96228.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:432:5-56
        if (memblock_alloc(ctx, &mem_97723, (int64_t) 2048, "mem_97723")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:432:5-56
        // futhark/microgpt.fut:432:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97723.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97473, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:432:5-56
        if (futrts_adam_opt_w_10525(ctx, &ext_mem_97727, &ext_mem_97726, &ext_mem_97725, mem_97717, mem_97719, mem_97721, mem_97723, (int64_t) 16, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97717, "mem_97717") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97719, "mem_97719") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97721, "mem_97721") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97723, "mem_97723") != 0)
            return 1;
        // futhark/microgpt.fut:434:5-56
        if (memblock_alloc(ctx, &mem_97728, (int64_t) 2048, "mem_97728")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:434:5-56
        // futhark/microgpt.fut:434:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97728.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96144.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:434:5-56
        if (memblock_alloc(ctx, &mem_97730, (int64_t) 2048, "mem_97730")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:434:5-56
        // futhark/microgpt.fut:434:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97730.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96180.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:434:5-56
        if (memblock_alloc(ctx, &mem_97732, (int64_t) 2048, "mem_97732")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:434:5-56
        // futhark/microgpt.fut:434:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97732.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96216.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:434:5-56
        if (memblock_alloc(ctx, &mem_97734, (int64_t) 2048, "mem_97734")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:434:5-56
        // futhark/microgpt.fut:434:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97734.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97472, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:434:5-56
        if (futrts_adam_opt_w_10525(ctx, &ext_mem_97738, &ext_mem_97737, &ext_mem_97736, mem_97728, mem_97730, mem_97732, mem_97734, (int64_t) 16, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97728, "mem_97728") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97730, "mem_97730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97732, "mem_97732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97734, "mem_97734") != 0)
            return 1;
        // futhark/microgpt.fut:436:5-56
        if (memblock_alloc(ctx, &mem_97739, (int64_t) 2048, "mem_97739")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:436:5-56
        // futhark/microgpt.fut:436:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97739.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96168.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:436:5-56
        if (memblock_alloc(ctx, &mem_97741, (int64_t) 2048, "mem_97741")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:436:5-56
        // futhark/microgpt.fut:436:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97741.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96204.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:436:5-56
        if (memblock_alloc(ctx, &mem_97743, (int64_t) 2048, "mem_97743")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:436:5-56
        // futhark/microgpt.fut:436:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97743.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96240.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:436:5-56
        if (memblock_alloc(ctx, &mem_97745, (int64_t) 2048, "mem_97745")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:436:5-56
        // futhark/microgpt.fut:436:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97745.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97471, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:436:5-56
        if (futrts_adam_opt_w_10525(ctx, &ext_mem_97749, &ext_mem_97748, &ext_mem_97747, mem_97739, mem_97741, mem_97743, mem_97745, (int64_t) 16, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97739, "mem_97739") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97741, "mem_97741") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97743, "mem_97743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97745, "mem_97745") != 0)
            return 1;
        // futhark/microgpt.fut:438:5-56
        if (memblock_alloc(ctx, &mem_97750, (int64_t) 2048, "mem_97750")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:438:5-56
        // futhark/microgpt.fut:438:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97750.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96148.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:438:5-56
        if (memblock_alloc(ctx, &mem_97752, (int64_t) 2048, "mem_97752")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:438:5-56
        // futhark/microgpt.fut:438:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97752.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96184.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:438:5-56
        if (memblock_alloc(ctx, &mem_97754, (int64_t) 2048, "mem_97754")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:438:5-56
        // futhark/microgpt.fut:438:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97754.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96220.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:438:5-56
        if (memblock_alloc(ctx, &mem_97756, (int64_t) 2048, "mem_97756")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:438:5-56
        // futhark/microgpt.fut:438:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97756.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97023, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:438:5-56
        if (futrts_adam_opt_w_10525(ctx, &ext_mem_97760, &ext_mem_97759, &ext_mem_97758, mem_97750, mem_97752, mem_97754, mem_97756, (int64_t) 16, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97750, "mem_97750") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97752, "mem_97752") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97754, "mem_97754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97756, "mem_97756") != 0)
            return 1;
        // futhark/microgpt.fut:440:5-52
        if (memblock_alloc(ctx, &mem_97761, (int64_t) 8192, "mem_97761")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:440:5-52
        // futhark/microgpt.fut:440:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97761.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96164.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:440:5-52
        if (memblock_alloc(ctx, &mem_97763, (int64_t) 8192, "mem_97763")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:440:5-52
        // futhark/microgpt.fut:440:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97763.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96200.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:440:5-52
        if (memblock_alloc(ctx, &mem_97765, (int64_t) 8192, "mem_97765")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:440:5-52
        // futhark/microgpt.fut:440:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97765.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96236.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:440:5-52
        if (memblock_alloc(ctx, &mem_97767, (int64_t) 8192, "mem_97767")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:440:5-52
        // futhark/microgpt.fut:440:5-52
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97767.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97647, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:440:5-52
        if (futrts_adam_opt_w_10524(ctx, &ext_mem_97771, &ext_mem_97770, &ext_mem_97769, mem_97761, mem_97763, mem_97765, mem_97767, (int64_t) 64, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97761, "mem_97761") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97763, "mem_97763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97765, "mem_97765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97767, "mem_97767") != 0)
            return 1;
        // futhark/microgpt.fut:442:5-60
        if (memblock_alloc(ctx, &mem_97772, (int64_t) 8192, "mem_97772")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:442:5-60
        // futhark/microgpt.fut:442:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97772.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_96140.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:442:5-60
        if (memblock_alloc(ctx, &mem_97774, (int64_t) 8192, "mem_97774")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:442:5-60
        // futhark/microgpt.fut:442:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97774.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_96176.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:442:5-60
        if (memblock_alloc(ctx, &mem_97776, (int64_t) 8192, "mem_97776")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:442:5-60
        // futhark/microgpt.fut:442:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97776.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_param_96212.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:442:5-60
        if (memblock_alloc(ctx, &mem_97778, (int64_t) 8192, "mem_97778")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:442:5-60
        // futhark/microgpt.fut:442:5-60
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97778.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint64_t *) mem_96919, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:442:5-60
        if (futrts_adam_opt_w_10524(ctx, &ext_mem_97782, &ext_mem_97781, &ext_mem_97780, mem_97772, mem_97774, mem_97776, mem_97778, (int64_t) 16, (int64_t) 64, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97772, "mem_97772") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97774, "mem_97774") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97776, "mem_97776") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97778, "mem_97778") != 0)
            return 1;
        // futhark/microgpt.fut:444:5-56
        if (memblock_alloc(ctx, &mem_97783, (int64_t) 3456, "mem_97783")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:444:5-56
        // futhark/microgpt.fut:444:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97783.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96172.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:444:5-56
        if (memblock_alloc(ctx, &mem_97785, (int64_t) 3456, "mem_97785")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:444:5-56
        // futhark/microgpt.fut:444:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97785.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96208.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:444:5-56
        if (memblock_alloc(ctx, &mem_97787, (int64_t) 3456, "mem_97787")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:444:5-56
        // futhark/microgpt.fut:444:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97787.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_param_96244.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:444:5-56
        if (memblock_alloc(ctx, &mem_97789, (int64_t) 3456, "mem_97789")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:444:5-56
        // futhark/microgpt.fut:444:5-56
        lmad_copy_8b(ctx, 2, (uint64_t *) mem_97789.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint64_t *) mem_97664, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:444:5-56
        if (futrts_adam_opt_w_10524(ctx, &ext_mem_97793, &ext_mem_97792, &ext_mem_97791, mem_97783, mem_97785, mem_97787, mem_97789, (int64_t) 27, (int64_t) 16, step_88717, lt_r_90800) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_97783, "mem_97783") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97785, "mem_97785") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97787, "mem_97787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97789, "mem_97789") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98001, &ext_mem_97782, "ext_mem_97782") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98002, &ext_mem_97738, "ext_mem_97738") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98003, &ext_mem_97760, "ext_mem_97760") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98004, &ext_mem_97716, "ext_mem_97716") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98005, &ext_mem_97727, "ext_mem_97727") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98006, &ext_mem_97705, "ext_mem_97705") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98007, &ext_mem_97771, "ext_mem_97771") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98008, &ext_mem_97749, "ext_mem_97749") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98009, &ext_mem_97793, "ext_mem_97793") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98010, &ext_mem_97781, "ext_mem_97781") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98011, &ext_mem_97737, "ext_mem_97737") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98012, &ext_mem_97759, "ext_mem_97759") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98013, &ext_mem_97715, "ext_mem_97715") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98014, &ext_mem_97726, "ext_mem_97726") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98015, &ext_mem_97704, "ext_mem_97704") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98016, &ext_mem_97770, "ext_mem_97770") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98017, &ext_mem_97748, "ext_mem_97748") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98018, &ext_mem_97792, "ext_mem_97792") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98019, &ext_mem_97780, "ext_mem_97780") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98020, &ext_mem_97736, "ext_mem_97736") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98021, &ext_mem_97758, "ext_mem_97758") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98022, &ext_mem_97714, "ext_mem_97714") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98023, &ext_mem_97725, "ext_mem_97725") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98024, &ext_mem_97703, "ext_mem_97703") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98025, &ext_mem_97769, "ext_mem_97769") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98026, &ext_mem_97747, "ext_mem_97747") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_98027, &ext_mem_97791, "ext_mem_97791") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96140, &mem_param_tmp_98001, "mem_param_tmp_98001") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96144, &mem_param_tmp_98002, "mem_param_tmp_98002") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96148, &mem_param_tmp_98003, "mem_param_tmp_98003") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96152, &mem_param_tmp_98004, "mem_param_tmp_98004") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96156, &mem_param_tmp_98005, "mem_param_tmp_98005") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96160, &mem_param_tmp_98006, "mem_param_tmp_98006") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96164, &mem_param_tmp_98007, "mem_param_tmp_98007") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96168, &mem_param_tmp_98008, "mem_param_tmp_98008") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96172, &mem_param_tmp_98009, "mem_param_tmp_98009") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96176, &mem_param_tmp_98010, "mem_param_tmp_98010") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96180, &mem_param_tmp_98011, "mem_param_tmp_98011") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96184, &mem_param_tmp_98012, "mem_param_tmp_98012") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96188, &mem_param_tmp_98013, "mem_param_tmp_98013") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96192, &mem_param_tmp_98014, "mem_param_tmp_98014") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96196, &mem_param_tmp_98015, "mem_param_tmp_98015") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96200, &mem_param_tmp_98016, "mem_param_tmp_98016") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96204, &mem_param_tmp_98017, "mem_param_tmp_98017") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96208, &mem_param_tmp_98018, "mem_param_tmp_98018") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96212, &mem_param_tmp_98019, "mem_param_tmp_98019") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96216, &mem_param_tmp_98020, "mem_param_tmp_98020") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96220, &mem_param_tmp_98021, "mem_param_tmp_98021") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96224, &mem_param_tmp_98022, "mem_param_tmp_98022") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96228, &mem_param_tmp_98023, "mem_param_tmp_98023") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96232, &mem_param_tmp_98024, "mem_param_tmp_98024") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96236, &mem_param_tmp_98025, "mem_param_tmp_98025") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96240, &mem_param_tmp_98026, "mem_param_tmp_98026") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_96244, &mem_param_tmp_98027, "mem_param_tmp_98027") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_97901, &mem_param_96140, "mem_param_96140") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97900, &mem_param_96144, "mem_param_96144") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97899, &mem_param_96148, "mem_param_96148") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97898, &mem_param_96152, "mem_param_96152") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97897, &mem_param_96156, "mem_param_96156") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97896, &mem_param_96160, "mem_param_96160") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97895, &mem_param_96164, "mem_param_96164") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97894, &mem_param_96168, "mem_param_96168") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97893, &mem_param_96172, "mem_param_96172") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97892, &mem_param_96176, "mem_param_96176") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97891, &mem_param_96180, "mem_param_96180") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97890, &mem_param_96184, "mem_param_96184") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97889, &mem_param_96188, "mem_param_96188") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97888, &mem_param_96192, "mem_param_96192") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97887, &mem_param_96196, "mem_param_96196") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97886, &mem_param_96200, "mem_param_96200") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97885, &mem_param_96204, "mem_param_96204") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97884, &mem_param_96208, "mem_param_96208") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97883, &mem_param_96212, "mem_param_96212") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97882, &mem_param_96216, "mem_param_96216") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97881, &mem_param_96220, "mem_param_96220") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97880, &mem_param_96224, "mem_param_96224") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97879, &mem_param_96228, "mem_param_96228") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97878, &mem_param_96232, "mem_param_96232") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97877, &mem_param_96236, "mem_param_96236") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97876, &mem_param_96240, "mem_param_96240") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_97875, &mem_param_96244, "mem_param_96244") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97974, &ext_mem_97896, "ext_mem_97896") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97975, &ext_mem_97898, "ext_mem_97898") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97976, &ext_mem_97897, "ext_mem_97897") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97977, &ext_mem_97900, "ext_mem_97900") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97978, &ext_mem_97894, "ext_mem_97894") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97979, &ext_mem_97899, "ext_mem_97899") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97980, &ext_mem_97895, "ext_mem_97895") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97981, &ext_mem_97901, "ext_mem_97901") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97982, &ext_mem_97893, "ext_mem_97893") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97983, &ext_mem_97887, "ext_mem_97887") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97984, &ext_mem_97889, "ext_mem_97889") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97985, &ext_mem_97888, "ext_mem_97888") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97986, &ext_mem_97891, "ext_mem_97891") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97987, &ext_mem_97885, "ext_mem_97885") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97988, &ext_mem_97890, "ext_mem_97890") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97989, &ext_mem_97886, "ext_mem_97886") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97990, &ext_mem_97892, "ext_mem_97892") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97991, &ext_mem_97884, "ext_mem_97884") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97992, &ext_mem_97878, "ext_mem_97878") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97993, &ext_mem_97880, "ext_mem_97880") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97994, &ext_mem_97879, "ext_mem_97879") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97995, &ext_mem_97882, "ext_mem_97882") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97996, &ext_mem_97876, "ext_mem_97876") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97997, &ext_mem_97881, "ext_mem_97881") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97998, &ext_mem_97877, "ext_mem_97877") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97999, &ext_mem_97883, "ext_mem_97883") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_98000, &ext_mem_97875, "ext_mem_97875") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98390, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98391, &mem_out_97975, "mem_out_97975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98392, &mem_out_97976, "mem_out_97976") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98393, &mem_out_97977, "mem_out_97977") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98394, &mem_out_97978, "mem_out_97978") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98395, &mem_out_97979, "mem_out_97979") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98396, &mem_out_97980, "mem_out_97980") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98397, &mem_out_97981, "mem_out_97981") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98398, &mem_out_97982, "mem_out_97982") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98399, &mem_out_97983, "mem_out_97983") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98400, &mem_out_97984, "mem_out_97984") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98401, &mem_out_97985, "mem_out_97985") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98402, &mem_out_97986, "mem_out_97986") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98403, &mem_out_97987, "mem_out_97987") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98404, &mem_out_97988, "mem_out_97988") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98405, &mem_out_97989, "mem_out_97989") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98406, &mem_out_97990, "mem_out_97990") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98407, &mem_out_97991, "mem_out_97991") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98408, &mem_out_97992, "mem_out_97992") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98409, &mem_out_97993, "mem_out_97993") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98410, &mem_out_97994, "mem_out_97994") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98411, &mem_out_97995, "mem_out_97995") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98412, &mem_out_97996, "mem_out_97996") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98413, &mem_out_97997, "mem_out_97997") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98414, &mem_out_97998, "mem_out_97998") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98415, &mem_out_97999, "mem_out_97999") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98416, &mem_out_98000, "mem_out_98000") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_96245);
        free(mem_96246);
        free(mem_96255);
        free(mem_96262);
        free(mem_96277);
        free(mem_96278);
        free(mem_96287);
        free(mem_96294);
        free(mem_96309);
        free(mem_96310);
        free(mem_96319);
        free(mem_96320);
        free(mem_96341);
        free(mem_96342);
        free(mem_96343);
        free(mem_96355);
        free(mem_96356);
        free(mem_96380);
        free(mem_96381);
        free(mem_96382);
        free(mem_96383);
        free(mem_96384);
        free(mem_96403);
        free(mem_96404);
        free(mem_96405);
        free(mem_96442);
        free(mem_96443);
        free(mem_96444);
        free(mem_96460);
        free(mem_96461);
        free(mem_96462);
        free(mem_96475);
        free(mem_96476);
        free(mem_96477);
        free(mem_96523);
        free(mem_96524);
        free(mem_96535);
        free(mem_96536);
        free(mem_96545);
        free(mem_96546);
        free(mem_96567);
        free(mem_96572);
        free(mem_96583);
        free(mem_96588);
        free(mem_96595);
        free(mem_96606);
        free(mem_96611);
        free(mem_96632);
        free(mem_96633);
        free(mem_96641);
        free(mem_96655);
        free(mem_96660);
        free(mem_96671);
        free(mem_96676);
        free(mem_96687);
        free(mem_96688);
        free(mem_96697);
        free(mem_96698);
        free(mem_96719);
        free(mem_96720);
        free(mem_96728);
        free(mem_96742);
        free(mem_96743);
        free(mem_96751);
        free(mem_96765);
        free(mem_96770);
        free(mem_96781);
        free(mem_96786);
        free(mem_96797);
        free(mem_96802);
        free(mem_96813);
        free(mem_96820);
        free(mem_96825);
        free(mem_96836);
        free(mem_96843);
        free(mem_96847);
        free(mem_96857);
        free(mem_96862);
        free(mem_96869);
        free(mem_96880);
        free(mem_96887);
        free(mem_96892);
        free(mem_96903);
        free(mem_96908);
        free(mem_96919);
        free(mem_96920);
        free(mem_96929);
        free(mem_96930);
        free(mem_96951);
        free(mem_96956);
        free(mem_96967);
        free(mem_96972);
        free(mem_96983);
        free(mem_96990);
        free(mem_96997);
        free(mem_97007);
        free(mem_97012);
        free(mem_97023);
        free(mem_97024);
        free(mem_97033);
        free(mem_97034);
        free(mem_97055);
        free(mem_97056);
        free(mem_97067);
        free(mem_97068);
        free(mem_97077);
        free(mem_97084);
        free(mem_97109);
        free(mem_97110);
        free(mem_97111);
        free(mem_97126);
        free(mem_97127);
        free(mem_97128);
        free(mem_97140);
        free(mem_97147);
        free(mem_97154);
        free(mem_97186);
        free(mem_97187);
        free(mem_97198);
        free(mem_97199);
        free(mem_97208);
        free(mem_97215);
        free(mem_97240);
        free(mem_97245);
        free(mem_97256);
        free(mem_97261);
        free(mem_97272);
        free(mem_97278);
        free(mem_97283);
        free(mem_97299);
        free(mem_97304);
        free(mem_97315);
        free(mem_97321);
        free(mem_97326);
        free(mem_97342);
        free(mem_97348);
        free(mem_97353);
        free(mem_97369);
        free(mem_97370);
        free(mem_97381);
        free(mem_97382);
        free(mem_97391);
        free(mem_97392);
        free(mem_97423);
        free(mem_97424);
        free(mem_97425);
        free(mem_97438);
        free(mem_97439);
        free(mem_97440);
        free(mem_97471);
        free(mem_97472);
        free(mem_97473);
        free(mem_97474);
        free(mem_97491);
        free(mem_97492);
        free(mem_97493);
        free(mem_97494);
        free(mem_97535);
        free(mem_97542);
        free(mem_97549);
        free(mem_97559);
        free(mem_97564);
        free(mem_97575);
        free(mem_97582);
        free(mem_97589);
        free(mem_97599);
        free(mem_97604);
        free(mem_97615);
        free(mem_97616);
        free(mem_97625);
        free(mem_97626);
        free(mem_97647);
        free(mem_97652);
        free(mem_97663);
        free(mem_97664);
        free(mem_97673);
        free(mem_97674);
        if (memblock_unref(ctx, &mem_param_tmp_98027, "mem_param_tmp_98027") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98026, "mem_param_tmp_98026") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98025, "mem_param_tmp_98025") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98024, "mem_param_tmp_98024") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98023, "mem_param_tmp_98023") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98022, "mem_param_tmp_98022") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98021, "mem_param_tmp_98021") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98020, "mem_param_tmp_98020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98019, "mem_param_tmp_98019") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98018, "mem_param_tmp_98018") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98017, "mem_param_tmp_98017") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98016, "mem_param_tmp_98016") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98015, "mem_param_tmp_98015") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98014, "mem_param_tmp_98014") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98013, "mem_param_tmp_98013") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98012, "mem_param_tmp_98012") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98011, "mem_param_tmp_98011") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98010, "mem_param_tmp_98010") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98009, "mem_param_tmp_98009") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98008, "mem_param_tmp_98008") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98007, "mem_param_tmp_98007") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98006, "mem_param_tmp_98006") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98005, "mem_param_tmp_98005") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98004, "mem_param_tmp_98004") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98003, "mem_param_tmp_98003") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98002, "mem_param_tmp_98002") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_98001, "mem_param_tmp_98001") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97791, "ext_mem_97791") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97792, "ext_mem_97792") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97793, "ext_mem_97793") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97789, "mem_97789") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97787, "mem_97787") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97785, "mem_97785") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97783, "mem_97783") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97780, "ext_mem_97780") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97781, "ext_mem_97781") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97782, "ext_mem_97782") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97778, "mem_97778") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97776, "mem_97776") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97774, "mem_97774") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97772, "mem_97772") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97769, "ext_mem_97769") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97770, "ext_mem_97770") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97771, "ext_mem_97771") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97767, "mem_97767") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97765, "mem_97765") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97763, "mem_97763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97761, "mem_97761") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97758, "ext_mem_97758") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97759, "ext_mem_97759") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97760, "ext_mem_97760") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97756, "mem_97756") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97754, "mem_97754") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97752, "mem_97752") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97750, "mem_97750") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97747, "ext_mem_97747") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97748, "ext_mem_97748") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97749, "ext_mem_97749") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97745, "mem_97745") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97743, "mem_97743") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97741, "mem_97741") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97739, "mem_97739") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97736, "ext_mem_97736") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97737, "ext_mem_97737") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97738, "ext_mem_97738") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97734, "mem_97734") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97732, "mem_97732") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97730, "mem_97730") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97728, "mem_97728") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97725, "ext_mem_97725") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97726, "ext_mem_97726") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97727, "ext_mem_97727") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97723, "mem_97723") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97721, "mem_97721") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97719, "mem_97719") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97717, "mem_97717") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97714, "ext_mem_97714") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97715, "ext_mem_97715") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97716, "ext_mem_97716") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97712, "mem_97712") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97710, "mem_97710") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97708, "mem_97708") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97706, "mem_97706") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97703, "ext_mem_97703") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97704, "ext_mem_97704") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97705, "ext_mem_97705") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97701, "mem_97701") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97699, "mem_97699") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97697, "mem_97697") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_97695, "mem_97695") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96244, "mem_param_96244") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96240, "mem_param_96240") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96236, "mem_param_96236") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96232, "mem_param_96232") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96228, "mem_param_96228") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96224, "mem_param_96224") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96220, "mem_param_96220") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96216, "mem_param_96216") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96212, "mem_param_96212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96208, "mem_param_96208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96204, "mem_param_96204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96200, "mem_param_96200") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96196, "mem_param_96196") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96192, "mem_param_96192") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96188, "mem_param_96188") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96184, "mem_param_96184") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96180, "mem_param_96180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96176, "mem_param_96176") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96172, "mem_param_96172") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96168, "mem_param_96168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96164, "mem_param_96164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96160, "mem_param_96160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96156, "mem_param_96156") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96152, "mem_param_96152") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96148, "mem_param_96148") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96144, "mem_param_96144") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_96140, "mem_param_96140") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97875, "ext_mem_97875") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97876, "ext_mem_97876") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97877, "ext_mem_97877") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97878, "ext_mem_97878") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97879, "ext_mem_97879") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97880, "ext_mem_97880") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97881, "ext_mem_97881") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97882, "ext_mem_97882") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97883, "ext_mem_97883") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97884, "ext_mem_97884") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97885, "ext_mem_97885") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97886, "ext_mem_97886") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97887, "ext_mem_97887") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97888, "ext_mem_97888") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97889, "ext_mem_97889") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97890, "ext_mem_97890") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97891, "ext_mem_97891") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97892, "ext_mem_97892") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97893, "ext_mem_97893") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97894, "ext_mem_97894") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97895, "ext_mem_97895") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97896, "ext_mem_97896") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97897, "ext_mem_97897") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97898, "ext_mem_97898") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97899, "ext_mem_97899") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97900, "ext_mem_97900") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_97901, "ext_mem_97901") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_98000, "mem_out_98000") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97999, "mem_out_97999") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97998, "mem_out_97998") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97997, "mem_out_97997") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97996, "mem_out_97996") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97995, "mem_out_97995") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97994, "mem_out_97994") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97993, "mem_out_97993") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97992, "mem_out_97992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97991, "mem_out_97991") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97990, "mem_out_97990") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97989, "mem_out_97989") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97988, "mem_out_97988") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97987, "mem_out_97987") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97986, "mem_out_97986") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97985, "mem_out_97985") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97984, "mem_out_97984") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97983, "mem_out_97983") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97982, "mem_out_97982") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97981, "mem_out_97981") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97980, "mem_out_97980") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97979, "mem_out_97979") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97978, "mem_out_97978") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97977, "mem_out_97977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97976, "mem_out_97976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97975, "mem_out_97975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_98594, struct memblock *mem_out_p_98595, struct memblock *mem_out_p_98596, struct memblock *mem_out_p_98597, struct memblock *mem_out_p_98598, struct memblock *mem_out_p_98599, struct memblock *mem_out_p_98600, struct memblock *mem_out_p_98601, struct memblock *mem_out_p_98602)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mem_96098 = ctx->constants->mem_96098;
    struct memblock mem_96099 = ctx->constants->mem_96099;
    struct memblock mem_96100 = ctx->constants->mem_96100;
    struct memblock mem_96101 = ctx->constants->mem_96101;
    struct memblock mem_96102 = ctx->constants->mem_96102;
    struct memblock mem_96103 = ctx->constants->mem_96103;
    struct memblock mem_96104 = ctx->constants->mem_96104;
    struct memblock mem_96105 = ctx->constants->mem_96105;
    struct memblock mem_96106 = ctx->constants->mem_96106;
    
    if (memblock_set(ctx, &mem_out_97974, &mem_96105, "mem_96105") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97975, &mem_96101, "mem_96101") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97976, &mem_96103, "mem_96103") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97977, &mem_96099, "mem_96099") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97978, &mem_96100, "mem_96100") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97979, &mem_96098, "mem_96098") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97980, &mem_96104, "mem_96104") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97981, &mem_96102, "mem_96102") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_97982, &mem_96106, "mem_96106") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98594, &mem_out_97974, "mem_out_97974") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98595, &mem_out_97975, "mem_out_97975") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98596, &mem_out_97976, "mem_out_97976") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98597, &mem_out_97977, "mem_out_97977") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98598, &mem_out_97978, "mem_out_97978") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98599, &mem_out_97979, "mem_out_97979") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98600, &mem_out_97980, "mem_out_97980") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98601, &mem_out_97981, "mem_out_97981") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_98602, &mem_out_97982, "mem_out_97982") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_97982, "mem_out_97982") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97981, "mem_out_97981") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97980, "mem_out_97980") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97979, "mem_out_97979") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97978, "mem_out_97978") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97977, "mem_out_97977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97976, "mem_out_97976") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97975, "mem_out_97975") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_97974, "mem_out_97974") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward_seq(struct futhark_context *ctx, struct futhark_f64_2d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_1d *in1, const struct futhark_f64_2d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock mask_mem_96117;
    
    mask_mem_96117.references = NULL;
    
    struct memblock tokens_mem_96116;
    
    tokens_mem_96116.references = NULL;
    
    struct memblock wvoc_mem_96115;
    
    wvoc_mem_96115.references = NULL;
    
    struct memblock wval_mem_96114;
    
    wval_mem_96114.references = NULL;
    
    struct memblock wup_mem_96113;
    
    wup_mem_96113.references = NULL;
    
    struct memblock wte_mem_96112;
    
    wte_mem_96112.references = NULL;
    
    struct memblock wqry_mem_96111;
    
    wqry_mem_96111.references = NULL;
    
    struct memblock wpe_mem_96110;
    
    wpe_mem_96110.references = NULL;
    
    struct memblock wout_mem_96109;
    
    wout_mem_96109.references = NULL;
    
    struct memblock wkey_mem_96108;
    
    wkey_mem_96108.references = NULL;
    
    struct memblock wdown_mem_96107;
    
    wdown_mem_96107.references = NULL;
    wdown_mem_96107 = in0->v0->mem;
    wkey_mem_96108 = in0->v1->mem;
    wout_mem_96109 = in0->v2->mem;
    wpe_mem_96110 = in0->v3->mem;
    wqry_mem_96111 = in0->v4->mem;
    wte_mem_96112 = in0->v5->mem;
    wup_mem_96113 = in0->v6->mem;
    wval_mem_96114 = in0->v7->mem;
    wvoc_mem_96115 = in0->v8->mem;
    tokens_mem_96116 = in1->mem;
    mask_mem_96117 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && ((int64_t) 16 == in1->shape[0] && ((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1])))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward_seq(ctx, &mem_out_97974, wdown_mem_96107, wkey_mem_96108, wout_mem_96109, wpe_mem_96110, wqry_mem_96111, wte_mem_96112, wup_mem_96113, wval_mem_96114, wvoc_mem_96115, tokens_mem_96116, mask_mem_96117);
        if (ret == 0) {
            struct memblock mem_96098 = ctx->constants->mem_96098;
            struct memblock mem_96099 = ctx->constants->mem_96099;
            struct memblock mem_96100 = ctx->constants->mem_96100;
            struct memblock mem_96101 = ctx->constants->mem_96101;
            struct memblock mem_96102 = ctx->constants->mem_96102;
            struct memblock mem_96103 = ctx->constants->mem_96103;
            struct memblock mem_96104 = ctx->constants->mem_96104;
            struct memblock mem_96105 = ctx->constants->mem_96105;
            struct memblock mem_96106 = ctx->constants->mem_96106;
            
            assert((*out = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->mem = mem_out_97974;
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
    
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock wvoc_mem_96115;
    
    wvoc_mem_96115.references = NULL;
    
    struct memblock wdown_mem_96114;
    
    wdown_mem_96114.references = NULL;
    
    struct memblock wup_mem_96113;
    
    wup_mem_96113.references = NULL;
    
    struct memblock wout_mem_96112;
    
    wout_mem_96112.references = NULL;
    
    struct memblock wval_mem_96111;
    
    wval_mem_96111.references = NULL;
    
    struct memblock wkey_mem_96110;
    
    wkey_mem_96110.references = NULL;
    
    struct memblock wqry_mem_96109;
    
    wqry_mem_96109.references = NULL;
    
    struct memblock wpe_mem_96108;
    
    wpe_mem_96108.references = NULL;
    
    struct memblock wte_mem_96107;
    
    wte_mem_96107.references = NULL;
    wte_mem_96107 = in0->mem;
    wpe_mem_96108 = in1->mem;
    wqry_mem_96109 = in2->mem;
    wkey_mem_96110 = in3->mem;
    wval_mem_96111 = in4->mem;
    wout_mem_96112 = in5->mem;
    wup_mem_96113 = in6->mem;
    wdown_mem_96114 = in7->mem;
    wvoc_mem_96115 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_97974, &mem_out_97975, &mem_out_97976, &mem_out_97977, &mem_out_97978, &mem_out_97979, &mem_out_97980, &mem_out_97981, &mem_out_97982, wte_mem_96107, wpe_mem_96108, wqry_mem_96109, wkey_mem_96110, wval_mem_96111, wout_mem_96112, wup_mem_96113, wdown_mem_96114, wvoc_mem_96115);
        if (ret == 0) {
            struct memblock mem_96098 = ctx->constants->mem_96098;
            struct memblock mem_96099 = ctx->constants->mem_96099;
            struct memblock mem_96100 = ctx->constants->mem_96100;
            struct memblock mem_96101 = ctx->constants->mem_96101;
            struct memblock mem_96102 = ctx->constants->mem_96102;
            struct memblock mem_96103 = ctx->constants->mem_96103;
            struct memblock mem_96104 = ctx->constants->mem_96104;
            struct memblock mem_96105 = ctx->constants->mem_96105;
            struct memblock mem_96106 = ctx->constants->mem_96106;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_97974;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_97975;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_97976;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_97977;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_97978;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_97979;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_97980;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_97981;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_97982;
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
    
    struct memblock mem_out_98000;
    
    mem_out_98000.references = NULL;
    
    struct memblock mem_out_97999;
    
    mem_out_97999.references = NULL;
    
    struct memblock mem_out_97998;
    
    mem_out_97998.references = NULL;
    
    struct memblock mem_out_97997;
    
    mem_out_97997.references = NULL;
    
    struct memblock mem_out_97996;
    
    mem_out_97996.references = NULL;
    
    struct memblock mem_out_97995;
    
    mem_out_97995.references = NULL;
    
    struct memblock mem_out_97994;
    
    mem_out_97994.references = NULL;
    
    struct memblock mem_out_97993;
    
    mem_out_97993.references = NULL;
    
    struct memblock mem_out_97992;
    
    mem_out_97992.references = NULL;
    
    struct memblock mem_out_97991;
    
    mem_out_97991.references = NULL;
    
    struct memblock mem_out_97990;
    
    mem_out_97990.references = NULL;
    
    struct memblock mem_out_97989;
    
    mem_out_97989.references = NULL;
    
    struct memblock mem_out_97988;
    
    mem_out_97988.references = NULL;
    
    struct memblock mem_out_97987;
    
    mem_out_97987.references = NULL;
    
    struct memblock mem_out_97986;
    
    mem_out_97986.references = NULL;
    
    struct memblock mem_out_97985;
    
    mem_out_97985.references = NULL;
    
    struct memblock mem_out_97984;
    
    mem_out_97984.references = NULL;
    
    struct memblock mem_out_97983;
    
    mem_out_97983.references = NULL;
    
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    
    struct memblock seqs_mem_96136;
    
    seqs_mem_96136.references = NULL;
    
    struct memblock dls_mem_96135;
    
    dls_mem_96135.references = NULL;
    
    struct memblock masks_mem_96134;
    
    masks_mem_96134.references = NULL;
    
    struct memblock wvoc_mem_96133;
    
    wvoc_mem_96133.references = NULL;
    
    struct memblock wval_mem_96132;
    
    wval_mem_96132.references = NULL;
    
    struct memblock wup_mem_96131;
    
    wup_mem_96131.references = NULL;
    
    struct memblock wte_mem_96130;
    
    wte_mem_96130.references = NULL;
    
    struct memblock wqry_mem_96129;
    
    wqry_mem_96129.references = NULL;
    
    struct memblock wpe_mem_96128;
    
    wpe_mem_96128.references = NULL;
    
    struct memblock wout_mem_96127;
    
    wout_mem_96127.references = NULL;
    
    struct memblock wkey_mem_96126;
    
    wkey_mem_96126.references = NULL;
    
    struct memblock wdown_mem_96125;
    
    wdown_mem_96125.references = NULL;
    
    struct memblock wvoc_mem_96124;
    
    wvoc_mem_96124.references = NULL;
    
    struct memblock wval_mem_96123;
    
    wval_mem_96123.references = NULL;
    
    struct memblock wup_mem_96122;
    
    wup_mem_96122.references = NULL;
    
    struct memblock wte_mem_96121;
    
    wte_mem_96121.references = NULL;
    
    struct memblock wqry_mem_96120;
    
    wqry_mem_96120.references = NULL;
    
    struct memblock wpe_mem_96119;
    
    wpe_mem_96119.references = NULL;
    
    struct memblock wout_mem_96118;
    
    wout_mem_96118.references = NULL;
    
    struct memblock wkey_mem_96117;
    
    wkey_mem_96117.references = NULL;
    
    struct memblock wdown_mem_96116;
    
    wdown_mem_96116.references = NULL;
    
    struct memblock wvoc_mem_96115;
    
    wvoc_mem_96115.references = NULL;
    
    struct memblock wval_mem_96114;
    
    wval_mem_96114.references = NULL;
    
    struct memblock wup_mem_96113;
    
    wup_mem_96113.references = NULL;
    
    struct memblock wte_mem_96112;
    
    wte_mem_96112.references = NULL;
    
    struct memblock wqry_mem_96111;
    
    wqry_mem_96111.references = NULL;
    
    struct memblock wpe_mem_96110;
    
    wpe_mem_96110.references = NULL;
    
    struct memblock wout_mem_96109;
    
    wout_mem_96109.references = NULL;
    
    struct memblock wkey_mem_96108;
    
    wkey_mem_96108.references = NULL;
    
    struct memblock wdown_mem_96107;
    
    wdown_mem_96107.references = NULL;
    wdown_mem_96107 = in0->v0->mem;
    wkey_mem_96108 = in0->v1->mem;
    wout_mem_96109 = in0->v2->mem;
    wpe_mem_96110 = in0->v3->mem;
    wqry_mem_96111 = in0->v4->mem;
    wte_mem_96112 = in0->v5->mem;
    wup_mem_96113 = in0->v6->mem;
    wval_mem_96114 = in0->v7->mem;
    wvoc_mem_96115 = in0->v8->mem;
    wdown_mem_96116 = in1->v0->mem;
    wkey_mem_96117 = in1->v1->mem;
    wout_mem_96118 = in1->v2->mem;
    wpe_mem_96119 = in1->v3->mem;
    wqry_mem_96120 = in1->v4->mem;
    wte_mem_96121 = in1->v5->mem;
    wup_mem_96122 = in1->v6->mem;
    wval_mem_96123 = in1->v7->mem;
    wvoc_mem_96124 = in1->v8->mem;
    wdown_mem_96125 = in2->v0->mem;
    wkey_mem_96126 = in2->v1->mem;
    wout_mem_96127 = in2->v2->mem;
    wpe_mem_96128 = in2->v3->mem;
    wqry_mem_96129 = in2->v4->mem;
    wte_mem_96130 = in2->v5->mem;
    wup_mem_96131 = in2->v6->mem;
    wval_mem_96132 = in2->v7->mem;
    wvoc_mem_96133 = in2->v8->mem;
    masks_mem_96134 = in3->mem;
    dls_mem_96135 = in4->mem;
    seqs_mem_96136 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 500 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2])) && ((int64_t) 500 == in4->shape[0] && ((int64_t) 500 == in5->shape[0] && (int64_t) 16 == in5->shape[1]))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_97974, &mem_out_97975, &mem_out_97976, &mem_out_97977, &mem_out_97978, &mem_out_97979, &mem_out_97980, &mem_out_97981, &mem_out_97982, &mem_out_97983, &mem_out_97984, &mem_out_97985, &mem_out_97986, &mem_out_97987, &mem_out_97988, &mem_out_97989, &mem_out_97990, &mem_out_97991, &mem_out_97992, &mem_out_97993, &mem_out_97994, &mem_out_97995, &mem_out_97996, &mem_out_97997, &mem_out_97998, &mem_out_97999, &mem_out_98000, wdown_mem_96107, wkey_mem_96108, wout_mem_96109, wpe_mem_96110, wqry_mem_96111, wte_mem_96112, wup_mem_96113, wval_mem_96114, wvoc_mem_96115, wdown_mem_96116, wkey_mem_96117, wout_mem_96118, wpe_mem_96119, wqry_mem_96120, wte_mem_96121, wup_mem_96122, wval_mem_96123, wvoc_mem_96124, wdown_mem_96125, wkey_mem_96126, wout_mem_96127, wpe_mem_96128, wqry_mem_96129, wte_mem_96130, wup_mem_96131, wval_mem_96132, wvoc_mem_96133, masks_mem_96134, dls_mem_96135, seqs_mem_96136);
        if (ret == 0) {
            struct memblock mem_96098 = ctx->constants->mem_96098;
            struct memblock mem_96099 = ctx->constants->mem_96099;
            struct memblock mem_96100 = ctx->constants->mem_96100;
            struct memblock mem_96101 = ctx->constants->mem_96101;
            struct memblock mem_96102 = ctx->constants->mem_96102;
            struct memblock mem_96103 = ctx->constants->mem_96103;
            struct memblock mem_96104 = ctx->constants->mem_96104;
            struct memblock mem_96105 = ctx->constants->mem_96105;
            struct memblock mem_96106 = ctx->constants->mem_96106;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_tup9_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64_arr2d_f64))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_97974;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_97975;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_97976;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_97977;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_97978;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_97979;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_97980;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_97981;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_97982;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v9->mem = mem_out_97983;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v10->mem = mem_out_97984;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v11->mem = mem_out_97985;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v12->mem = mem_out_97986;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v13->mem = mem_out_97987;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v14->mem = mem_out_97988;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v15->mem = mem_out_97989;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v16->mem = mem_out_97990;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v17->mem = mem_out_97991;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v18->mem = mem_out_97992;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v19->mem = mem_out_97993;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v20->mem = mem_out_97994;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v21->mem = mem_out_97995;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v22->mem = mem_out_97996;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v23->mem = mem_out_97997;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v24->mem = mem_out_97998;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v25->mem = mem_out_97999;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v26->mem = mem_out_98000;
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
    
    struct memblock mem_out_97982;
    
    mem_out_97982.references = NULL;
    
    struct memblock mem_out_97981;
    
    mem_out_97981.references = NULL;
    
    struct memblock mem_out_97980;
    
    mem_out_97980.references = NULL;
    
    struct memblock mem_out_97979;
    
    mem_out_97979.references = NULL;
    
    struct memblock mem_out_97978;
    
    mem_out_97978.references = NULL;
    
    struct memblock mem_out_97977;
    
    mem_out_97977.references = NULL;
    
    struct memblock mem_out_97976;
    
    mem_out_97976.references = NULL;
    
    struct memblock mem_out_97975;
    
    mem_out_97975.references = NULL;
    
    struct memblock mem_out_97974;
    
    mem_out_97974.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_97974, &mem_out_97975, &mem_out_97976, &mem_out_97977, &mem_out_97978, &mem_out_97979, &mem_out_97980, &mem_out_97981, &mem_out_97982);
        if (ret == 0) {
            struct memblock mem_96098 = ctx->constants->mem_96098;
            struct memblock mem_96099 = ctx->constants->mem_96099;
            struct memblock mem_96100 = ctx->constants->mem_96100;
            struct memblock mem_96101 = ctx->constants->mem_96101;
            struct memblock mem_96102 = ctx->constants->mem_96102;
            struct memblock mem_96103 = ctx->constants->mem_96103;
            struct memblock mem_96104 = ctx->constants->mem_96104;
            struct memblock mem_96105 = ctx->constants->mem_96105;
            struct memblock mem_96106 = ctx->constants->mem_96106;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v0->mem = mem_out_97974;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v1->mem = mem_out_97975;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v2->mem = mem_out_97976;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v3->mem = mem_out_97977;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v4->mem = mem_out_97978;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v5->mem = mem_out_97979;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v6->mem = mem_out_97980;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v7->mem = mem_out_97981;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) != NULL);
            (*out)->v8->mem = mem_out_97982;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
