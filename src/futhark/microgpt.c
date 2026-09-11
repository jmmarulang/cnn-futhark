
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
struct futhark_f32_2d;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data);
int futhark_index_f32_2d(struct futhark_context *ctx, float *out, struct futhark_f32_2d *arr, int64_t i0, int64_t i1);
unsigned char *futhark_values_raw_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);
struct futhark_f32_3d;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2);
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2);
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr, float *data);
int futhark_index_f32_3d(struct futhark_context *ctx, float *out, struct futhark_f32_3d *arr, int64_t i0, int64_t i1, int64_t i2);
unsigned char *futhark_values_raw_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr);
const int64_t *futhark_shape_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr);
struct futhark_f32_4d;
struct futhark_f32_4d *futhark_new_f32_4d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3);
struct futhark_f32_4d *futhark_new_raw_f32_4d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3);
int futhark_free_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr);
int futhark_values_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr, float *data);
int futhark_index_f32_4d(struct futhark_context *ctx, float *out, struct futhark_f32_4d *arr, int64_t i0, int64_t i1, int64_t i2, int64_t i3);
unsigned char *futhark_values_raw_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr);
const int64_t *futhark_shape_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr);
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
struct futhark_i64_3d;
struct futhark_i64_3d *futhark_new_i64_3d(struct futhark_context *ctx, const int64_t *data, int64_t dim0, int64_t dim1, int64_t dim2);
struct futhark_i64_3d *futhark_new_raw_i64_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2);
int futhark_free_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);
int futhark_values_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr, int64_t *data);
int futhark_index_i64_3d(struct futhark_context *ctx, int64_t *out, struct futhark_i64_3d *arr, int64_t i0, int64_t i1, int64_t i2);
unsigned char *futhark_values_raw_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);
const int64_t *futhark_shape_i64_3d(struct futhark_context *ctx, struct futhark_i64_3d *arr);

// Opaque values
struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32;
struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32;
struct futhark_opaque_params;
int futhark_free_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_store_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj, void **p, size_t *n);
struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *futhark_restore_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_new_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_0, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_1, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_2);
int futhark_free_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_store_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj, void **p, size_t *n);
struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *futhark_restore_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_3(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_4(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_5(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_6(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_7(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_8(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj);
int futhark_new_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_f32_2d *f_0, const struct futhark_f32_2d *f_1, const struct futhark_f32_2d *f_2, const struct futhark_f32_2d *f_3, const struct futhark_f32_2d *f_4, const struct futhark_f32_2d *f_5, const struct futhark_f32_2d *f_6, const struct futhark_f32_2d *f_7, const struct futhark_f32_2d *f_8);
int futhark_free_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params *obj);
int futhark_store_opaque_params(struct futhark_context *ctx, const struct futhark_opaque_params *obj, void **p, size_t *n);
struct futhark_opaque_params *futhark_restore_opaque_params(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_params_wdown(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wkey(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wout(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wpe(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wqry(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wte(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wup(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wval(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_project_opaque_params_wvoc(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj);
int futhark_new_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f32_2d *f_wdown, const struct futhark_f32_2d *f_wkey, const struct futhark_f32_2d *f_wout, const struct futhark_f32_2d *f_wpe, const struct futhark_f32_2d *f_wqry, const struct futhark_f32_2d *f_wte, const struct futhark_f32_2d *f_wup, const struct futhark_f32_2d *f_wval, const struct futhark_f32_2d *f_wvoc);

// Entry points
int futhark_entry_forward(struct futhark_context *ctx, struct futhark_f32_3d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_2d *in1, const struct futhark_f32_3d *in2);
int futhark_entry_loss(struct futhark_context *ctx, float *out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_i64_2d *in2, const struct futhark_f32_3d *in3);
int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f32_2d *in0, const struct futhark_f32_2d *in1, const struct futhark_f32_2d *in2, const struct futhark_f32_2d *in3, const struct futhark_f32_2d *in4, const struct futhark_f32_2d *in5, const struct futhark_f32_2d *in6, const struct futhark_f32_2d *in7, const struct futhark_f32_2d *in8);
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f32_4d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_3d *in5);
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

const struct type type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR;
const struct type type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR;
const struct type type_ZMZNZMZNZMZNZMZNf32;
const struct type type_ZMZNZMZNZMZNf32;
const struct type type_ZMZNZMZNZMZNi64;
const struct type type_ZMZNZMZNf32;
const struct type type_ZMZNZMZNi64;
const struct type type_ZMZNi64;
const struct type type_params;
const struct field type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_fields[] = {{.name ="0", .type =&type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0}, {.name ="1", .type =&type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1}, {.name ="2", .type =&type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR, .project =(project_fn) futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2}};
int futhark_new_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *out = (struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *) outp;
    const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * v0 = *(const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *) fields[0];
    const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * v1 = *(const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *) fields[1];
    const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * v2 = *(const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *) fields[2];
    
    return futhark_new_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(ctx, out, v0, v1, v2);
}
const struct record type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_record = {.num_fields =3, .fields =type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_fields, .new =futhark_new_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_wrap};
const struct opaque_aux type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32, .restore =(opaque_restore_fn) futhark_restore_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32, .free =(opaque_free_fn) futhark_free_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32};
const struct type type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR = {.name ="(([][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32), ([][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32), ([][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32))", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_aux, .kind =RECORD, .info =&type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR_record};
const struct field type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_fields[] = {{.name ="0", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0}, {.name ="1", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1}, {.name ="2", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2}, {.name ="3", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_3}, {.name ="4", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_4}, {.name ="5", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_5}, {.name ="6", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_6}, {.name ="7", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_7}, {.name ="8", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_8}};
int futhark_new_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *out = (struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 * *) outp;
    const struct futhark_f32_2d * v0 = *(const struct futhark_f32_2d * *) fields[0];
    const struct futhark_f32_2d * v1 = *(const struct futhark_f32_2d * *) fields[1];
    const struct futhark_f32_2d * v2 = *(const struct futhark_f32_2d * *) fields[2];
    const struct futhark_f32_2d * v3 = *(const struct futhark_f32_2d * *) fields[3];
    const struct futhark_f32_2d * v4 = *(const struct futhark_f32_2d * *) fields[4];
    const struct futhark_f32_2d * v5 = *(const struct futhark_f32_2d * *) fields[5];
    const struct futhark_f32_2d * v6 = *(const struct futhark_f32_2d * *) fields[6];
    const struct futhark_f32_2d * v7 = *(const struct futhark_f32_2d * *) fields[7];
    const struct futhark_f32_2d * v8 = *(const struct futhark_f32_2d * *) fields[8];
    
    return futhark_new_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(ctx, out, v0, v1, v2, v3, v4, v5, v6, v7, v8);
}
const struct record type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_record = {.num_fields =9, .fields =type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_fields, .new =futhark_new_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_wrap};
const struct opaque_aux type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_aux = {.store =(opaque_store_fn) futhark_store_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32, .restore =(opaque_restore_fn) futhark_restore_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32, .free =(opaque_free_fn) futhark_free_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32};
const struct type type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR = {.name ="([][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32, [][]f32)", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_aux, .kind =RECORD, .info =&type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR_record};
void *futhark_new_f32_4d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f32_4d(ctx, p, shape[0], shape[1], shape[2], shape[3]);
}
int futhark_new_f32_4d_wrap(struct futhark_context *ctx, struct futhark_f32_4d * *outp, float *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 4; ++i)
        n_values *= shape[i];
    
    float *values = alloca(n_values * sizeof(float));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f32_4d(ctx, values, shape[0], shape[1], shape[2], shape[3]);
    return 0;
}
int futhark_new_f32_4d_set(struct futhark_context *ctx, struct futhark_f32_4d * arr, float *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f32_4d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 4; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((float *) futhark_values_raw_f32_4d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f32_4d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f32_4d * arr, const int64_t *is)
{
    return futhark_index_f32_4d(ctx, dest, arr, is[0], is[1], is[2], is[3]);
}
const struct array type_ZMZNZMZNZMZNZMZNf32_array = {.rank =4, .element_type =&type_f32, .new =(array_new_fn) futhark_new_f32_4d_wrap, .set =(array_set_fn) futhark_new_f32_4d_set, .shape =(array_shape_fn) futhark_shape_f32_4d, .index =(array_index_fn) futhark_index_f32_4d_wrap};
const struct array_aux type_ZMZNZMZNZMZNZMZNf32_aux = {.name ="[][][][]f32", .rank =4, .info =&f32_info, .new =(aux_array_new_fn) futhark_new_f32_4d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f32_4d, .shape =(aux_array_shape_fn) futhark_shape_f32_4d, .values =(aux_array_values_fn) futhark_values_f32_4d};
const struct type type_ZMZNZMZNZMZNZMZNf32 = {.name ="[][][][]f32", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNZMZNZMZNf32_aux, .kind =ARRAY, .info =&type_ZMZNZMZNZMZNZMZNf32_array};
void *futhark_new_f32_3d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f32_3d(ctx, p, shape[0], shape[1], shape[2]);
}
int futhark_new_f32_3d_wrap(struct futhark_context *ctx, struct futhark_f32_3d * *outp, float *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 3; ++i)
        n_values *= shape[i];
    
    float *values = alloca(n_values * sizeof(float));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f32_3d(ctx, values, shape[0], shape[1], shape[2]);
    return 0;
}
int futhark_new_f32_3d_set(struct futhark_context *ctx, struct futhark_f32_3d * arr, float *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f32_3d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 3; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((float *) futhark_values_raw_f32_3d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f32_3d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f32_3d * arr, const int64_t *is)
{
    return futhark_index_f32_3d(ctx, dest, arr, is[0], is[1], is[2]);
}
const struct array type_ZMZNZMZNZMZNf32_array = {.rank =3, .element_type =&type_f32, .new =(array_new_fn) futhark_new_f32_3d_wrap, .set =(array_set_fn) futhark_new_f32_3d_set, .shape =(array_shape_fn) futhark_shape_f32_3d, .index =(array_index_fn) futhark_index_f32_3d_wrap};
const struct array_aux type_ZMZNZMZNZMZNf32_aux = {.name ="[][][]f32", .rank =3, .info =&f32_info, .new =(aux_array_new_fn) futhark_new_f32_3d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f32_3d, .shape =(aux_array_shape_fn) futhark_shape_f32_3d, .values =(aux_array_values_fn) futhark_values_f32_3d};
const struct type type_ZMZNZMZNZMZNf32 = {.name ="[][][]f32", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNZMZNf32_aux, .kind =ARRAY, .info =&type_ZMZNZMZNZMZNf32_array};
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
void *futhark_new_f32_2d_aux_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f32_2d(ctx, p, shape[0], shape[1]);
}
int futhark_new_f32_2d_wrap(struct futhark_context *ctx, struct futhark_f32_2d * *outp, float *ps[], const int64_t *shape)
{
    int64_t n_values = 1;
    
    for (int i = 0; i < 2; ++i)
        n_values *= shape[i];
    
    float *values = alloca(n_values * sizeof(float));
    
    for (int64_t i = 0; i < n_values; ++i)
        values[i] = *ps[i];
    *outp = futhark_new_f32_2d(ctx, values, shape[0], shape[1]);
    return 0;
}
int futhark_new_f32_2d_set(struct futhark_context *ctx, struct futhark_f32_2d * arr, float *val, const int64_t *is)
{
    const int64_t *shape = futhark_shape_f32_2d(ctx, arr);
    uint64_t idx = is[0];
    
    for (int i = 1; i < 2; ++i) {
        idx *= shape[i - 1];
        idx += is[i];
    }
    ((float *) futhark_values_raw_f32_2d(ctx, arr))[idx] = *val;
    return 0;
}
int futhark_index_f32_2d_wrap(struct futhark_context *ctx, void *dest, struct futhark_f32_2d * arr, const int64_t *is)
{
    return futhark_index_f32_2d(ctx, dest, arr, is[0], is[1]);
}
const struct array type_ZMZNZMZNf32_array = {.rank =2, .element_type =&type_f32, .new =(array_new_fn) futhark_new_f32_2d_wrap, .set =(array_set_fn) futhark_new_f32_2d_set, .shape =(array_shape_fn) futhark_shape_f32_2d, .index =(array_index_fn) futhark_index_f32_2d_wrap};
const struct array_aux type_ZMZNZMZNf32_aux = {.name ="[][]f32", .rank =2, .info =&f32_info, .new =(aux_array_new_fn) futhark_new_f32_2d_aux_wrap, .free =(aux_array_free_fn) futhark_free_f32_2d, .shape =(aux_array_shape_fn) futhark_shape_f32_2d, .values =(aux_array_values_fn) futhark_values_f32_2d};
const struct type type_ZMZNZMZNf32 = {.name ="[][]f32", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNZMZNf32_aux, .kind =ARRAY, .info =&type_ZMZNZMZNf32_array};
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
const struct field type_params_fields[] = {{.name ="wdown", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wdown}, {.name ="wkey", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wkey}, {.name ="wout", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wout}, {.name ="wpe", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wpe}, {.name ="wqry", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wqry}, {.name ="wte", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wte}, {.name ="wup", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wup}, {.name ="wval", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wval}, {.name ="wvoc", .type =&type_ZMZNZMZNf32, .project =(project_fn) futhark_project_opaque_params_wvoc}};
int futhark_new_opaque_params_wrap(struct futhark_context *ctx, void **outp, const void *fields[])
{
    struct futhark_opaque_params * *out = (struct futhark_opaque_params * *) outp;
    const struct futhark_f32_2d * v0 = *(const struct futhark_f32_2d * *) fields[0];
    const struct futhark_f32_2d * v1 = *(const struct futhark_f32_2d * *) fields[1];
    const struct futhark_f32_2d * v2 = *(const struct futhark_f32_2d * *) fields[2];
    const struct futhark_f32_2d * v3 = *(const struct futhark_f32_2d * *) fields[3];
    const struct futhark_f32_2d * v4 = *(const struct futhark_f32_2d * *) fields[4];
    const struct futhark_f32_2d * v5 = *(const struct futhark_f32_2d * *) fields[5];
    const struct futhark_f32_2d * v6 = *(const struct futhark_f32_2d * *) fields[6];
    const struct futhark_f32_2d * v7 = *(const struct futhark_f32_2d * *) fields[7];
    const struct futhark_f32_2d * v8 = *(const struct futhark_f32_2d * *) fields[8];
    
    return futhark_new_opaque_params(ctx, out, v0, v1, v2, v3, v4, v5, v6, v7, v8);
}
const struct record type_params_record = {.num_fields =9, .fields =type_params_fields, .new =futhark_new_opaque_params_wrap};
const struct opaque_aux type_params_aux = {.store =(opaque_store_fn) futhark_store_opaque_params, .restore =(opaque_restore_fn) futhark_restore_opaque_params, .free =(opaque_free_fn) futhark_free_opaque_params};
const struct type type_params = {.name ="params", .restore =(restore_fn) restore_opaque, .store =(store_fn) store_opaque, .free =(free_fn) free_opaque, .aux =&type_params_aux, .kind =RECORD, .info =&type_params_record};
const struct type *forward_in_types[] = {&type_params, &type_ZMZNZMZNi64, &type_ZMZNZMZNZMZNf32, NULL};
bool forward_in_unique[] = {false, false, false};
const char *forward_tuning_params[] = {NULL};
const char *forward_attrs[] = {NULL};
int call_forward(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_i64_2d * in1 = *(struct futhark_i64_2d * *) ins[1];
    struct futhark_f32_3d * in2 = *(struct futhark_f32_3d * *) ins[2];
    
    return futhark_entry_forward(ctx, out, in0, in1, in2);
}
const struct type *loss_in_types[] = {&type_i64, &type_params, &type_ZMZNZMZNi64, &type_ZMZNZMZNZMZNf32, NULL};
bool loss_in_unique[] = {false, false, false, false};
const char *loss_tuning_params[] = {NULL};
const char *loss_attrs[] = {NULL};
int call_loss(struct futhark_context *ctx, void *out, void **ins)
{
    int64_t in0 = *(int64_t *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_i64_2d * in2 = *(struct futhark_i64_2d * *) ins[2];
    struct futhark_f32_3d * in3 = *(struct futhark_f32_3d * *) ins[3];
    
    return futhark_entry_loss(ctx, out, in0, in1, in2, in3);
}
const struct type *to_params_in_types[] = {&type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, &type_ZMZNZMZNf32, NULL};
bool to_params_in_unique[] = {false, false, false, false, false, false, false, false, false};
const char *to_params_tuning_params[] = {NULL};
const char *to_params_attrs[] = {NULL};
int call_to_params(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_f32_2d * in0 = *(struct futhark_f32_2d * *) ins[0];
    struct futhark_f32_2d * in1 = *(struct futhark_f32_2d * *) ins[1];
    struct futhark_f32_2d * in2 = *(struct futhark_f32_2d * *) ins[2];
    struct futhark_f32_2d * in3 = *(struct futhark_f32_2d * *) ins[3];
    struct futhark_f32_2d * in4 = *(struct futhark_f32_2d * *) ins[4];
    struct futhark_f32_2d * in5 = *(struct futhark_f32_2d * *) ins[5];
    struct futhark_f32_2d * in6 = *(struct futhark_f32_2d * *) ins[6];
    struct futhark_f32_2d * in7 = *(struct futhark_f32_2d * *) ins[7];
    struct futhark_f32_2d * in8 = *(struct futhark_f32_2d * *) ins[8];
    
    return futhark_entry_to_params(ctx, out, in0, in1, in2, in3, in4, in5, in6, in7, in8);
}
const struct type *train_in_types[] = {&type_params, &type_params, &type_params, &type_ZMZNZMZNZMZNZMZNf32, &type_ZMZNi64, &type_ZMZNZMZNZMZNi64, NULL};
bool train_in_unique[] = {false, false, false, false, false, false};
const char *train_tuning_params[] = {NULL};
const char *train_attrs[] = {NULL};
int call_train(struct futhark_context *ctx, void *out, void **ins)
{
    struct futhark_opaque_params * in0 = *(struct futhark_opaque_params * *) ins[0];
    struct futhark_opaque_params * in1 = *(struct futhark_opaque_params * *) ins[1];
    struct futhark_opaque_params * in2 = *(struct futhark_opaque_params * *) ins[2];
    struct futhark_f32_4d * in3 = *(struct futhark_f32_4d * *) ins[3];
    struct futhark_i64_1d * in4 = *(struct futhark_i64_1d * *) ins[4];
    struct futhark_i64_3d * in5 = *(struct futhark_i64_3d * *) ins[5];
    
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
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR, &type_ZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZR, &type_ZMZNZMZNZMZNZMZNf32, &type_ZMZNZMZNZMZNf32, &type_ZMZNZMZNZMZNi64, &type_ZMZNZMZNf32, &type_ZMZNZMZNi64, &type_ZMZNi64, &type_params, NULL};
struct entry_point entry_points[] = {{.name ="forward", .f =call_forward, .tuning_params =forward_tuning_params, .in_types =forward_in_types, .out_type =&type_ZMZNZMZNZMZNf32, .in_unique =forward_in_unique, .out_unique =false, .attrs =forward_attrs}, {.name ="loss", .f =call_loss, .tuning_params =loss_tuning_params, .in_types =loss_in_types, .out_type =&type_f32, .in_unique =loss_in_unique, .out_unique =false, .attrs =loss_attrs}, {.name ="to_params", .f =call_to_params, .tuning_params =to_params_tuning_params, .in_types =to_params_in_types, .out_type =&type_params, .in_unique =to_params_in_unique, .out_unique =false, .attrs =to_params_attrs}, {.name ="train", .f =call_train, .tuning_params =train_tuning_params, .in_types =train_in_types, .out_type =&type_ZLZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRz2cUz20UZLZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32z2cUz20UZMZNZMZNf32ZRZR, .in_unique =train_in_unique, .out_unique =false, .attrs =train_attrs}, {.name ="zero_params", .f =call_zzero_params, .tuning_params =zzero_params_tuning_params, .in_types =zzero_params_in_types, .out_type =&type_params, .in_unique =zzero_params_in_unique, .out_unique =false, .attrs =zzero_params_attrs}, {.name =NULL}};
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
    struct memblock mem_145202;
    struct memblock mem_145203;
    struct memblock mem_145204;
    struct memblock mem_145205;
    struct memblock mem_145206;
    struct memblock mem_145207;
    struct memblock mem_145208;
    struct memblock mem_145209;
    struct memblock mem_145210;
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

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12790(struct futhark_context *ctx, struct memblock *mem_out_p_147484, struct memblock *mem_out_p_147485, struct memblock *mem_out_p_147486, struct memblock w_mem_145211, struct memblock mw_mem_145212, struct memblock vw_mem_145213, struct memblock dw_mem_145214, int64_t n_119001, int64_t m_119002, int64_t step_119007, float lt_r_119008);
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12791(struct futhark_context *ctx, struct memblock *mem_out_p_147490, struct memblock *mem_out_p_147491, struct memblock *mem_out_p_147492, struct memblock w_mem_145211, struct memblock mw_mem_145212, struct memblock vw_mem_145213, struct memblock dw_mem_145214, int64_t n_120034, int64_t m_120035, int64_t step_120040, float lt_r_120041);
FUTHARK_FUN_ATTR int futrts_cal_target_9281(struct futhark_context *ctx, struct memblock *mem_out_p_147496, struct memblock seq_mem_145211, int64_t n_72699);
FUTHARK_FUN_ATTR int futrts_entry_forward(struct futhark_context *ctx, struct memblock *mem_out_p_147498, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock seqs_mem_145220, struct memblock masks_mem_145221);
FUTHARK_FUN_ATTR int futrts_entry_loss(struct futhark_context *ctx, float *out_prim_out_147561, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock seqs_mem_145220, struct memblock masks_mem_145221, int64_t dl_83939);
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_147628, struct memblock *mem_out_p_147629, struct memblock *mem_out_p_147630, struct memblock *mem_out_p_147631, struct memblock *mem_out_p_147632, struct memblock *mem_out_p_147633, struct memblock *mem_out_p_147634, struct memblock *mem_out_p_147635, struct memblock *mem_out_p_147636, struct memblock wte_mem_145211, struct memblock wpe_mem_145212, struct memblock wqry_mem_145213, struct memblock wkey_mem_145214, struct memblock wval_mem_145215, struct memblock wout_mem_145216, struct memblock wup_mem_145217, struct memblock wdown_mem_145218, struct memblock wvoc_mem_145219);
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_147637, struct memblock *mem_out_p_147638, struct memblock *mem_out_p_147639, struct memblock *mem_out_p_147640, struct memblock *mem_out_p_147641, struct memblock *mem_out_p_147642, struct memblock *mem_out_p_147643, struct memblock *mem_out_p_147644, struct memblock *mem_out_p_147645, struct memblock *mem_out_p_147646, struct memblock *mem_out_p_147647, struct memblock *mem_out_p_147648, struct memblock *mem_out_p_147649, struct memblock *mem_out_p_147650, struct memblock *mem_out_p_147651, struct memblock *mem_out_p_147652, struct memblock *mem_out_p_147653, struct memblock *mem_out_p_147654, struct memblock *mem_out_p_147655, struct memblock *mem_out_p_147656, struct memblock *mem_out_p_147657, struct memblock *mem_out_p_147658, struct memblock *mem_out_p_147659, struct memblock *mem_out_p_147660, struct memblock *mem_out_p_147661, struct memblock *mem_out_p_147662, struct memblock *mem_out_p_147663, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock wdown_mem_145220, struct memblock wkey_mem_145221, struct memblock wout_mem_145222, struct memblock wpe_mem_145223, struct memblock wqry_mem_145224, struct memblock wte_mem_145225, struct memblock wup_mem_145226, struct memblock wval_mem_145227, struct memblock wvoc_mem_145228, struct memblock wdown_mem_145229, struct memblock wkey_mem_145230, struct memblock wout_mem_145231, struct memblock wpe_mem_145232, struct memblock wqry_mem_145233, struct memblock wte_mem_145234, struct memblock wup_mem_145235, struct memblock wval_mem_145236, struct memblock wvoc_mem_145237, struct memblock masks_mem_145238, struct memblock dls_mem_145239, struct memblock seqs_mem_145240);
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_147847, struct memblock *mem_out_p_147848, struct memblock *mem_out_p_147849, struct memblock *mem_out_p_147850, struct memblock *mem_out_p_147851, struct memblock *mem_out_p_147852, struct memblock *mem_out_p_147853, struct memblock *mem_out_p_147854, struct memblock *mem_out_p_147855);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_145202 (ctx->constants->mem_145202)
    #define mem_145203 (ctx->constants->mem_145203)
    #define mem_145204 (ctx->constants->mem_145204)
    #define mem_145205 (ctx->constants->mem_145205)
    #define mem_145206 (ctx->constants->mem_145206)
    #define mem_145207 (ctx->constants->mem_145207)
    #define mem_145208 (ctx->constants->mem_145208)
    #define mem_145209 (ctx->constants->mem_145209)
    #define mem_145210 (ctx->constants->mem_145210)
    mem_145202.references = NULL;
    mem_145203.references = NULL;
    mem_145204.references = NULL;
    mem_145205.references = NULL;
    mem_145206.references = NULL;
    mem_145207.references = NULL;
    mem_145208.references = NULL;
    mem_145209.references = NULL;
    mem_145210.references = NULL;
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145202, (int64_t) 1728, "mem_145202")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147466 = 0; nest_i_147466 < (int64_t) 27; nest_i_147466++) {
        for (int64_t nest_i_147467 = 0; nest_i_147467 < (int64_t) 16; nest_i_147467++) {
            ((float *) mem_145202.mem)[nest_i_147466 * (int64_t) 16 + nest_i_147467] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145203, (int64_t) 1024, "mem_145203")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147468 = 0; nest_i_147468 < (int64_t) 16; nest_i_147468++) {
        for (int64_t nest_i_147469 = 0; nest_i_147469 < (int64_t) 16; nest_i_147469++) {
            ((float *) mem_145203.mem)[nest_i_147468 * (int64_t) 16 + nest_i_147469] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145204, (int64_t) 1024, "mem_145204")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147470 = 0; nest_i_147470 < (int64_t) 16; nest_i_147470++) {
        for (int64_t nest_i_147471 = 0; nest_i_147471 < (int64_t) 16; nest_i_147471++) {
            ((float *) mem_145204.mem)[nest_i_147470 * (int64_t) 16 + nest_i_147471] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145205, (int64_t) 1024, "mem_145205")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147472 = 0; nest_i_147472 < (int64_t) 16; nest_i_147472++) {
        for (int64_t nest_i_147473 = 0; nest_i_147473 < (int64_t) 16; nest_i_147473++) {
            ((float *) mem_145205.mem)[nest_i_147472 * (int64_t) 16 + nest_i_147473] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145206, (int64_t) 1024, "mem_145206")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147474 = 0; nest_i_147474 < (int64_t) 16; nest_i_147474++) {
        for (int64_t nest_i_147475 = 0; nest_i_147475 < (int64_t) 16; nest_i_147475++) {
            ((float *) mem_145206.mem)[nest_i_147474 * (int64_t) 16 + nest_i_147475] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145207, (int64_t) 1024, "mem_145207")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147476 = 0; nest_i_147476 < (int64_t) 16; nest_i_147476++) {
        for (int64_t nest_i_147477 = 0; nest_i_147477 < (int64_t) 16; nest_i_147477++) {
            ((float *) mem_145207.mem)[nest_i_147476 * (int64_t) 16 + nest_i_147477] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145208, (int64_t) 4096, "mem_145208")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147478 = 0; nest_i_147478 < (int64_t) 64; nest_i_147478++) {
        for (int64_t nest_i_147479 = 0; nest_i_147479 < (int64_t) 16; nest_i_147479++) {
            ((float *) mem_145208.mem)[nest_i_147478 * (int64_t) 16 + nest_i_147479] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145209, (int64_t) 4096, "mem_145209")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147480 = 0; nest_i_147480 < (int64_t) 16; nest_i_147480++) {
        for (int64_t nest_i_147481 = 0; nest_i_147481 < (int64_t) 64; nest_i_147481++) {
            ((float *) mem_145209.mem)[nest_i_147480 * (int64_t) 64 + nest_i_147481] = 0.0F;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145210, (int64_t) 1728, "mem_145210")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t nest_i_147482 = 0; nest_i_147482 < (int64_t) 27; nest_i_147482++) {
        for (int64_t nest_i_147483 = 0; nest_i_147483 < (int64_t) 16; nest_i_147483++) {
            ((float *) mem_145210.mem)[nest_i_147482 * (int64_t) 16 + nest_i_147483] = 0.0F;
        }
    }
    #undef mem_145202
    #undef mem_145203
    #undef mem_145204
    #undef mem_145205
    #undef mem_145206
    #undef mem_145207
    #undef mem_145208
    #undef mem_145209
    #undef mem_145210
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_145202, "ctx->constants->mem_145202") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145203, "ctx->constants->mem_145203") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145204, "ctx->constants->mem_145204") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145205, "ctx->constants->mem_145205") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145206, "ctx->constants->mem_145206") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145207, "ctx->constants->mem_145207") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145208, "ctx->constants->mem_145208") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145209, "ctx->constants->mem_145209") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_145210, "ctx->constants->mem_145210") != 0)
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
struct futhark_f32_2d {
    struct memblock mem;
    int64_t shape[2];
};
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * 4, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1) * 4 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1) * 4);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1)
{
    int err = 0;
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
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
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1]) * 4 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1]) * 4);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f32_2d(struct futhark_context *ctx, float *out, struct futhark_f32_2d *arr, int64_t i0, int64_t i1)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && (i1 >= 0 && i1 < arr->shape[1])) {
        lock_lock(&ctx->lock);
        if (4 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 4 * (i0 * arr->shape[1] + i1 * 1), 4);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f32_3d {
    struct memblock mem;
    int64_t shape[3];
};
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr = (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * arr->shape[2] * 4, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1 * dim2) * 4 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1 * dim2) * 4);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2)
{
    int err = 0;
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr = (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
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
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr, float *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 4 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) * 4);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f32_3d(struct futhark_context *ctx, float *out, struct futhark_f32_3d *arr, int64_t i0, int64_t i1, int64_t i2)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && ((i1 >= 0 && i1 < arr->shape[1]) && (i2 >= 0 && i2 < arr->shape[2]))) {
        lock_lock(&ctx->lock);
        if (4 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 4 * (i0 * (arr->shape[1] * arr->shape[2]) + i1 * arr->shape[2] + i2 * 1), 4);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f32_4d {
    struct memblock mem;
    int64_t shape[4];
};
struct futhark_f32_4d *futhark_new_f32_4d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
{
    int err = 0;
    struct futhark_f32_4d *bad = NULL;
    struct futhark_f32_4d *arr = (struct futhark_f32_4d *) malloc(sizeof(struct futhark_f32_4d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    arr->shape[3] = dim3;
    if (memblock_alloc(ctx, &arr->mem, arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3] * 4, "arr->mem"))
        err = 1;
    if ((size_t) (dim0 * dim1 * dim2 * dim3) * 4 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) (dim0 * dim1 * dim2 * dim3) * 4);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f32_4d *futhark_new_raw_f32_4d(struct futhark_context *ctx, unsigned char *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
{
    int err = 0;
    struct futhark_f32_4d *bad = NULL;
    struct futhark_f32_4d *arr = (struct futhark_f32_4d *) malloc(sizeof(struct futhark_f32_4d));
    
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
int futhark_free_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr, float *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3]) * 4 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2] * arr->shape[3]) * 4);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f32_4d(struct futhark_context *ctx, float *out, struct futhark_f32_4d *arr, int64_t i0, int64_t i1, int64_t i2, int64_t i3)
{
    int err = 0;
    
    if ((i0 >= 0 && i0 < arr->shape[0]) && ((i1 >= 0 && i1 < arr->shape[1]) && ((i2 >= 0 && i2 < arr->shape[2]) && (i3 >= 0 && i3 < arr->shape[3])))) {
        lock_lock(&ctx->lock);
        if (4 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 4 * (i0 * (arr->shape[1] * arr->shape[2] * arr->shape[3]) + i1 * (arr->shape[2] * arr->shape[3]) + i2 * arr->shape[3] + i3 * 1), 4);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_4d(struct futhark_context *ctx, struct futhark_f32_4d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_opaque_params {
    struct futhark_f32_2d *v0;
    struct futhark_f32_2d *v1;
    struct futhark_f32_2d *v2;
    struct futhark_f32_2d *v3;
    struct futhark_f32_2d *v4;
    struct futhark_f32_2d *v5;
    struct futhark_f32_2d *v6;
    struct futhark_f32_2d *v7;
    struct futhark_f32_2d *v8;
};
int futhark_project_opaque_params_wdown(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v0, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wkey(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v1, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wout(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v2, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wpe(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v3, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wqry(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v4, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wte(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v5, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wup(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v6, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wval(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v7, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_params_wvoc(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_params *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v8, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f32_2d *f_wdown, const struct futhark_f32_2d *f_wkey, const struct futhark_f32_2d *f_wout, const struct futhark_f32_2d *f_wpe, const struct futhark_f32_2d *f_wqry, const struct futhark_f32_2d *f_wte, const struct futhark_f32_2d *f_wup, const struct futhark_f32_2d *f_wval, const struct futhark_f32_2d *f_wvoc)
{
    struct futhark_opaque_params *v = malloc(sizeof(struct futhark_opaque_params));
    
    lock_lock(&ctx->lock);
    {
        v->v0 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v0, f_wdown, sizeof(struct futhark_f32_2d));
        (void) (*v->v0->mem.references)++;
    }
    {
        v->v1 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v1, f_wkey, sizeof(struct futhark_f32_2d));
        (void) (*v->v1->mem.references)++;
    }
    {
        v->v2 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v2, f_wout, sizeof(struct futhark_f32_2d));
        (void) (*v->v2->mem.references)++;
    }
    {
        v->v3 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v3, f_wpe, sizeof(struct futhark_f32_2d));
        (void) (*v->v3->mem.references)++;
    }
    {
        v->v4 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v4, f_wqry, sizeof(struct futhark_f32_2d));
        (void) (*v->v4->mem.references)++;
    }
    {
        v->v5 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v5, f_wte, sizeof(struct futhark_f32_2d));
        (void) (*v->v5->mem.references)++;
    }
    {
        v->v6 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v6, f_wup, sizeof(struct futhark_f32_2d));
        (void) (*v->v6->mem.references)++;
    }
    {
        v->v7 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v7, f_wval, sizeof(struct futhark_f32_2d));
        (void) (*v->v7->mem.references)++;
    }
    {
        v->v8 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v8, f_wvoc, sizeof(struct futhark_f32_2d));
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
    
    if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
        ret = tmp;
    if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
        ret = tmp;
    if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
        ret = tmp;
    if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
        ret = tmp;
    if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
        ret = tmp;
    if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
        ret = tmp;
    if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
        ret = tmp;
    if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
        ret = tmp;
    if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_params(struct futhark_context *ctx, const struct futhark_opaque_params *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
    int64_t size_1 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
    int64_t size_2 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
    int64_t size_3 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
    int64_t size_4 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
    int64_t size_5 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
    int64_t size_6 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
    int64_t size_7 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
    int64_t size_8 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v0), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v0, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v1), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v2), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v2, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v3), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v3, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v4), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v4, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v5), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v5, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v6), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v6, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v7), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v7, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v8), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v8, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
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
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_0, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    obj->v0 = NULL;
    src += shape_0[0] * shape_0[1] * sizeof(float);
    
    int64_t shape_1[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * shape_1[1] * sizeof(float);
    
    int64_t shape_2[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_2, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_2 = src;
    
    obj->v2 = NULL;
    src += shape_2[0] * shape_2[1] * sizeof(float);
    
    int64_t shape_3[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_3, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_3 = src;
    
    obj->v3 = NULL;
    src += shape_3[0] * shape_3[1] * sizeof(float);
    
    int64_t shape_4[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_4, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_4 = src;
    
    obj->v4 = NULL;
    src += shape_4[0] * shape_4[1] * sizeof(float);
    
    int64_t shape_5[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_5, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_5 = src;
    
    obj->v5 = NULL;
    src += shape_5[0] * shape_5[1] * sizeof(float);
    
    int64_t shape_6[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_6, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_6 = src;
    
    obj->v6 = NULL;
    src += shape_6[0] * shape_6[1] * sizeof(float);
    
    int64_t shape_7[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_7, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_7 = src;
    
    obj->v7 = NULL;
    src += shape_7[0] * shape_7[1] * sizeof(float);
    
    int64_t shape_8[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_8, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_8 = src;
    
    obj->v8 = NULL;
    src += shape_8[0] * shape_8[1] * sizeof(float);
    if (err == 0) {
        obj->v0 = futhark_new_f32_2d(ctx, data_0, shape_0[0], shape_0[1]);
        if (obj->v0 == NULL)
            err = 1;
        obj->v1 = futhark_new_f32_2d(ctx, data_1, shape_1[0], shape_1[1]);
        if (obj->v1 == NULL)
            err = 1;
        obj->v2 = futhark_new_f32_2d(ctx, data_2, shape_2[0], shape_2[1]);
        if (obj->v2 == NULL)
            err = 1;
        obj->v3 = futhark_new_f32_2d(ctx, data_3, shape_3[0], shape_3[1]);
        if (obj->v3 == NULL)
            err = 1;
        obj->v4 = futhark_new_f32_2d(ctx, data_4, shape_4[0], shape_4[1]);
        if (obj->v4 == NULL)
            err = 1;
        obj->v5 = futhark_new_f32_2d(ctx, data_5, shape_5[0], shape_5[1]);
        if (obj->v5 == NULL)
            err = 1;
        obj->v6 = futhark_new_f32_2d(ctx, data_6, shape_6[0], shape_6[1]);
        if (obj->v6 == NULL)
            err = 1;
        obj->v7 = futhark_new_f32_2d(ctx, data_7, shape_7[0], shape_7[1]);
        if (obj->v7 == NULL)
            err = 1;
        obj->v8 = futhark_new_f32_2d(ctx, data_8, shape_8[0], shape_8[1]);
        if (obj->v8 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
            ret = tmp;
        if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
            ret = tmp;
        if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
            ret = tmp;
        if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
            ret = tmp;
        if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
            ret = tmp;
        if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
            ret = tmp;
        if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
            ret = tmp;
        if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
            ret = tmp;
        if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}
struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 {
    struct futhark_f32_2d *v0;
    struct futhark_f32_2d *v1;
    struct futhark_f32_2d *v2;
    struct futhark_f32_2d *v3;
    struct futhark_f32_2d *v4;
    struct futhark_f32_2d *v5;
    struct futhark_f32_2d *v6;
    struct futhark_f32_2d *v7;
    struct futhark_f32_2d *v8;
};
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v0, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v1, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v2, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_3(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v3, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_4(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v4, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_5(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v5, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_6(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v6, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_7(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v7, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_project_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_8(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_f32_2d *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_f32_2d));
    memcpy(v, obj->v8, sizeof(struct futhark_f32_2d));
    (void) (*v->mem.references)++;
    lock_unlock(&ctx->lock);
    *out = v;
    return 0;
}
int futhark_new_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_f32_2d *f_0, const struct futhark_f32_2d *f_1, const struct futhark_f32_2d *f_2, const struct futhark_f32_2d *f_3, const struct futhark_f32_2d *f_4, const struct futhark_f32_2d *f_5, const struct futhark_f32_2d *f_6, const struct futhark_f32_2d *f_7, const struct futhark_f32_2d *f_8)
{
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
    
    lock_lock(&ctx->lock);
    {
        v->v0 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v0, f_0, sizeof(struct futhark_f32_2d));
        (void) (*v->v0->mem.references)++;
    }
    {
        v->v1 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v1, f_1, sizeof(struct futhark_f32_2d));
        (void) (*v->v1->mem.references)++;
    }
    {
        v->v2 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v2, f_2, sizeof(struct futhark_f32_2d));
        (void) (*v->v2->mem.references)++;
    }
    {
        v->v3 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v3, f_3, sizeof(struct futhark_f32_2d));
        (void) (*v->v3->mem.references)++;
    }
    {
        v->v4 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v4, f_4, sizeof(struct futhark_f32_2d));
        (void) (*v->v4->mem.references)++;
    }
    {
        v->v5 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v5, f_5, sizeof(struct futhark_f32_2d));
        (void) (*v->v5->mem.references)++;
    }
    {
        v->v6 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v6, f_6, sizeof(struct futhark_f32_2d));
        (void) (*v->v6->mem.references)++;
    }
    {
        v->v7 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v7, f_7, sizeof(struct futhark_f32_2d));
        (void) (*v->v7->mem.references)++;
    }
    {
        v->v8 = malloc(sizeof(struct futhark_f32_2d));
        memcpy(v->v8, f_8, sizeof(struct futhark_f32_2d));
        (void) (*v->v8->mem.references)++;
    }
    lock_unlock(&ctx->lock);
    *out = v;
    return FUTHARK_SUCCESS;
}
int futhark_free_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    int ret = 0, tmp;
    
    if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
        ret = tmp;
    if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
        ret = tmp;
    if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
        ret = tmp;
    if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
        ret = tmp;
    if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
        ret = tmp;
    if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
        ret = tmp;
    if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
        ret = tmp;
    if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
        ret = tmp;
    if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
    int64_t size_1 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
    int64_t size_2 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
    int64_t size_3 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
    int64_t size_4 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
    int64_t size_5 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
    int64_t size_6 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
    int64_t size_7 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
    int64_t size_8 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v0), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v0, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v1), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v2), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v2, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v3), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v3, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v4), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v4, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v5), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v5, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v6), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v6, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v7), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v7, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v8), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v8, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
    }
    return ret;
}
struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *futhark_restore_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
    int64_t shape_0[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_0, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    obj->v0 = NULL;
    src += shape_0[0] * shape_0[1] * sizeof(float);
    
    int64_t shape_1[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * shape_1[1] * sizeof(float);
    
    int64_t shape_2[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_2, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_2 = src;
    
    obj->v2 = NULL;
    src += shape_2[0] * shape_2[1] * sizeof(float);
    
    int64_t shape_3[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_3, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_3 = src;
    
    obj->v3 = NULL;
    src += shape_3[0] * shape_3[1] * sizeof(float);
    
    int64_t shape_4[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_4, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_4 = src;
    
    obj->v4 = NULL;
    src += shape_4[0] * shape_4[1] * sizeof(float);
    
    int64_t shape_5[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_5, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_5 = src;
    
    obj->v5 = NULL;
    src += shape_5[0] * shape_5[1] * sizeof(float);
    
    int64_t shape_6[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_6, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_6 = src;
    
    obj->v6 = NULL;
    src += shape_6[0] * shape_6[1] * sizeof(float);
    
    int64_t shape_7[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_7, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_7 = src;
    
    obj->v7 = NULL;
    src += shape_7[0] * shape_7[1] * sizeof(float);
    
    int64_t shape_8[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_8, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_8 = src;
    
    obj->v8 = NULL;
    src += shape_8[0] * shape_8[1] * sizeof(float);
    if (err == 0) {
        obj->v0 = futhark_new_f32_2d(ctx, data_0, shape_0[0], shape_0[1]);
        if (obj->v0 == NULL)
            err = 1;
        obj->v1 = futhark_new_f32_2d(ctx, data_1, shape_1[0], shape_1[1]);
        if (obj->v1 == NULL)
            err = 1;
        obj->v2 = futhark_new_f32_2d(ctx, data_2, shape_2[0], shape_2[1]);
        if (obj->v2 == NULL)
            err = 1;
        obj->v3 = futhark_new_f32_2d(ctx, data_3, shape_3[0], shape_3[1]);
        if (obj->v3 == NULL)
            err = 1;
        obj->v4 = futhark_new_f32_2d(ctx, data_4, shape_4[0], shape_4[1]);
        if (obj->v4 == NULL)
            err = 1;
        obj->v5 = futhark_new_f32_2d(ctx, data_5, shape_5[0], shape_5[1]);
        if (obj->v5 == NULL)
            err = 1;
        obj->v6 = futhark_new_f32_2d(ctx, data_6, shape_6[0], shape_6[1]);
        if (obj->v6 == NULL)
            err = 1;
        obj->v7 = futhark_new_f32_2d(ctx, data_7, shape_7[0], shape_7[1]);
        if (obj->v7 == NULL)
            err = 1;
        obj->v8 = futhark_new_f32_2d(ctx, data_8, shape_8[0], shape_8[1]);
        if (obj->v8 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
            ret = tmp;
        if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
            ret = tmp;
        if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
            ret = tmp;
        if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
            ret = tmp;
        if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
            ret = tmp;
        if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
            ret = tmp;
        if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
            ret = tmp;
        if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
            ret = tmp;
        if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}
struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 {
    struct futhark_f32_2d *v0;
    struct futhark_f32_2d *v1;
    struct futhark_f32_2d *v2;
    struct futhark_f32_2d *v3;
    struct futhark_f32_2d *v4;
    struct futhark_f32_2d *v5;
    struct futhark_f32_2d *v6;
    struct futhark_f32_2d *v7;
    struct futhark_f32_2d *v8;
    struct futhark_f32_2d *v9;
    struct futhark_f32_2d *v10;
    struct futhark_f32_2d *v11;
    struct futhark_f32_2d *v12;
    struct futhark_f32_2d *v13;
    struct futhark_f32_2d *v14;
    struct futhark_f32_2d *v15;
    struct futhark_f32_2d *v16;
    struct futhark_f32_2d *v17;
    struct futhark_f32_2d *v18;
    struct futhark_f32_2d *v19;
    struct futhark_f32_2d *v20;
    struct futhark_f32_2d *v21;
    struct futhark_f32_2d *v22;
    struct futhark_f32_2d *v23;
    struct futhark_f32_2d *v24;
    struct futhark_f32_2d *v25;
    struct futhark_f32_2d *v26;
};
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_0(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
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
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_1(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
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
int futhark_project_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_2(struct futhark_context *ctx, struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *v;
    
    lock_lock(&ctx->lock);
    v = malloc(sizeof(struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
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
int futhark_new_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_0, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_1, const struct futhark_opaque_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *f_2)
{
    struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *v = malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
    
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
int futhark_free_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj)
{
    (void) ctx;
    
    int ret = 0, tmp;
    
    if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
        ret = tmp;
    if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
        ret = tmp;
    if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
        ret = tmp;
    if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
        ret = tmp;
    if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
        ret = tmp;
    if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
        ret = tmp;
    if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
        ret = tmp;
    if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
        ret = tmp;
    if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
        ret = tmp;
    if (obj->v9 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v9)) != 0)
        ret = tmp;
    if (obj->v10 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v10)) != 0)
        ret = tmp;
    if (obj->v11 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v11)) != 0)
        ret = tmp;
    if (obj->v12 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v12)) != 0)
        ret = tmp;
    if (obj->v13 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v13)) != 0)
        ret = tmp;
    if (obj->v14 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v14)) != 0)
        ret = tmp;
    if (obj->v15 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v15)) != 0)
        ret = tmp;
    if (obj->v16 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v16)) != 0)
        ret = tmp;
    if (obj->v17 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v17)) != 0)
        ret = tmp;
    if (obj->v18 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v18)) != 0)
        ret = tmp;
    if (obj->v19 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v19)) != 0)
        ret = tmp;
    if (obj->v20 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v20)) != 0)
        ret = tmp;
    if (obj->v21 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v21)) != 0)
        ret = tmp;
    if (obj->v22 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v22)) != 0)
        ret = tmp;
    if (obj->v23 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v23)) != 0)
        ret = tmp;
    if (obj->v24 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v24)) != 0)
        ret = tmp;
    if (obj->v25 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v25)) != 0)
        ret = tmp;
    if (obj->v26 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v26)) != 0)
        ret = tmp;
    free(obj);
    return ret;
}
int futhark_store_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj, void **p, size_t *n)
{
    (void) ctx;
    
    int ret = 0;
    int64_t size_0 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
    int64_t size_1 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
    int64_t size_2 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
    int64_t size_3 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
    int64_t size_4 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
    int64_t size_5 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
    int64_t size_6 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
    int64_t size_7 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
    int64_t size_8 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
    int64_t size_9 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v9)[0] * futhark_shape_f32_2d(ctx, obj->v9)[1] * sizeof(float);
    int64_t size_10 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v10)[0] * futhark_shape_f32_2d(ctx, obj->v10)[1] * sizeof(float);
    int64_t size_11 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v11)[0] * futhark_shape_f32_2d(ctx, obj->v11)[1] * sizeof(float);
    int64_t size_12 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v12)[0] * futhark_shape_f32_2d(ctx, obj->v12)[1] * sizeof(float);
    int64_t size_13 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v13)[0] * futhark_shape_f32_2d(ctx, obj->v13)[1] * sizeof(float);
    int64_t size_14 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v14)[0] * futhark_shape_f32_2d(ctx, obj->v14)[1] * sizeof(float);
    int64_t size_15 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v15)[0] * futhark_shape_f32_2d(ctx, obj->v15)[1] * sizeof(float);
    int64_t size_16 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v16)[0] * futhark_shape_f32_2d(ctx, obj->v16)[1] * sizeof(float);
    int64_t size_17 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v17)[0] * futhark_shape_f32_2d(ctx, obj->v17)[1] * sizeof(float);
    int64_t size_18 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v18)[0] * futhark_shape_f32_2d(ctx, obj->v18)[1] * sizeof(float);
    int64_t size_19 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v19)[0] * futhark_shape_f32_2d(ctx, obj->v19)[1] * sizeof(float);
    int64_t size_20 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v20)[0] * futhark_shape_f32_2d(ctx, obj->v20)[1] * sizeof(float);
    int64_t size_21 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v21)[0] * futhark_shape_f32_2d(ctx, obj->v21)[1] * sizeof(float);
    int64_t size_22 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v22)[0] * futhark_shape_f32_2d(ctx, obj->v22)[1] * sizeof(float);
    int64_t size_23 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v23)[0] * futhark_shape_f32_2d(ctx, obj->v23)[1] * sizeof(float);
    int64_t size_24 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v24)[0] * futhark_shape_f32_2d(ctx, obj->v24)[1] * sizeof(float);
    int64_t size_25 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v25)[0] * futhark_shape_f32_2d(ctx, obj->v25)[1] * sizeof(float);
    int64_t size_26 = 7 + 2 * sizeof(int64_t) + futhark_shape_f32_2d(ctx, obj->v26)[0] * futhark_shape_f32_2d(ctx, obj->v26)[1] * sizeof(float);
    
    *n = size_0 + size_1 + size_2 + size_3 + size_4 + size_5 + size_6 + size_7 + size_8 + size_9 + size_10 + size_11 + size_12 + size_13 + size_14 + size_15 + size_16 + size_17 + size_18 + size_19 + size_20 + size_21 + size_22 + size_23 + size_24 + size_25 + size_26;
    if (p != NULL && *p == NULL)
        *p = malloc(*n);
    if (p != NULL) {
        unsigned char *out = *p;
        
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v0), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v0, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v0)[0] * futhark_shape_f32_2d(ctx, obj->v0)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v1), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v1, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v1)[0] * futhark_shape_f32_2d(ctx, obj->v1)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v2), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v2, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v2)[0] * futhark_shape_f32_2d(ctx, obj->v2)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v3), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v3, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v3)[0] * futhark_shape_f32_2d(ctx, obj->v3)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v4), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v4, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v4)[0] * futhark_shape_f32_2d(ctx, obj->v4)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v5), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v5, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v5)[0] * futhark_shape_f32_2d(ctx, obj->v5)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v6), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v6, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v6)[0] * futhark_shape_f32_2d(ctx, obj->v6)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v7), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v7, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v7)[0] * futhark_shape_f32_2d(ctx, obj->v7)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v8), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v8, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v8)[0] * futhark_shape_f32_2d(ctx, obj->v8)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v9), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v9, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v9)[0] * futhark_shape_f32_2d(ctx, obj->v9)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v10), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v10, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v10)[0] * futhark_shape_f32_2d(ctx, obj->v10)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v11), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v11, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v11)[0] * futhark_shape_f32_2d(ctx, obj->v11)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v12), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v12, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v12)[0] * futhark_shape_f32_2d(ctx, obj->v12)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v13), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v13, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v13)[0] * futhark_shape_f32_2d(ctx, obj->v13)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v14), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v14, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v14)[0] * futhark_shape_f32_2d(ctx, obj->v14)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v15), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v15, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v15)[0] * futhark_shape_f32_2d(ctx, obj->v15)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v16), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v16, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v16)[0] * futhark_shape_f32_2d(ctx, obj->v16)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v17), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v17, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v17)[0] * futhark_shape_f32_2d(ctx, obj->v17)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v18), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v18, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v18)[0] * futhark_shape_f32_2d(ctx, obj->v18)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v19), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v19, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v19)[0] * futhark_shape_f32_2d(ctx, obj->v19)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v20), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v20, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v20)[0] * futhark_shape_f32_2d(ctx, obj->v20)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v21), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v21, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v21)[0] * futhark_shape_f32_2d(ctx, obj->v21)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v22), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v22, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v22)[0] * futhark_shape_f32_2d(ctx, obj->v22)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v23), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v23, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v23)[0] * futhark_shape_f32_2d(ctx, obj->v23)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v24), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v24, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v24)[0] * futhark_shape_f32_2d(ctx, obj->v24)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v25), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v25, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v25)[0] * futhark_shape_f32_2d(ctx, obj->v25)[1] * sizeof(float);
        *out++ = 'b';
        *out++ = 2;
        *out++ = 2;
        memcpy(out, " f32", 4);
        out += 4;
        memcpy(out, futhark_shape_f32_2d(ctx, obj->v26), 2 * sizeof(int64_t));
        out += 2 * sizeof(int64_t);
        ret |= futhark_values_f32_2d(ctx, obj->v26, (void *) out);
        out += futhark_shape_f32_2d(ctx, obj->v26)[0] * futhark_shape_f32_2d(ctx, obj->v26)[1] * sizeof(float);
    }
    return ret;
}
struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *futhark_restore_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32(struct futhark_context *ctx, const void *p)
{
    (void) ctx;
    
    int err = 0;
    const unsigned char *src = p;
    struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *obj = malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32));
    int64_t shape_0[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_0, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_0 = src;
    
    obj->v0 = NULL;
    src += shape_0[0] * shape_0[1] * sizeof(float);
    
    int64_t shape_1[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_1, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_1 = src;
    
    obj->v1 = NULL;
    src += shape_1[0] * shape_1[1] * sizeof(float);
    
    int64_t shape_2[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_2, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_2 = src;
    
    obj->v2 = NULL;
    src += shape_2[0] * shape_2[1] * sizeof(float);
    
    int64_t shape_3[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_3, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_3 = src;
    
    obj->v3 = NULL;
    src += shape_3[0] * shape_3[1] * sizeof(float);
    
    int64_t shape_4[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_4, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_4 = src;
    
    obj->v4 = NULL;
    src += shape_4[0] * shape_4[1] * sizeof(float);
    
    int64_t shape_5[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_5, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_5 = src;
    
    obj->v5 = NULL;
    src += shape_5[0] * shape_5[1] * sizeof(float);
    
    int64_t shape_6[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_6, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_6 = src;
    
    obj->v6 = NULL;
    src += shape_6[0] * shape_6[1] * sizeof(float);
    
    int64_t shape_7[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_7, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_7 = src;
    
    obj->v7 = NULL;
    src += shape_7[0] * shape_7[1] * sizeof(float);
    
    int64_t shape_8[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_8, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_8 = src;
    
    obj->v8 = NULL;
    src += shape_8[0] * shape_8[1] * sizeof(float);
    
    int64_t shape_9[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_9, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_9 = src;
    
    obj->v9 = NULL;
    src += shape_9[0] * shape_9[1] * sizeof(float);
    
    int64_t shape_10[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_10, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_10 = src;
    
    obj->v10 = NULL;
    src += shape_10[0] * shape_10[1] * sizeof(float);
    
    int64_t shape_11[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_11, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_11 = src;
    
    obj->v11 = NULL;
    src += shape_11[0] * shape_11[1] * sizeof(float);
    
    int64_t shape_12[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_12, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_12 = src;
    
    obj->v12 = NULL;
    src += shape_12[0] * shape_12[1] * sizeof(float);
    
    int64_t shape_13[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_13, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_13 = src;
    
    obj->v13 = NULL;
    src += shape_13[0] * shape_13[1] * sizeof(float);
    
    int64_t shape_14[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_14, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_14 = src;
    
    obj->v14 = NULL;
    src += shape_14[0] * shape_14[1] * sizeof(float);
    
    int64_t shape_15[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_15, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_15 = src;
    
    obj->v15 = NULL;
    src += shape_15[0] * shape_15[1] * sizeof(float);
    
    int64_t shape_16[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_16, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_16 = src;
    
    obj->v16 = NULL;
    src += shape_16[0] * shape_16[1] * sizeof(float);
    
    int64_t shape_17[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_17, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_17 = src;
    
    obj->v17 = NULL;
    src += shape_17[0] * shape_17[1] * sizeof(float);
    
    int64_t shape_18[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_18, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_18 = src;
    
    obj->v18 = NULL;
    src += shape_18[0] * shape_18[1] * sizeof(float);
    
    int64_t shape_19[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_19, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_19 = src;
    
    obj->v19 = NULL;
    src += shape_19[0] * shape_19[1] * sizeof(float);
    
    int64_t shape_20[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_20, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_20 = src;
    
    obj->v20 = NULL;
    src += shape_20[0] * shape_20[1] * sizeof(float);
    
    int64_t shape_21[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_21, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_21 = src;
    
    obj->v21 = NULL;
    src += shape_21[0] * shape_21[1] * sizeof(float);
    
    int64_t shape_22[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_22, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_22 = src;
    
    obj->v22 = NULL;
    src += shape_22[0] * shape_22[1] * sizeof(float);
    
    int64_t shape_23[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_23, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_23 = src;
    
    obj->v23 = NULL;
    src += shape_23[0] * shape_23[1] * sizeof(float);
    
    int64_t shape_24[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_24, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_24 = src;
    
    obj->v24 = NULL;
    src += shape_24[0] * shape_24[1] * sizeof(float);
    
    int64_t shape_25[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_25, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_25 = src;
    
    obj->v25 = NULL;
    src += shape_25[0] * shape_25[1] * sizeof(float);
    
    int64_t shape_26[2] = {0};
    
    err |= *src++ != 'b';
    err |= *src++ != 2;
    err |= *src++ != 2;
    err |= memcmp(src, " f32", 4) != 0;
    src += 4;
    if (err == 0) {
        memcpy(shape_26, src, 2 * sizeof(int64_t));
        src += 2 * sizeof(int64_t);
    }
    
    const void *data_26 = src;
    
    obj->v26 = NULL;
    src += shape_26[0] * shape_26[1] * sizeof(float);
    if (err == 0) {
        obj->v0 = futhark_new_f32_2d(ctx, data_0, shape_0[0], shape_0[1]);
        if (obj->v0 == NULL)
            err = 1;
        obj->v1 = futhark_new_f32_2d(ctx, data_1, shape_1[0], shape_1[1]);
        if (obj->v1 == NULL)
            err = 1;
        obj->v2 = futhark_new_f32_2d(ctx, data_2, shape_2[0], shape_2[1]);
        if (obj->v2 == NULL)
            err = 1;
        obj->v3 = futhark_new_f32_2d(ctx, data_3, shape_3[0], shape_3[1]);
        if (obj->v3 == NULL)
            err = 1;
        obj->v4 = futhark_new_f32_2d(ctx, data_4, shape_4[0], shape_4[1]);
        if (obj->v4 == NULL)
            err = 1;
        obj->v5 = futhark_new_f32_2d(ctx, data_5, shape_5[0], shape_5[1]);
        if (obj->v5 == NULL)
            err = 1;
        obj->v6 = futhark_new_f32_2d(ctx, data_6, shape_6[0], shape_6[1]);
        if (obj->v6 == NULL)
            err = 1;
        obj->v7 = futhark_new_f32_2d(ctx, data_7, shape_7[0], shape_7[1]);
        if (obj->v7 == NULL)
            err = 1;
        obj->v8 = futhark_new_f32_2d(ctx, data_8, shape_8[0], shape_8[1]);
        if (obj->v8 == NULL)
            err = 1;
        obj->v9 = futhark_new_f32_2d(ctx, data_9, shape_9[0], shape_9[1]);
        if (obj->v9 == NULL)
            err = 1;
        obj->v10 = futhark_new_f32_2d(ctx, data_10, shape_10[0], shape_10[1]);
        if (obj->v10 == NULL)
            err = 1;
        obj->v11 = futhark_new_f32_2d(ctx, data_11, shape_11[0], shape_11[1]);
        if (obj->v11 == NULL)
            err = 1;
        obj->v12 = futhark_new_f32_2d(ctx, data_12, shape_12[0], shape_12[1]);
        if (obj->v12 == NULL)
            err = 1;
        obj->v13 = futhark_new_f32_2d(ctx, data_13, shape_13[0], shape_13[1]);
        if (obj->v13 == NULL)
            err = 1;
        obj->v14 = futhark_new_f32_2d(ctx, data_14, shape_14[0], shape_14[1]);
        if (obj->v14 == NULL)
            err = 1;
        obj->v15 = futhark_new_f32_2d(ctx, data_15, shape_15[0], shape_15[1]);
        if (obj->v15 == NULL)
            err = 1;
        obj->v16 = futhark_new_f32_2d(ctx, data_16, shape_16[0], shape_16[1]);
        if (obj->v16 == NULL)
            err = 1;
        obj->v17 = futhark_new_f32_2d(ctx, data_17, shape_17[0], shape_17[1]);
        if (obj->v17 == NULL)
            err = 1;
        obj->v18 = futhark_new_f32_2d(ctx, data_18, shape_18[0], shape_18[1]);
        if (obj->v18 == NULL)
            err = 1;
        obj->v19 = futhark_new_f32_2d(ctx, data_19, shape_19[0], shape_19[1]);
        if (obj->v19 == NULL)
            err = 1;
        obj->v20 = futhark_new_f32_2d(ctx, data_20, shape_20[0], shape_20[1]);
        if (obj->v20 == NULL)
            err = 1;
        obj->v21 = futhark_new_f32_2d(ctx, data_21, shape_21[0], shape_21[1]);
        if (obj->v21 == NULL)
            err = 1;
        obj->v22 = futhark_new_f32_2d(ctx, data_22, shape_22[0], shape_22[1]);
        if (obj->v22 == NULL)
            err = 1;
        obj->v23 = futhark_new_f32_2d(ctx, data_23, shape_23[0], shape_23[1]);
        if (obj->v23 == NULL)
            err = 1;
        obj->v24 = futhark_new_f32_2d(ctx, data_24, shape_24[0], shape_24[1]);
        if (obj->v24 == NULL)
            err = 1;
        obj->v25 = futhark_new_f32_2d(ctx, data_25, shape_25[0], shape_25[1]);
        if (obj->v25 == NULL)
            err = 1;
        obj->v26 = futhark_new_f32_2d(ctx, data_26, shape_26[0], shape_26[1]);
        if (obj->v26 == NULL)
            err = 1;
    }
    if (err != 0) {
        int ret = 0, tmp;
        
        if (obj->v0 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v0)) != 0)
            ret = tmp;
        if (obj->v1 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v1)) != 0)
            ret = tmp;
        if (obj->v2 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v2)) != 0)
            ret = tmp;
        if (obj->v3 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v3)) != 0)
            ret = tmp;
        if (obj->v4 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v4)) != 0)
            ret = tmp;
        if (obj->v5 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v5)) != 0)
            ret = tmp;
        if (obj->v6 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v6)) != 0)
            ret = tmp;
        if (obj->v7 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v7)) != 0)
            ret = tmp;
        if (obj->v8 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v8)) != 0)
            ret = tmp;
        if (obj->v9 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v9)) != 0)
            ret = tmp;
        if (obj->v10 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v10)) != 0)
            ret = tmp;
        if (obj->v11 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v11)) != 0)
            ret = tmp;
        if (obj->v12 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v12)) != 0)
            ret = tmp;
        if (obj->v13 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v13)) != 0)
            ret = tmp;
        if (obj->v14 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v14)) != 0)
            ret = tmp;
        if (obj->v15 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v15)) != 0)
            ret = tmp;
        if (obj->v16 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v16)) != 0)
            ret = tmp;
        if (obj->v17 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v17)) != 0)
            ret = tmp;
        if (obj->v18 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v18)) != 0)
            ret = tmp;
        if (obj->v19 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v19)) != 0)
            ret = tmp;
        if (obj->v20 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v20)) != 0)
            ret = tmp;
        if (obj->v21 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v21)) != 0)
            ret = tmp;
        if (obj->v22 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v22)) != 0)
            ret = tmp;
        if (obj->v23 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v23)) != 0)
            ret = tmp;
        if (obj->v24 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v24)) != 0)
            ret = tmp;
        if (obj->v25 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v25)) != 0)
            ret = tmp;
        if (obj->v26 != NULL && (tmp = futhark_free_f32_2d(ctx, obj->v26)) != 0)
            ret = tmp;
        free(obj);
        obj = NULL;
    }
    return obj;
}

FUTHARK_FUN_ATTR int futrts_adam_opt_w_12790(struct futhark_context *ctx, struct memblock *mem_out_p_147484, struct memblock *mem_out_p_147485, struct memblock *mem_out_p_147486, struct memblock w_mem_145211, struct memblock mw_mem_145212, struct memblock vw_mem_145213, struct memblock dw_mem_145214, int64_t n_119001, int64_t m_119002, int64_t step_119007, float lt_r_119008)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145232_cached_sizze_147487 = 0;
    unsigned char *mem_145232 = NULL;
    int64_t mem_145255_cached_sizze_147488 = 0;
    unsigned char *mem_145255 = NULL;
    int64_t mem_145258_cached_sizze_147489 = 0;
    unsigned char *mem_145258 = NULL;
    struct memblock mem_145293;
    
    mem_145293.references = NULL;
    
    struct memblock mem_145220;
    
    mem_145220.references = NULL;
    
    struct memblock mem_145217;
    
    mem_145217.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_145215 = (int64_t) 4 * n_119001;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_145216 = m_119002 * binop_x_145215;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_145229 = (int64_t) 4 * m_119002;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145217, bytes_145216, "mem_145217")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145220, bytes_145216, "mem_145220")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145232_cached_sizze_147487 < bytes_145229) {
        err = lexical_realloc(ctx, &mem_145232, &mem_145232_cached_sizze_147487, bytes_145229);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144256 = 0; i_144256 < n_119001; i_144256++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144249 = 0; i_144249 < m_119002; i_144249++) {
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141002 = ((float *) mw_mem_145212.mem)[i_144256 * m_119002 + i_144249];
            
            // futhark/microgpt.fut:430:10-20
            
            float zp_lhs_141003 = 0.85F * zt_rhs_141002;
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141004 = ((float *) dw_mem_145214.mem)[i_144256 * m_119002 + i_144249];
            
            // futhark/microgpt.fut:430:35-45
            
            float zp_rhs_141005 = 0.14999998F * zt_rhs_141004;
            
            // futhark/microgpt.fut:430:21-45
            
            float lifted_lambda_res_141006 = zp_lhs_141003 + zp_rhs_141005;
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141013 = ((float *) vw_mem_145213.mem)[i_144256 * m_119002 + i_144249];
            
            // futhark/microgpt.fut:432:10-20
            
            float zp_lhs_141014 = 0.99F * zt_rhs_141013;
            
            // futhark/microgpt.fut:432:35-45
            
            float zt_lhs_141016 = 9.99999e-3F * zt_rhs_141004;
            
            // futhark/microgpt.fut:432:46-56
            
            float zp_rhs_141017 = zt_rhs_141004 * zt_lhs_141016;
            
            // futhark/microgpt.fut:432:21-56
            
            float lifted_lambda_res_141018 = zp_lhs_141014 + zp_rhs_141017;
            
            ((float *) mem_145217.mem)[i_144256 * m_119002 + i_144249] = lifted_lambda_res_141018;
            ((float *) mem_145232)[i_144249] = lifted_lambda_res_141006;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145220.mem, i_144256 * m_119002, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145232, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_119002});
    }
    // futhark/microgpt.fut:66:26-45
    
    float i64_res_123659 = sitofp_i64_f32(step_119007);
    
    // futhark/microgpt.fut:434:54-57
    
    float ztzt_rhs_123660 = 1.0F + i64_res_123659;
    
    // futhark/microgpt.fut:434:30-57
    
    float zm_rhs_123661 = fpow32(0.85F, ztzt_rhs_123660);
    
    // futhark/microgpt.fut:434:23-57
    
    float zs_rhs_123662 = 1.0F - zm_rhs_123661;
    
    // futhark/microgpt.fut:436:31-58
    
    float zm_rhs_123700 = fpow32(0.99F, ztzt_rhs_123660);
    
    // futhark/microgpt.fut:436:23-58
    
    float zs_rhs_123701 = 1.0F - zm_rhs_123700;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_145255_cached_sizze_147488 < bytes_145216) {
        err = lexical_realloc(ctx, &mem_145255, &mem_145255_cached_sizze_147488, bytes_145216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145258_cached_sizze_147489 < bytes_145216) {
        err = lexical_realloc(ctx, &mem_145258, &mem_145258_cached_sizze_147489, bytes_145216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144270 = 0; i_144270 < n_119001; i_144270++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144263 = 0; i_144263 < m_119002; i_144263++) {
            // futhark/microgpt.fut:4:11-25
            
            float zs_lhs_141038 = ((float *) mem_145220.mem)[i_144270 * m_119002 + i_144263];
            
            // futhark/microgpt.fut:434:18-57
            
            float lifted_lambda_res_141039 = zs_lhs_141038 / zs_rhs_123662;
            
            // futhark/microgpt.fut:4:11-25
            
            float zs_lhs_141046 = ((float *) mem_145217.mem)[i_144270 * m_119002 + i_144263];
            
            // futhark/microgpt.fut:436:18-58
            
            float lifted_lambda_res_141047 = zs_lhs_141046 / zs_rhs_123701;
            
            ((float *) mem_145255)[i_144270 * m_119002 + i_144263] = lifted_lambda_res_141047;
            ((float *) mem_145258)[i_144270 * m_119002 + i_144263] = lifted_lambda_res_141039;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145293, bytes_145216, "mem_145293")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144279 = 0; i_144279 < n_119001; i_144279++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144275 = 0; i_144275 < m_119002; i_144275++) {
            // futhark/microgpt.fut:4:11-25
            
            float zm_lhs_123397 = ((float *) w_mem_145211.mem)[i_144279 * m_119002 + i_144275];
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_123398 = ((float *) mem_145258)[i_144279 * m_119002 + i_144275];
            
            // futhark/microgpt.fut:438:21-34
            
            float zs_lhs_123399 = lt_r_119008 * zt_rhs_123398;
            
            // futhark/microgpt.fut:4:11-25
            
            float ztzt_lhs_123400 = ((float *) mem_145255)[i_144279 * m_119002 + i_144275];
            
            // futhark/microgpt.fut:438:51-57
            
            float zp_lhs_123401 = fpow32(ztzt_lhs_123400, 0.5F);
            
            // futhark/microgpt.fut:438:59-71
            
            float zs_rhs_123402 = 1.0e-8F + zp_lhs_123401;
            
            // futhark/microgpt.fut:438:35-71
            
            float zm_rhs_123403 = zs_lhs_123399 / zs_rhs_123402;
            
            // futhark/microgpt.fut:438:13-71
            
            float lifted_lambda_res_123404 = zm_lhs_123397 - zm_rhs_123403;
            
            ((float *) mem_145293.mem)[i_144279 * m_119002 + i_144275] = lifted_lambda_res_123404;
        }
    }
    if (memblock_set(ctx, &mem_out_147157, &mem_145293, "mem_145293") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147158, &mem_145220, "mem_145220") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147159, &mem_145217, "mem_145217") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147484, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147485, &mem_out_147158, "mem_out_147158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147486, &mem_out_147159, "mem_out_147159") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_145232);
        free(mem_145255);
        free(mem_145258);
        if (memblock_unref(ctx, &mem_145293, "mem_145293") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145220, "mem_145220") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145217, "mem_145217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147159, "mem_out_147159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147158, "mem_out_147158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_adam_opt_w_12791(struct futhark_context *ctx, struct memblock *mem_out_p_147490, struct memblock *mem_out_p_147491, struct memblock *mem_out_p_147492, struct memblock w_mem_145211, struct memblock mw_mem_145212, struct memblock vw_mem_145213, struct memblock dw_mem_145214, int64_t n_120034, int64_t m_120035, int64_t step_120040, float lt_r_120041)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145232_cached_sizze_147493 = 0;
    unsigned char *mem_145232 = NULL;
    int64_t mem_145255_cached_sizze_147494 = 0;
    unsigned char *mem_145255 = NULL;
    int64_t mem_145258_cached_sizze_147495 = 0;
    unsigned char *mem_145258 = NULL;
    struct memblock mem_145293;
    
    mem_145293.references = NULL;
    
    struct memblock mem_145220;
    
    mem_145220.references = NULL;
    
    struct memblock mem_145217;
    
    mem_145217.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t binop_x_145215 = (int64_t) 4 * n_120034;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_145216 = m_120035 * binop_x_145215;
    
    // futhark/microgpt.fut:4:11-25
    
    int64_t bytes_145229 = (int64_t) 4 * m_120035;
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145217, bytes_145216, "mem_145217")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145220, bytes_145216, "mem_145220")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145232_cached_sizze_147493 < bytes_145229) {
        err = lexical_realloc(ctx, &mem_145232, &mem_145232_cached_sizze_147493, bytes_145229);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144256 = 0; i_144256 < n_120034; i_144256++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144249 = 0; i_144249 < m_120035; i_144249++) {
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141002 = ((float *) mw_mem_145212.mem)[i_144256 * m_120035 + i_144249];
            
            // futhark/microgpt.fut:430:10-20
            
            float zp_lhs_141003 = 0.85F * zt_rhs_141002;
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141004 = ((float *) dw_mem_145214.mem)[i_144256 * m_120035 + i_144249];
            
            // futhark/microgpt.fut:430:35-45
            
            float zp_rhs_141005 = 0.14999998F * zt_rhs_141004;
            
            // futhark/microgpt.fut:430:21-45
            
            float lifted_lambda_res_141006 = zp_lhs_141003 + zp_rhs_141005;
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_141013 = ((float *) vw_mem_145213.mem)[i_144256 * m_120035 + i_144249];
            
            // futhark/microgpt.fut:432:10-20
            
            float zp_lhs_141014 = 0.99F * zt_rhs_141013;
            
            // futhark/microgpt.fut:432:35-45
            
            float zt_lhs_141016 = 9.99999e-3F * zt_rhs_141004;
            
            // futhark/microgpt.fut:432:46-56
            
            float zp_rhs_141017 = zt_rhs_141004 * zt_lhs_141016;
            
            // futhark/microgpt.fut:432:21-56
            
            float lifted_lambda_res_141018 = zp_lhs_141014 + zp_rhs_141017;
            
            ((float *) mem_145217.mem)[i_144256 * m_120035 + i_144249] = lifted_lambda_res_141018;
            ((float *) mem_145232)[i_144249] = lifted_lambda_res_141006;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145220.mem, i_144256 * m_120035, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145232, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_120035});
    }
    // futhark/microgpt.fut:66:26-45
    
    float i64_res_123659 = sitofp_i64_f32(step_120040);
    
    // futhark/microgpt.fut:434:54-57
    
    float ztzt_rhs_123660 = 1.0F + i64_res_123659;
    
    // futhark/microgpt.fut:434:30-57
    
    float zm_rhs_123661 = fpow32(0.85F, ztzt_rhs_123660);
    
    // futhark/microgpt.fut:434:23-57
    
    float zs_rhs_123662 = 1.0F - zm_rhs_123661;
    
    // futhark/microgpt.fut:436:31-58
    
    float zm_rhs_123700 = fpow32(0.99F, ztzt_rhs_123660);
    
    // futhark/microgpt.fut:436:23-58
    
    float zs_rhs_123701 = 1.0F - zm_rhs_123700;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_145255_cached_sizze_147494 < bytes_145216) {
        err = lexical_realloc(ctx, &mem_145255, &mem_145255_cached_sizze_147494, bytes_145216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145258_cached_sizze_147495 < bytes_145216) {
        err = lexical_realloc(ctx, &mem_145258, &mem_145258_cached_sizze_147495, bytes_145216);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144270 = 0; i_144270 < n_120034; i_144270++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144263 = 0; i_144263 < m_120035; i_144263++) {
            // futhark/microgpt.fut:4:11-25
            
            float zs_lhs_141038 = ((float *) mem_145220.mem)[i_144270 * m_120035 + i_144263];
            
            // futhark/microgpt.fut:434:18-57
            
            float lifted_lambda_res_141039 = zs_lhs_141038 / zs_rhs_123662;
            
            // futhark/microgpt.fut:4:11-25
            
            float zs_lhs_141046 = ((float *) mem_145217.mem)[i_144270 * m_120035 + i_144263];
            
            // futhark/microgpt.fut:436:18-58
            
            float lifted_lambda_res_141047 = zs_lhs_141046 / zs_rhs_123701;
            
            ((float *) mem_145255)[i_144270 * m_120035 + i_144263] = lifted_lambda_res_141047;
            ((float *) mem_145258)[i_144270 * m_120035 + i_144263] = lifted_lambda_res_141039;
        }
    }
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145293, bytes_145216, "mem_145293")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144279 = 0; i_144279 < n_120034; i_144279++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144275 = 0; i_144275 < m_120035; i_144275++) {
            // futhark/microgpt.fut:4:11-25
            
            float zm_lhs_123397 = ((float *) w_mem_145211.mem)[i_144279 * m_120035 + i_144275];
            
            // futhark/microgpt.fut:4:11-25
            
            float zt_rhs_123398 = ((float *) mem_145258)[i_144279 * m_120035 + i_144275];
            
            // futhark/microgpt.fut:438:21-34
            
            float zs_lhs_123399 = lt_r_120041 * zt_rhs_123398;
            
            // futhark/microgpt.fut:4:11-25
            
            float ztzt_lhs_123400 = ((float *) mem_145255)[i_144279 * m_120035 + i_144275];
            
            // futhark/microgpt.fut:438:51-57
            
            float zp_lhs_123401 = fpow32(ztzt_lhs_123400, 0.5F);
            
            // futhark/microgpt.fut:438:59-71
            
            float zs_rhs_123402 = 1.0e-8F + zp_lhs_123401;
            
            // futhark/microgpt.fut:438:35-71
            
            float zm_rhs_123403 = zs_lhs_123399 / zs_rhs_123402;
            
            // futhark/microgpt.fut:438:13-71
            
            float lifted_lambda_res_123404 = zm_lhs_123397 - zm_rhs_123403;
            
            ((float *) mem_145293.mem)[i_144279 * m_120035 + i_144275] = lifted_lambda_res_123404;
        }
    }
    if (memblock_set(ctx, &mem_out_147157, &mem_145293, "mem_145293") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147158, &mem_145220, "mem_145220") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147159, &mem_145217, "mem_145217") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147490, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147491, &mem_out_147158, "mem_out_147158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147492, &mem_out_147159, "mem_out_147159") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_145232);
        free(mem_145255);
        free(mem_145258);
        if (memblock_unref(ctx, &mem_145293, "mem_145293") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145220, "mem_145220") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145217, "mem_145217") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147159, "mem_out_147159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147158, "mem_out_147158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_cal_target_9281(struct futhark_context *ctx, struct memblock *mem_out_p_147496, struct memblock seq_mem_145211, int64_t n_72699)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145217_cached_sizze_147497 = 0;
    unsigned char *mem_145217 = NULL;
    struct memblock mem_145212;
    
    mem_145212.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    // futhark/microgpt.fut:417:37-40
    
    int64_t zl_rhs_123288 = sub64(n_72699, (int64_t) 1);
    
    // futhark/microgpt.fut:4:11-25
    if (memblock_alloc(ctx, &mem_145212, (int64_t) 1728, "mem_145212")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145217_cached_sizze_147497 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145217, &mem_145217_cached_sizze_147497, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144251 = 0; i_144251 < (int64_t) 16; i_144251++) {
        // futhark/microgpt.fut:417:25-78
        
        bool cond_123291 = slt64(i_144251, zl_rhs_123288);
        
        // futhark/microgpt.fut:417:53-56
        
        int64_t zeze_lhs_123292 = add64((int64_t) 1, i_144251);
        
        // futhark/microgpt.fut:417:47-57
        
        bool x_123293 = sle64((int64_t) 0, zeze_lhs_123292);
        
        // futhark/microgpt.fut:417:47-57
        
        bool y_123294 = slt64(zeze_lhs_123292, (int64_t) 16);
        
        // futhark/microgpt.fut:417:47-57
        
        bool bounds_check_123295 = x_123293 && y_123294;
        
        // futhark/microgpt.fut:9:27-39
        
        bool loop_not_taken_123296 = !cond_123291;
        
        // futhark/microgpt.fut:9:27-39
        
        bool protect_assert_disj_123297 = bounds_check_123295 || loop_not_taken_123296;
        
        // futhark/microgpt.fut:417:47-57
        
        bool index_certs_123298;
        
        if (!protect_assert_disj_123297) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) zeze_lhs_123292, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:417:47-57\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:417:3-80\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:417:47-57
        
        int64_t zeze_lhs_123299;
        
        if (cond_123291) {
            // futhark/microgpt.fut:9:27-39
            
            int64_t x_140898 = ((int64_t *) seq_mem_145211.mem)[zeze_lhs_123292];
            
            zeze_lhs_123299 = x_140898;
        } else {
            zeze_lhs_123299 = (int64_t) 0;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144247 = 0; i_144247 < (int64_t) 27; i_144247++) {
            // futhark/microgpt.fut:417:58-62
            
            bool cond_t_res_123303 = zeze_lhs_123299 == i_144247;
            
            // futhark/microgpt.fut:9:27-39
            
            bool x_123304 = cond_123291 && cond_t_res_123303;
            
            // futhark/microgpt.fut:417:25-78
            
            float lifted_lambda_res_123305;
            
            if (x_123304) {
                lifted_lambda_res_123305 = 1.0F;
            } else {
                lifted_lambda_res_123305 = 0.0F;
            }
            ((float *) mem_145217)[i_144247] = lifted_lambda_res_123305;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145212.mem, i_144251 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145217, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_147157, &mem_145212, "mem_145212") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147496, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_145217);
        if (memblock_unref(ctx, &mem_145212, "mem_145212") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_forward(struct futhark_context *ctx, struct memblock *mem_out_p_147498, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock seqs_mem_145220, struct memblock masks_mem_145221)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145222_cached_sizze_147499 = 0;
    unsigned char *mem_145222 = NULL;
    int64_t mem_145227_cached_sizze_147500 = 0;
    unsigned char *mem_145227 = NULL;
    int64_t mem_145238_cached_sizze_147501 = 0;
    unsigned char *mem_145238 = NULL;
    int64_t mem_145243_cached_sizze_147502 = 0;
    unsigned char *mem_145243 = NULL;
    int64_t mem_145250_cached_sizze_147503 = 0;
    unsigned char *mem_145250 = NULL;
    int64_t mem_145261_cached_sizze_147504 = 0;
    unsigned char *mem_145261 = NULL;
    int64_t mem_145262_cached_sizze_147505 = 0;
    unsigned char *mem_145262 = NULL;
    int64_t mem_145263_cached_sizze_147506 = 0;
    unsigned char *mem_145263 = NULL;
    int64_t mem_145276_cached_sizze_147507 = 0;
    unsigned char *mem_145276 = NULL;
    int64_t mem_145277_cached_sizze_147508 = 0;
    unsigned char *mem_145277 = NULL;
    int64_t mem_145278_cached_sizze_147509 = 0;
    unsigned char *mem_145278 = NULL;
    int64_t mem_145288_cached_sizze_147510 = 0;
    unsigned char *mem_145288 = NULL;
    int64_t mem_145295_cached_sizze_147511 = 0;
    unsigned char *mem_145295 = NULL;
    int64_t mem_145302_cached_sizze_147512 = 0;
    unsigned char *mem_145302 = NULL;
    int64_t mem_145330_cached_sizze_147513 = 0;
    unsigned char *mem_145330 = NULL;
    int64_t mem_145331_cached_sizze_147514 = 0;
    unsigned char *mem_145331 = NULL;
    int64_t mem_145332_cached_sizze_147515 = 0;
    unsigned char *mem_145332 = NULL;
    int64_t mem_145348_cached_sizze_147516 = 0;
    unsigned char *mem_145348 = NULL;
    int64_t mem_145349_cached_sizze_147517 = 0;
    unsigned char *mem_145349 = NULL;
    int64_t mem_145350_cached_sizze_147518 = 0;
    unsigned char *mem_145350 = NULL;
    int64_t mem_145363_cached_sizze_147519 = 0;
    unsigned char *mem_145363 = NULL;
    int64_t mem_145364_cached_sizze_147520 = 0;
    unsigned char *mem_145364 = NULL;
    int64_t mem_145365_cached_sizze_147521 = 0;
    unsigned char *mem_145365 = NULL;
    int64_t mem_145411_cached_sizze_147522 = 0;
    unsigned char *mem_145411 = NULL;
    int64_t mem_145412_cached_sizze_147523 = 0;
    unsigned char *mem_145412 = NULL;
    int64_t mem_145413_cached_sizze_147524 = 0;
    unsigned char *mem_145413 = NULL;
    int64_t mem_145429_cached_sizze_147525 = 0;
    unsigned char *mem_145429 = NULL;
    int64_t mem_145430_cached_sizze_147526 = 0;
    unsigned char *mem_145430 = NULL;
    int64_t mem_145431_cached_sizze_147527 = 0;
    unsigned char *mem_145431 = NULL;
    int64_t mem_145444_cached_sizze_147528 = 0;
    unsigned char *mem_145444 = NULL;
    int64_t mem_145445_cached_sizze_147529 = 0;
    unsigned char *mem_145445 = NULL;
    int64_t mem_145446_cached_sizze_147530 = 0;
    unsigned char *mem_145446 = NULL;
    int64_t mem_145492_cached_sizze_147531 = 0;
    unsigned char *mem_145492 = NULL;
    int64_t mem_145498_cached_sizze_147532 = 0;
    unsigned char *mem_145498 = NULL;
    int64_t mem_145503_cached_sizze_147533 = 0;
    unsigned char *mem_145503 = NULL;
    int64_t mem_145519_cached_sizze_147534 = 0;
    unsigned char *mem_145519 = NULL;
    int64_t mem_145525_cached_sizze_147535 = 0;
    unsigned char *mem_145525 = NULL;
    int64_t mem_145530_cached_sizze_147536 = 0;
    unsigned char *mem_145530 = NULL;
    int64_t mem_145546_cached_sizze_147537 = 0;
    unsigned char *mem_145546 = NULL;
    int64_t mem_145552_cached_sizze_147538 = 0;
    unsigned char *mem_145552 = NULL;
    int64_t mem_145557_cached_sizze_147539 = 0;
    unsigned char *mem_145557 = NULL;
    int64_t mem_145564_cached_sizze_147540 = 0;
    unsigned char *mem_145564 = NULL;
    int64_t mem_145571_cached_sizze_147541 = 0;
    unsigned char *mem_145571 = NULL;
    int64_t mem_145587_cached_sizze_147542 = 0;
    unsigned char *mem_145587 = NULL;
    int64_t mem_145593_cached_sizze_147543 = 0;
    unsigned char *mem_145593 = NULL;
    int64_t mem_145598_cached_sizze_147544 = 0;
    unsigned char *mem_145598 = NULL;
    int64_t mem_145614_cached_sizze_147545 = 0;
    unsigned char *mem_145614 = NULL;
    int64_t mem_145620_cached_sizze_147546 = 0;
    unsigned char *mem_145620 = NULL;
    int64_t mem_145625_cached_sizze_147547 = 0;
    unsigned char *mem_145625 = NULL;
    int64_t mem_145641_cached_sizze_147548 = 0;
    unsigned char *mem_145641 = NULL;
    int64_t mem_145646_cached_sizze_147549 = 0;
    unsigned char *mem_145646 = NULL;
    int64_t mem_145657_cached_sizze_147550 = 0;
    unsigned char *mem_145657 = NULL;
    int64_t mem_145662_cached_sizze_147551 = 0;
    unsigned char *mem_145662 = NULL;
    int64_t mem_145673_cached_sizze_147552 = 0;
    unsigned char *mem_145673 = NULL;
    int64_t mem_145678_cached_sizze_147553 = 0;
    unsigned char *mem_145678 = NULL;
    int64_t mem_145689_cached_sizze_147554 = 0;
    unsigned char *mem_145689 = NULL;
    int64_t mem_145694_cached_sizze_147555 = 0;
    unsigned char *mem_145694 = NULL;
    int64_t mem_145698_cached_sizze_147556 = 0;
    unsigned char *mem_145698 = NULL;
    int64_t mem_145712_cached_sizze_147557 = 0;
    unsigned char *mem_145712 = NULL;
    int64_t mem_145717_cached_sizze_147558 = 0;
    unsigned char *mem_145717 = NULL;
    int64_t mem_145728_cached_sizze_147559 = 0;
    unsigned char *mem_145728 = NULL;
    int64_t mem_145733_cached_sizze_147560 = 0;
    unsigned char *mem_145733 = NULL;
    struct memblock mem_145744;
    
    mem_145744.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    // futhark/microgpt.fut:4:11-25
    if (mem_145222_cached_sizze_147499 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145222, &mem_145222_cached_sizze_147499, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145227_cached_sizze_147500 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145227, &mem_145227_cached_sizze_147500, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144251 = 0; i_144251 < (int64_t) 16; i_144251++) {
        // futhark/microgpt.fut:4:11-25
        
        int64_t tmp_140177 = ((int64_t *) seqs_mem_145220.mem)[i_144251];
        
        // futhark/microgpt.fut:413:39-54
        
        bool x_140178 = sle64((int64_t) 0, tmp_140177);
        
        // futhark/microgpt.fut:413:39-54
        
        bool y_140179 = slt64(tmp_140177, (int64_t) 27);
        
        // futhark/microgpt.fut:413:39-54
        
        bool bounds_check_140180 = x_140178 && y_140179;
        
        // futhark/microgpt.fut:413:39-54
        
        bool index_certs_140181;
        
        if (!bounds_check_140180) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140177, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:413:39-54\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:413:14-58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144247 = 0; i_144247 < (int64_t) 16; i_144247++) {
            // futhark/microgpt.fut:4:11-25
            
            float lifted_lambda_res_140188 = ((float *) wte_mem_145216.mem)[tmp_140177 * (int64_t) 16 + i_144247];
            
            ((float *) mem_145227)[i_144247] = lifted_lambda_res_140188;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145222, i_144251 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145227, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145238_cached_sizze_147501 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145238, &mem_145238_cached_sizze_147501, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145243_cached_sizze_147502 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145243, &mem_145243_cached_sizze_147502, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145250_cached_sizze_147503 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145250, &mem_145250_cached_sizze_147503, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144263 = 0; i_144263 < (int64_t) 16; i_144263++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144255 = 0; i_144255 < (int64_t) 16; i_144255++) {
            float zp_lhs_140212 = ((float *) mem_145222)[i_144263 * (int64_t) 16 + i_144255];
            
            // futhark/microgpt.fut:4:11-25
            
            float zp_rhs_140213 = ((float *) wpe_mem_145214.mem)[i_144263 * (int64_t) 16 + i_144255];
            
            // futhark/microgpt.fut:148:68-103
            
            float zp_res_140214 = zp_lhs_140212 + zp_rhs_140213;
            
            // futhark/microgpt.fut:148:86-145
            
            float zt_res_140215 = zp_res_140214 * zp_res_140214;
            
            ((float *) mem_145243)[i_144255] = zt_res_140215;
        }
        // futhark/microgpt.fut:71:13-49
        
        float defunc_0_lifted_lambda_res_140217;
        float r_140219 = 0.0F;
        
        for (int64_t i_140218 = 0; i_140218 < (int64_t) 16; i_140218++) {
            // futhark/microgpt.fut:149:35-43
            
            float lifted_lambda_res_140220 = ((float *) mem_145243)[i_140218];
            
            // futhark/microgpt.fut:71:40-49
            
            float zp_res_140221 = r_140219 + lifted_lambda_res_140220;
            float r_tmp_147162 = zp_res_140221;
            
            r_140219 = r_tmp_147162;
        }
        defunc_0_lifted_lambda_res_140217 = r_140219;
        // futhark/microgpt.fut:149:17-60
        
        float zs_res_140222 = defunc_0_lifted_lambda_res_140217 / 16.0F;
        
        // futhark/microgpt.fut:150:24-55
        
        float zp_res_140223 = 1.0e-5F + zs_res_140222;
        
        // futhark/microgpt.fut:150:16-55
        
        float sqrt_res_140224 = futrts_sqrt32(zp_res_140223);
        
        // futhark/microgpt.fut:151:67-78
        
        float zs_res_140225 = 1.0F / sqrt_res_140224;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144259 = 0; i_144259 < (int64_t) 16; i_144259++) {
            float zp_lhs_140232 = ((float *) mem_145222)[i_144263 * (int64_t) 16 + i_144259];
            
            // futhark/microgpt.fut:4:11-25
            
            float zp_rhs_140233 = ((float *) wpe_mem_145214.mem)[i_144263 * (int64_t) 16 + i_144259];
            
            // futhark/microgpt.fut:151:25-60
            
            float zp_res_140234 = zp_lhs_140232 + zp_rhs_140233;
            
            // futhark/microgpt.fut:151:43-78
            
            float zt_res_140235 = zs_res_140225 * zp_res_140234;
            
            ((float *) mem_145250)[i_144259] = zt_res_140235;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145238, i_144263 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145250, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145261_cached_sizze_147504 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145261, &mem_145261_cached_sizze_147504, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145262_cached_sizze_147505 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145262, &mem_145262_cached_sizze_147505, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145263_cached_sizze_147506 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145263, &mem_145263_cached_sizze_147506, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145276_cached_sizze_147507 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145276, &mem_145276_cached_sizze_147507, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145277_cached_sizze_147508 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145277, &mem_145277_cached_sizze_147508, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145278_cached_sizze_147509 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145278, &mem_145278_cached_sizze_147509, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145288_cached_sizze_147510 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145288, &mem_145288_cached_sizze_147510, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145295_cached_sizze_147511 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145295, &mem_145295_cached_sizze_147511, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145302_cached_sizze_147512 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145302, &mem_145302_cached_sizze_147512, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144293 = 0; i_144293 < (int64_t) 16; i_144293++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144283 = 0; i_144283 < (int64_t) 16; i_144283++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141488;
            float r_141490 = 0.0F;
            
            for (int64_t i_141489 = 0; i_141489 < (int64_t) 16; i_141489++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141491 = ((float *) wqry_mem_145215.mem)[i_144283 * (int64_t) 16 + i_141489];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144267 = 0; i_144267 < (int64_t) 16; i_144267++) {
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_141498 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_144267];
                    
                    // futhark/microgpt.fut:152:128-167
                    
                    float zt_res_141499 = zt_lhs_141498 * zt_lhs_141498;
                    
                    ((float *) mem_145288)[i_144267] = zt_res_141499;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141501;
                float r_141503 = 0.0F;
                
                for (int64_t i_141502 = 0; i_141502 < (int64_t) 16; i_141502++) {
                    // futhark/microgpt.fut:153:35-43
                    
                    float lifted_lambda_res_141504 = ((float *) mem_145288)[i_141502];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141505 = r_141503 + lifted_lambda_res_141504;
                    float r_tmp_147172 = zp_res_141505;
                    
                    r_141503 = r_tmp_147172;
                }
                defunc_0_lifted_lambda_res_141501 = r_141503;
                // futhark/microgpt.fut:153:17-60
                
                float zs_res_141506 = defunc_0_lifted_lambda_res_141501 / 16.0F;
                
                // futhark/microgpt.fut:154:24-55
                
                float zp_res_141507 = 1.0e-5F + zs_res_141506;
                
                // futhark/microgpt.fut:154:16-55
                
                float sqrt_res_141508 = futrts_sqrt32(zp_res_141507);
                
                // futhark/microgpt.fut:142:5-184:137
                
                float zt_lhs_141509 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_141489];
                
                // futhark/microgpt.fut:155:28-39
                
                float zs_res_141510 = 1.0F / sqrt_res_141508;
                
                // futhark/microgpt.fut:155:5-39
                
                float zt_res_141511 = zt_lhs_141509 * zs_res_141510;
                
                // futhark/microgpt.fut:152:78-155:39
                
                float zt_res_141512 = zt_lhs_141491 * zt_res_141511;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141513 = r_141490 + zt_res_141512;
                float r_tmp_147170 = zp_res_141513;
                
                r_141490 = r_tmp_147170;
            }
            defunc_0_lifted_lambda_res_141488 = r_141490;
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141520;
            float r_141522 = 0.0F;
            
            for (int64_t i_141521 = 0; i_141521 < (int64_t) 16; i_141521++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141523 = ((float *) wkey_mem_145212.mem)[i_144283 * (int64_t) 16 + i_141521];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144271 = 0; i_144271 < (int64_t) 16; i_144271++) {
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_141530 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_144271];
                    
                    // futhark/microgpt.fut:156:128-167
                    
                    float zt_res_141531 = zt_lhs_141530 * zt_lhs_141530;
                    
                    ((float *) mem_145295)[i_144271] = zt_res_141531;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141533;
                float r_141535 = 0.0F;
                
                for (int64_t i_141534 = 0; i_141534 < (int64_t) 16; i_141534++) {
                    // futhark/microgpt.fut:157:35-43
                    
                    float lifted_lambda_res_141536 = ((float *) mem_145295)[i_141534];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141537 = r_141535 + lifted_lambda_res_141536;
                    float r_tmp_147175 = zp_res_141537;
                    
                    r_141535 = r_tmp_147175;
                }
                defunc_0_lifted_lambda_res_141533 = r_141535;
                // futhark/microgpt.fut:157:17-60
                
                float zs_res_141538 = defunc_0_lifted_lambda_res_141533 / 16.0F;
                
                // futhark/microgpt.fut:158:24-55
                
                float zp_res_141539 = 1.0e-5F + zs_res_141538;
                
                // futhark/microgpt.fut:158:16-55
                
                float sqrt_res_141540 = futrts_sqrt32(zp_res_141539);
                
                // futhark/microgpt.fut:142:5-184:137
                
                float zt_lhs_141541 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_141521];
                
                // futhark/microgpt.fut:159:28-39
                
                float zs_res_141542 = 1.0F / sqrt_res_141540;
                
                // futhark/microgpt.fut:159:5-39
                
                float zt_res_141543 = zt_lhs_141541 * zs_res_141542;
                
                // futhark/microgpt.fut:156:78-159:39
                
                float zt_res_141544 = zt_lhs_141523 * zt_res_141543;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141545 = r_141522 + zt_res_141544;
                float r_tmp_147173 = zp_res_141545;
                
                r_141522 = r_tmp_147173;
            }
            defunc_0_lifted_lambda_res_141520 = r_141522;
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141555;
            float r_141557 = 0.0F;
            
            for (int64_t i_141556 = 0; i_141556 < (int64_t) 16; i_141556++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141558 = ((float *) wval_mem_145218.mem)[i_144283 * (int64_t) 16 + i_141556];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144275 = 0; i_144275 < (int64_t) 16; i_144275++) {
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_141565 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_144275];
                    
                    // futhark/microgpt.fut:160:128-167
                    
                    float zt_res_141566 = zt_lhs_141565 * zt_lhs_141565;
                    
                    ((float *) mem_145302)[i_144275] = zt_res_141566;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141568;
                float r_141570 = 0.0F;
                
                for (int64_t i_141569 = 0; i_141569 < (int64_t) 16; i_141569++) {
                    // futhark/microgpt.fut:161:35-43
                    
                    float lifted_lambda_res_141571 = ((float *) mem_145302)[i_141569];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141572 = r_141570 + lifted_lambda_res_141571;
                    float r_tmp_147178 = zp_res_141572;
                    
                    r_141570 = r_tmp_147178;
                }
                defunc_0_lifted_lambda_res_141568 = r_141570;
                // futhark/microgpt.fut:161:17-60
                
                float zs_res_141573 = defunc_0_lifted_lambda_res_141568 / 16.0F;
                
                // futhark/microgpt.fut:162:24-55
                
                float zp_res_141574 = 1.0e-5F + zs_res_141573;
                
                // futhark/microgpt.fut:162:16-55
                
                float sqrt_res_141575 = futrts_sqrt32(zp_res_141574);
                
                // futhark/microgpt.fut:142:5-184:137
                
                float zt_lhs_141576 = ((float *) mem_145238)[i_144293 * (int64_t) 16 + i_141556];
                
                // futhark/microgpt.fut:163:28-39
                
                float zs_res_141577 = 1.0F / sqrt_res_141575;
                
                // futhark/microgpt.fut:163:5-39
                
                float zt_res_141578 = zt_lhs_141576 * zs_res_141577;
                
                // futhark/microgpt.fut:160:78-163:39
                
                float zt_res_141579 = zt_lhs_141558 * zt_res_141578;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141580 = r_141557 + zt_res_141579;
                float r_tmp_147176 = zp_res_141580;
                
                r_141557 = r_tmp_147176;
            }
            defunc_0_lifted_lambda_res_141555 = r_141557;
            ((float *) mem_145276)[i_144283] = defunc_0_lifted_lambda_res_141555;
            ((float *) mem_145277)[i_144283] = defunc_0_lifted_lambda_res_141520;
            ((float *) mem_145278)[i_144283] = defunc_0_lifted_lambda_res_141488;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145261, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145276, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145262, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145263, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145278, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145330_cached_sizze_147513 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145330, &mem_145330_cached_sizze_147513, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145331_cached_sizze_147514 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145331, &mem_145331_cached_sizze_147514, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145332_cached_sizze_147515 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145332, &mem_145332_cached_sizze_147515, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145348_cached_sizze_147516 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145348, &mem_145348_cached_sizze_147516, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145349_cached_sizze_147517 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145349, &mem_145349_cached_sizze_147517, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145350_cached_sizze_147518 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145350, &mem_145350_cached_sizze_147518, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145363_cached_sizze_147519 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145363, &mem_145363_cached_sizze_147519, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145364_cached_sizze_147520 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145364, &mem_145364_cached_sizze_147520, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145365_cached_sizze_147521 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145365, &mem_145365_cached_sizze_147521, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144323 = 0; i_144323 < (int64_t) 16; i_144323++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144313 = 0; i_144313 < (int64_t) 4; i_144313++) {
            // futhark/microgpt.fut:164:94-97
            
            int64_t zp_lhs_141645 = mul64((int64_t) 4, i_144313);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144303 = 0; i_144303 < (int64_t) 4; i_144303++) {
                // futhark/microgpt.fut:164:99-104
                
                int64_t tmp_141729 = add64(zp_lhs_141645, i_144303);
                
                // futhark/microgpt.fut:164:75-106
                
                bool x_141730 = sle64((int64_t) 0, tmp_141729);
                
                // futhark/microgpt.fut:164:75-106
                
                bool y_141731 = slt64(tmp_141729, (int64_t) 16);
                
                // futhark/microgpt.fut:164:75-106
                
                bool bounds_check_141732 = x_141730 && y_141731;
                
                // futhark/microgpt.fut:164:75-106
                
                bool index_certs_141733;
                
                if (!bounds_check_141732) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141729, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:164:75-106\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:164:58-107\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:164:40-109\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:9:27-39\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:9:13-40\n   #11 futhark/microgpt.fut:164:15-111\n   #12 futhark/microgpt.fut:414:7-67\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141734 = ((float *) mem_145263)[i_144323 * (int64_t) 16 + tmp_141729];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141742 = ((float *) mem_145262)[i_144323 * (int64_t) 16 + tmp_141729];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141753 = ((float *) mem_145261)[i_144323 * (int64_t) 16 + tmp_141729];
                
                ((float *) mem_145363)[i_144303] = lifted_lambda_res_141753;
                ((float *) mem_145364)[i_144303] = lifted_lambda_res_141742;
                ((float *) mem_145365)[i_144303] = lifted_lambda_res_141734;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145348, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145363, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145349, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145364, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145350, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145365, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145330, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145348, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145331, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145349, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145332, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145350, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145411_cached_sizze_147522 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145411, &mem_145411_cached_sizze_147522, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145412_cached_sizze_147523 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145412, &mem_145412_cached_sizze_147523, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145413_cached_sizze_147524 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145413, &mem_145413_cached_sizze_147524, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145429_cached_sizze_147525 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145429, &mem_145429_cached_sizze_147525, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145430_cached_sizze_147526 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145430, &mem_145430_cached_sizze_147526, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145431_cached_sizze_147527 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145431, &mem_145431_cached_sizze_147527, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145444_cached_sizze_147528 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145444, &mem_145444_cached_sizze_147528, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145445_cached_sizze_147529 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145445, &mem_145445_cached_sizze_147529, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145446_cached_sizze_147530 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145446, &mem_145446_cached_sizze_147530, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144353 = 0; i_144353 < (int64_t) 4; i_144353++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144343 = 0; i_144343 < (int64_t) 16; i_144343++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144333 = 0; i_144333 < (int64_t) 4; i_144333++) {
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141914 = ((float *) mem_145332)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141921 = ((float *) mem_145331)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141931 = ((float *) mem_145330)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                ((float *) mem_145444)[i_144333] = lifted_lambda_res_141931;
                ((float *) mem_145445)[i_144333] = lifted_lambda_res_141921;
                ((float *) mem_145446)[i_144333] = lifted_lambda_res_141914;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145429, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145444, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145430, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145445, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145431, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145446, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145411, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145429, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145412, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145430, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145413, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145431, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145492_cached_sizze_147531 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145492, &mem_145492_cached_sizze_147531, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145498_cached_sizze_147532 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145498, &mem_145498_cached_sizze_147532, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145503_cached_sizze_147533 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145503, &mem_145503_cached_sizze_147533, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144367 = 0; i_144367 < (int64_t) 4; i_144367++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144363 = 0; i_144363 < (int64_t) 16; i_144363++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144359 = 0; i_144359 < (int64_t) 16; i_144359++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140503;
                float r_140505 = 0.0F;
                
                for (int64_t i_140504 = 0; i_140504 < (int64_t) 4; i_140504++) {
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_140506 = ((float *) mem_145413)[i_144367 * (int64_t) 64 + i_144363 * (int64_t) 4 + i_140504];
                    
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_rhs_140507 = ((float *) mem_145412)[i_144367 * (int64_t) 64 + i_144359 * (int64_t) 4 + i_140504];
                    
                    // futhark/microgpt.fut:170:96-145
                    
                    float zt_res_140508 = zt_lhs_140506 * zt_rhs_140507;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140509 = r_140505 + zt_res_140508;
                    float r_tmp_147200 = zp_res_140509;
                    
                    r_140505 = r_tmp_147200;
                }
                defunc_0_lifted_lambda_res_140503 = r_140505;
                ((float *) mem_145503)[i_144359] = defunc_0_lifted_lambda_res_140503;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145498, i_144363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145503, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145492, i_144367 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145498, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145519_cached_sizze_147534 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145519, &mem_145519_cached_sizze_147534, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145525_cached_sizze_147535 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145525, &mem_145525_cached_sizze_147535, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145530_cached_sizze_147536 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145530, &mem_145530_cached_sizze_147536, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144379 = 0; i_144379 < (int64_t) 4; i_144379++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144375 = 0; i_144375 < (int64_t) 16; i_144375++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144371 = 0; i_144371 < (int64_t) 16; i_144371++) {
                // futhark/microgpt.fut:142:5-184:137
                
                float zs_lhs_140531 = ((float *) mem_145492)[i_144379 * (int64_t) 256 + i_144375 * (int64_t) 16 + i_144371];
                
                // futhark/microgpt.fut:171:79-116
                
                float zs_res_140532 = zs_lhs_140531 / 2.0F;
                
                // futhark/microgpt.fut:4:11-25
                
                float zp_rhs_140533 = ((float *) masks_mem_145221.mem)[i_144375 * (int64_t) 16 + i_144371];
                
                // futhark/microgpt.fut:171:103-141
                
                float zp_res_140534 = zs_res_140532 + zp_rhs_140533;
                
                ((float *) mem_145530)[i_144371] = zp_res_140534;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145525, i_144375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145530, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145519, i_144379 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145525, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145546_cached_sizze_147537 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145546, &mem_145546_cached_sizze_147537, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145552_cached_sizze_147538 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145552, &mem_145552_cached_sizze_147538, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145557_cached_sizze_147539 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145557, &mem_145557_cached_sizze_147539, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145564_cached_sizze_147540 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145564, &mem_145564_cached_sizze_147540, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145571_cached_sizze_147541 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145571, &mem_145571_cached_sizze_147541, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144401 = 0; i_144401 < (int64_t) 4; i_144401++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144397 = 0; i_144397 < (int64_t) 16; i_144397++) {
            // futhark/microgpt.fut:103:13-33
            
            float defunc_0_reduce_res_142010;
            float redout_144381 = -INFINITY;
            
            for (int64_t i_144382 = 0; i_144382 < (int64_t) 16; i_144382++) {
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_141960 = ((float *) mem_145519)[i_144401 * (int64_t) 256 + i_144397 * (int64_t) 16 + i_144382];
                
                // futhark/microgpt.fut:103:13-33
                
                float max_res_140562 = fmax32(lifted_lambda_res_141960, redout_144381);
                float redout_tmp_147206 = max_res_140562;
                
                redout_144381 = redout_tmp_147206;
            }
            defunc_0_reduce_res_142010 = redout_144381;
            // futhark/microgpt.fut:113:47-56
            
            float neg_res_140563 = -defunc_0_reduce_res_142010;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144385 = 0; i_144385 < (int64_t) 16; i_144385++) {
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_140570 = ((float *) mem_145519)[i_144401 * (int64_t) 256 + i_144397 * (int64_t) 16 + i_144385];
                
                // futhark/microgpt.fut:113:38-56
                
                float zp_res_140571 = neg_res_140563 + lifted_lambda_res_140570;
                
                // futhark/microgpt.fut:113:31-56
                
                float exp_res_140572 = futrts_exp32(zp_res_140571);
                
                ((float *) mem_145557)[i_144385] = exp_res_140572;
            }
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140574;
            float r_140576 = 0.0F;
            
            for (int64_t i_140575 = 0; i_140575 < (int64_t) 16; i_140575++) {
                // futhark/microgpt.fut:114:32-39
                
                float lifted_lambda_res_140577 = ((float *) mem_145557)[i_140575];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140578 = r_140576 + lifted_lambda_res_140577;
                float r_tmp_147208 = zp_res_140578;
                
                r_140576 = r_tmp_147208;
            }
            defunc_0_lifted_lambda_res_140574 = r_140576;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144389 = 0; i_144389 < (int64_t) 16; i_144389++) {
                // futhark/microgpt.fut:115:23-30
                
                float zs_lhs_140585 = ((float *) mem_145557)[i_144389];
                
                // futhark/microgpt.fut:115:23-40
                
                float zs_res_140586 = zs_lhs_140585 / defunc_0_lifted_lambda_res_140574;
                
                ((float *) mem_145564)[i_144389] = zs_res_140586;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144393 = 0; i_144393 < (int64_t) 16; i_144393++) {
                // futhark/microgpt.fut:173:23-31
                
                float lifted_lambda_res_140594 = ((float *) mem_145564)[i_144393];
                
                ((float *) mem_145571)[i_144393] = lifted_lambda_res_140594;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145552, i_144397 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145571, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145546, i_144401 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145552, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145587_cached_sizze_147542 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145587, &mem_145587_cached_sizze_147542, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145593_cached_sizze_147543 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145593, &mem_145593_cached_sizze_147543, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145598_cached_sizze_147544 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145598, &mem_145598_cached_sizze_147544, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144413 = 0; i_144413 < (int64_t) 4; i_144413++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144409 = 0; i_144409 < (int64_t) 16; i_144409++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144405 = 0; i_144405 < (int64_t) 4; i_144405++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140616;
                float r_140618 = 0.0F;
                
                for (int64_t i_140617 = 0; i_140617 < (int64_t) 16; i_140617++) {
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_140619 = ((float *) mem_145546)[i_144413 * (int64_t) 256 + i_144409 * (int64_t) 16 + i_140617];
                    
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_rhs_140620 = ((float *) mem_145411)[i_144413 * (int64_t) 64 + i_140617 * (int64_t) 4 + i_144405];
                    
                    // futhark/microgpt.fut:174:97-149
                    
                    float zt_res_140621 = zt_lhs_140619 * zt_rhs_140620;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140622 = r_140618 + zt_res_140621;
                    float r_tmp_147214 = zp_res_140622;
                    
                    r_140618 = r_tmp_147214;
                }
                defunc_0_lifted_lambda_res_140616 = r_140618;
                ((float *) mem_145598)[i_144405] = defunc_0_lifted_lambda_res_140616;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145593, i_144409 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145598, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145587, i_144413 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145593, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145614_cached_sizze_147545 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145614, &mem_145614_cached_sizze_147545, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145620_cached_sizze_147546 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145620, &mem_145620_cached_sizze_147546, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145625_cached_sizze_147547 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145625, &mem_145625_cached_sizze_147547, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144425 = 0; i_144425 < (int64_t) 16; i_144425++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144421 = 0; i_144421 < (int64_t) 4; i_144421++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144417 = 0; i_144417 < (int64_t) 4; i_144417++) {
                // futhark/microgpt.fut:142:5-184:137
                
                float lifted_lambda_res_140644 = ((float *) mem_145587)[i_144421 * (int64_t) 64 + i_144425 * (int64_t) 4 + i_144417];
                
                ((float *) mem_145625)[i_144417] = lifted_lambda_res_140644;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145620, i_144421 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145625, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145614, i_144425 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145620, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145641_cached_sizze_147548 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145641, &mem_145641_cached_sizze_147548, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145646_cached_sizze_147549 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145646, &mem_145646_cached_sizze_147549, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144433 = 0; i_144433 < (int64_t) 16; i_144433++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144429 = 0; i_144429 < (int64_t) 16; i_144429++) {
            // futhark/microgpt.fut:176:84-87
            
            int64_t tmp_140656 = sdiv64(i_144429, (int64_t) 4);
            
            // futhark/microgpt.fut:176:62-89
            
            bool x_140657 = sle64((int64_t) 0, tmp_140656);
            
            // futhark/microgpt.fut:176:62-89
            
            bool y_140658 = slt64(tmp_140656, (int64_t) 4);
            
            // futhark/microgpt.fut:176:62-89
            
            bool bounds_check_140659 = x_140657 && y_140658;
            
            // futhark/microgpt.fut:176:62-89
            
            bool index_certs_140660;
            
            if (!bounds_check_140659) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140656, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:176:62-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:176:43-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:176:16-104\n   #9  futhark/microgpt.fut:414:7-67\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:176:96-99
            
            int64_t tmp_140661 = smod64(i_144429, (int64_t) 4);
            
            // futhark/microgpt.fut:176:62-101
            
            bool x_140662 = sle64((int64_t) 0, tmp_140661);
            
            // futhark/microgpt.fut:176:62-101
            
            bool y_140663 = slt64(tmp_140661, (int64_t) 4);
            
            // futhark/microgpt.fut:176:62-101
            
            bool bounds_check_140664 = x_140662 && y_140663;
            
            // futhark/microgpt.fut:176:62-101
            
            bool index_certs_140665;
            
            if (!bounds_check_140664) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140661, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:176:62-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:176:43-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:176:16-104\n   #9  futhark/microgpt.fut:414:7-67\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:142:5-184:137
            
            float lifted_lambda_res_140666 = ((float *) mem_145614)[i_144433 * (int64_t) 16 + tmp_140656 * (int64_t) 4 + tmp_140661];
            
            ((float *) mem_145646)[i_144429] = lifted_lambda_res_140666;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145641, i_144433 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145646, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145657_cached_sizze_147550 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145657, &mem_145657_cached_sizze_147550, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145662_cached_sizze_147551 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145662, &mem_145662_cached_sizze_147551, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144441 = 0; i_144441 < (int64_t) 16; i_144441++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144437 = 0; i_144437 < (int64_t) 16; i_144437++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140681;
            float r_140683 = 0.0F;
            
            for (int64_t i_140682 = 0; i_140682 < (int64_t) 16; i_140682++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140684 = ((float *) wout_mem_145213.mem)[i_144437 * (int64_t) 16 + i_140682];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float zt_rhs_140685 = ((float *) mem_145641)[i_144441 * (int64_t) 16 + i_140682];
                
                // futhark/microgpt.fut:177:83-125
                
                float zt_res_140686 = zt_lhs_140684 * zt_rhs_140685;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140687 = r_140683 + zt_res_140686;
                float r_tmp_147222 = zp_res_140687;
                
                r_140683 = r_tmp_147222;
            }
            defunc_0_lifted_lambda_res_140681 = r_140683;
            ((float *) mem_145662)[i_144437] = defunc_0_lifted_lambda_res_140681;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145657, i_144441 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145662, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145673_cached_sizze_147552 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145673, &mem_145673_cached_sizze_147552, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145678_cached_sizze_147553 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145678, &mem_145678_cached_sizze_147553, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144449 = 0; i_144449 < (int64_t) 16; i_144449++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144445 = 0; i_144445 < (int64_t) 16; i_144445++) {
            // futhark/microgpt.fut:142:5-184:137
            
            float zp_lhs_140702 = ((float *) mem_145238)[i_144449 * (int64_t) 16 + i_144445];
            
            // futhark/microgpt.fut:142:5-184:137
            
            float zp_rhs_140703 = ((float *) mem_145657)[i_144449 * (int64_t) 16 + i_144445];
            
            // futhark/microgpt.fut:178:51-97
            
            float zp_res_140704 = zp_lhs_140702 + zp_rhs_140703;
            
            ((float *) mem_145678)[i_144445] = zp_res_140704;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145673, i_144449 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145678, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145689_cached_sizze_147554 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145689, &mem_145689_cached_sizze_147554, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145694_cached_sizze_147555 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145694, &mem_145694_cached_sizze_147555, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145698_cached_sizze_147556 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145698, &mem_145698_cached_sizze_147556, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144461 = 0; i_144461 < (int64_t) 16; i_144461++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144457 = 0; i_144457 < (int64_t) 16; i_144457++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140719;
            float r_140721 = 0.0F;
            
            for (int64_t i_140720 = 0; i_140720 < (int64_t) 64; i_140720++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140722 = ((float *) wdown_mem_145211.mem)[i_144457 * (int64_t) 64 + i_140720];
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140723;
                float r_140725 = 0.0F;
                
                for (int64_t i_140724 = 0; i_140724 < (int64_t) 16; i_140724++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_140726 = ((float *) wup_mem_145217.mem)[i_140720 * (int64_t) 16 + i_140724];
                    
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_144453 = 0; i_144453 < (int64_t) 16; i_144453++) {
                        // futhark/microgpt.fut:142:5-184:137
                        
                        float zt_lhs_140733 = ((float *) mem_145673)[i_144461 * (int64_t) 16 + i_144453];
                        
                        // futhark/microgpt.fut:179:186-233
                        
                        float zt_res_140734 = zt_lhs_140733 * zt_lhs_140733;
                        
                        ((float *) mem_145698)[i_144453] = zt_res_140734;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_140736;
                    float r_140738 = 0.0F;
                    
                    for (int64_t i_140737 = 0; i_140737 < (int64_t) 16; i_140737++) {
                        // futhark/microgpt.fut:180:37-47
                        
                        float lifted_lambda_res_140739 = ((float *) mem_145698)[i_140737];
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_140740 = r_140738 + lifted_lambda_res_140739;
                        float r_tmp_147230 = zp_res_140740;
                        
                        r_140738 = r_tmp_147230;
                    }
                    defunc_0_lifted_lambda_res_140736 = r_140738;
                    // futhark/microgpt.fut:180:18-64
                    
                    float zs_res_140741 = defunc_0_lifted_lambda_res_140736 / 16.0F;
                    
                    // futhark/microgpt.fut:181:25-57
                    
                    float zp_res_140742 = 1.0e-5F + zs_res_140741;
                    
                    // futhark/microgpt.fut:181:17-57
                    
                    float sqrt_res_140743 = futrts_sqrt32(zp_res_140742);
                    
                    // futhark/microgpt.fut:142:5-184:137
                    
                    float zt_lhs_140744 = ((float *) mem_145673)[i_144461 * (int64_t) 16 + i_140724];
                    
                    // futhark/microgpt.fut:182:32-44
                    
                    float zs_res_140745 = 1.0F / sqrt_res_140743;
                    
                    // futhark/microgpt.fut:182:5-44
                    
                    float zt_res_140746 = zt_lhs_140744 * zs_res_140745;
                    
                    // futhark/microgpt.fut:179:133-182:44
                    
                    float zt_res_140747 = zt_lhs_140726 * zt_res_140746;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140748 = r_140725 + zt_res_140747;
                    float r_tmp_147228 = zp_res_140748;
                    
                    r_140725 = r_tmp_147228;
                }
                defunc_0_lifted_lambda_res_140723 = r_140725;
                // futhark/microgpt.fut:179:106-182:57
                
                float max_res_140749 = fmax32(0.0F, defunc_0_lifted_lambda_res_140723);
                
                // futhark/microgpt.fut:179:83-182:57
                
                float zt_res_140750 = zt_lhs_140722 * max_res_140749;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140751 = r_140721 + zt_res_140750;
                float r_tmp_147227 = zp_res_140751;
                
                r_140721 = r_tmp_147227;
            }
            defunc_0_lifted_lambda_res_140719 = r_140721;
            ((float *) mem_145694)[i_144457] = defunc_0_lifted_lambda_res_140719;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145689, i_144461 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145694, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145712_cached_sizze_147557 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145712, &mem_145712_cached_sizze_147557, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145717_cached_sizze_147558 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145717, &mem_145717_cached_sizze_147558, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144469 = 0; i_144469 < (int64_t) 16; i_144469++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144465 = 0; i_144465 < (int64_t) 16; i_144465++) {
            // futhark/microgpt.fut:142:5-184:137
            
            float zp_lhs_140766 = ((float *) mem_145673)[i_144469 * (int64_t) 16 + i_144465];
            
            // futhark/microgpt.fut:142:5-184:137
            
            float zp_rhs_140767 = ((float *) mem_145689)[i_144469 * (int64_t) 16 + i_144465];
            
            // futhark/microgpt.fut:183:51-98
            
            float zp_res_140768 = zp_lhs_140766 + zp_rhs_140767;
            
            ((float *) mem_145717)[i_144465] = zp_res_140768;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145712, i_144469 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145717, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145728_cached_sizze_147559 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_145728, &mem_145728_cached_sizze_147559, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145733_cached_sizze_147560 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145733, &mem_145733_cached_sizze_147560, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144477 = 0; i_144477 < (int64_t) 16; i_144477++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144473 = 0; i_144473 < (int64_t) 27; i_144473++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140784;
            float r_140786 = 0.0F;
            
            for (int64_t i_140785 = 0; i_140785 < (int64_t) 16; i_140785++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140787 = ((float *) wvoc_mem_145219.mem)[i_144473 * (int64_t) 16 + i_140785];
                
                // futhark/microgpt.fut:142:5-184:137
                
                float zt_rhs_140788 = ((float *) mem_145712)[i_144477 * (int64_t) 16 + i_140785];
                
                // futhark/microgpt.fut:184:70-110
                
                float zt_res_140789 = zt_lhs_140787 * zt_rhs_140788;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140790 = r_140786 + zt_res_140789;
                float r_tmp_147235 = zp_res_140790;
                
                r_140786 = r_tmp_147235;
            }
            defunc_0_lifted_lambda_res_140784 = r_140786;
            ((float *) mem_145733)[i_144473] = defunc_0_lifted_lambda_res_140784;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145728, i_144477 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145733, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:142:5-184:137
    if (memblock_alloc(ctx, &mem_145744, (int64_t) 1728, "mem_145744")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:142:5-184:137
    for (int64_t nest_i_147236 = 0; nest_i_147236 < (int64_t) 1; nest_i_147236++) {
        // futhark/microgpt.fut:142:5-184:137
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145744.mem, nest_i_147236 * (int64_t) 432, (int64_t []) {(int64_t) 27, (int64_t) 1}, (uint32_t *) mem_145728, (int64_t) 0, (int64_t []) {(int64_t) 27, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 27});
    }
    if (memblock_set(ctx, &mem_out_147157, &mem_145744, "mem_145744") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147498, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_145222);
        free(mem_145227);
        free(mem_145238);
        free(mem_145243);
        free(mem_145250);
        free(mem_145261);
        free(mem_145262);
        free(mem_145263);
        free(mem_145276);
        free(mem_145277);
        free(mem_145278);
        free(mem_145288);
        free(mem_145295);
        free(mem_145302);
        free(mem_145330);
        free(mem_145331);
        free(mem_145332);
        free(mem_145348);
        free(mem_145349);
        free(mem_145350);
        free(mem_145363);
        free(mem_145364);
        free(mem_145365);
        free(mem_145411);
        free(mem_145412);
        free(mem_145413);
        free(mem_145429);
        free(mem_145430);
        free(mem_145431);
        free(mem_145444);
        free(mem_145445);
        free(mem_145446);
        free(mem_145492);
        free(mem_145498);
        free(mem_145503);
        free(mem_145519);
        free(mem_145525);
        free(mem_145530);
        free(mem_145546);
        free(mem_145552);
        free(mem_145557);
        free(mem_145564);
        free(mem_145571);
        free(mem_145587);
        free(mem_145593);
        free(mem_145598);
        free(mem_145614);
        free(mem_145620);
        free(mem_145625);
        free(mem_145641);
        free(mem_145646);
        free(mem_145657);
        free(mem_145662);
        free(mem_145673);
        free(mem_145678);
        free(mem_145689);
        free(mem_145694);
        free(mem_145698);
        free(mem_145712);
        free(mem_145717);
        free(mem_145728);
        free(mem_145733);
        if (memblock_unref(ctx, &mem_145744, "mem_145744") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_loss(struct futhark_context *ctx, float *out_prim_out_147561, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock seqs_mem_145220, struct memblock masks_mem_145221, int64_t dl_83939)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145223_cached_sizze_147562 = 0;
    unsigned char *mem_145223 = NULL;
    int64_t mem_145228_cached_sizze_147563 = 0;
    unsigned char *mem_145228 = NULL;
    int64_t mem_145239_cached_sizze_147564 = 0;
    unsigned char *mem_145239 = NULL;
    int64_t mem_145244_cached_sizze_147565 = 0;
    unsigned char *mem_145244 = NULL;
    int64_t mem_145251_cached_sizze_147566 = 0;
    unsigned char *mem_145251 = NULL;
    int64_t mem_145262_cached_sizze_147567 = 0;
    unsigned char *mem_145262 = NULL;
    int64_t mem_145263_cached_sizze_147568 = 0;
    unsigned char *mem_145263 = NULL;
    int64_t mem_145264_cached_sizze_147569 = 0;
    unsigned char *mem_145264 = NULL;
    int64_t mem_145277_cached_sizze_147570 = 0;
    unsigned char *mem_145277 = NULL;
    int64_t mem_145278_cached_sizze_147571 = 0;
    unsigned char *mem_145278 = NULL;
    int64_t mem_145279_cached_sizze_147572 = 0;
    unsigned char *mem_145279 = NULL;
    int64_t mem_145289_cached_sizze_147573 = 0;
    unsigned char *mem_145289 = NULL;
    int64_t mem_145296_cached_sizze_147574 = 0;
    unsigned char *mem_145296 = NULL;
    int64_t mem_145303_cached_sizze_147575 = 0;
    unsigned char *mem_145303 = NULL;
    int64_t mem_145331_cached_sizze_147576 = 0;
    unsigned char *mem_145331 = NULL;
    int64_t mem_145332_cached_sizze_147577 = 0;
    unsigned char *mem_145332 = NULL;
    int64_t mem_145333_cached_sizze_147578 = 0;
    unsigned char *mem_145333 = NULL;
    int64_t mem_145349_cached_sizze_147579 = 0;
    unsigned char *mem_145349 = NULL;
    int64_t mem_145350_cached_sizze_147580 = 0;
    unsigned char *mem_145350 = NULL;
    int64_t mem_145351_cached_sizze_147581 = 0;
    unsigned char *mem_145351 = NULL;
    int64_t mem_145364_cached_sizze_147582 = 0;
    unsigned char *mem_145364 = NULL;
    int64_t mem_145365_cached_sizze_147583 = 0;
    unsigned char *mem_145365 = NULL;
    int64_t mem_145366_cached_sizze_147584 = 0;
    unsigned char *mem_145366 = NULL;
    int64_t mem_145412_cached_sizze_147585 = 0;
    unsigned char *mem_145412 = NULL;
    int64_t mem_145413_cached_sizze_147586 = 0;
    unsigned char *mem_145413 = NULL;
    int64_t mem_145414_cached_sizze_147587 = 0;
    unsigned char *mem_145414 = NULL;
    int64_t mem_145430_cached_sizze_147588 = 0;
    unsigned char *mem_145430 = NULL;
    int64_t mem_145431_cached_sizze_147589 = 0;
    unsigned char *mem_145431 = NULL;
    int64_t mem_145432_cached_sizze_147590 = 0;
    unsigned char *mem_145432 = NULL;
    int64_t mem_145445_cached_sizze_147591 = 0;
    unsigned char *mem_145445 = NULL;
    int64_t mem_145446_cached_sizze_147592 = 0;
    unsigned char *mem_145446 = NULL;
    int64_t mem_145447_cached_sizze_147593 = 0;
    unsigned char *mem_145447 = NULL;
    int64_t mem_145493_cached_sizze_147594 = 0;
    unsigned char *mem_145493 = NULL;
    int64_t mem_145499_cached_sizze_147595 = 0;
    unsigned char *mem_145499 = NULL;
    int64_t mem_145504_cached_sizze_147596 = 0;
    unsigned char *mem_145504 = NULL;
    int64_t mem_145520_cached_sizze_147597 = 0;
    unsigned char *mem_145520 = NULL;
    int64_t mem_145526_cached_sizze_147598 = 0;
    unsigned char *mem_145526 = NULL;
    int64_t mem_145531_cached_sizze_147599 = 0;
    unsigned char *mem_145531 = NULL;
    int64_t mem_145547_cached_sizze_147600 = 0;
    unsigned char *mem_145547 = NULL;
    int64_t mem_145553_cached_sizze_147601 = 0;
    unsigned char *mem_145553 = NULL;
    int64_t mem_145558_cached_sizze_147602 = 0;
    unsigned char *mem_145558 = NULL;
    int64_t mem_145565_cached_sizze_147603 = 0;
    unsigned char *mem_145565 = NULL;
    int64_t mem_145572_cached_sizze_147604 = 0;
    unsigned char *mem_145572 = NULL;
    int64_t mem_145588_cached_sizze_147605 = 0;
    unsigned char *mem_145588 = NULL;
    int64_t mem_145594_cached_sizze_147606 = 0;
    unsigned char *mem_145594 = NULL;
    int64_t mem_145599_cached_sizze_147607 = 0;
    unsigned char *mem_145599 = NULL;
    int64_t mem_145615_cached_sizze_147608 = 0;
    unsigned char *mem_145615 = NULL;
    int64_t mem_145621_cached_sizze_147609 = 0;
    unsigned char *mem_145621 = NULL;
    int64_t mem_145626_cached_sizze_147610 = 0;
    unsigned char *mem_145626 = NULL;
    int64_t mem_145642_cached_sizze_147611 = 0;
    unsigned char *mem_145642 = NULL;
    int64_t mem_145647_cached_sizze_147612 = 0;
    unsigned char *mem_145647 = NULL;
    int64_t mem_145658_cached_sizze_147613 = 0;
    unsigned char *mem_145658 = NULL;
    int64_t mem_145663_cached_sizze_147614 = 0;
    unsigned char *mem_145663 = NULL;
    int64_t mem_145674_cached_sizze_147615 = 0;
    unsigned char *mem_145674 = NULL;
    int64_t mem_145679_cached_sizze_147616 = 0;
    unsigned char *mem_145679 = NULL;
    int64_t mem_145690_cached_sizze_147617 = 0;
    unsigned char *mem_145690 = NULL;
    int64_t mem_145695_cached_sizze_147618 = 0;
    unsigned char *mem_145695 = NULL;
    int64_t mem_145699_cached_sizze_147619 = 0;
    unsigned char *mem_145699 = NULL;
    int64_t mem_145713_cached_sizze_147620 = 0;
    unsigned char *mem_145713 = NULL;
    int64_t mem_145718_cached_sizze_147621 = 0;
    unsigned char *mem_145718 = NULL;
    int64_t mem_145729_cached_sizze_147622 = 0;
    unsigned char *mem_145729 = NULL;
    int64_t mem_145734_cached_sizze_147623 = 0;
    unsigned char *mem_145734 = NULL;
    int64_t mem_145745_cached_sizze_147624 = 0;
    unsigned char *mem_145745 = NULL;
    int64_t mem_145749_cached_sizze_147625 = 0;
    unsigned char *mem_145749 = NULL;
    int64_t mem_145756_cached_sizze_147626 = 0;
    unsigned char *mem_145756 = NULL;
    int64_t mem_145763_cached_sizze_147627 = 0;
    unsigned char *mem_145763 = NULL;
    struct memblock ext_mem_145222;
    
    ext_mem_145222.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    float prim_out_147157;
    
    // futhark/microgpt.fut:421:33-54
    if (futrts_cal_target_9281(ctx, &ext_mem_145222, seqs_mem_145220, dl_83939) != 0) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145223_cached_sizze_147562 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145223, &mem_145223_cached_sizze_147562, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145228_cached_sizze_147563 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145228, &mem_145228_cached_sizze_147563, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144251 = 0; i_144251 < (int64_t) 16; i_144251++) {
        // futhark/microgpt.fut:4:11-25
        
        int64_t tmp_140203 = ((int64_t *) seqs_mem_145220.mem)[i_144251];
        
        // futhark/microgpt.fut:423:39-54
        
        bool x_140204 = sle64((int64_t) 0, tmp_140203);
        
        // futhark/microgpt.fut:423:39-54
        
        bool y_140205 = slt64(tmp_140203, (int64_t) 27);
        
        // futhark/microgpt.fut:423:39-54
        
        bool bounds_check_140206 = x_140204 && y_140205;
        
        // futhark/microgpt.fut:423:39-54
        
        bool index_certs_140207;
        
        if (!bounds_check_140206) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140203, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:423:39-54\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:423:14-58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144247 = 0; i_144247 < (int64_t) 16; i_144247++) {
            // futhark/microgpt.fut:4:11-25
            
            float lifted_lambda_res_140214 = ((float *) wte_mem_145216.mem)[tmp_140203 * (int64_t) 16 + i_144247];
            
            ((float *) mem_145228)[i_144247] = lifted_lambda_res_140214;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145223, i_144251 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145228, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145239_cached_sizze_147564 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145239, &mem_145239_cached_sizze_147564, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145244_cached_sizze_147565 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145244, &mem_145244_cached_sizze_147565, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145251_cached_sizze_147566 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145251, &mem_145251_cached_sizze_147566, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144263 = 0; i_144263 < (int64_t) 16; i_144263++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144255 = 0; i_144255 < (int64_t) 16; i_144255++) {
            float zp_lhs_140245 = ((float *) mem_145223)[i_144263 * (int64_t) 16 + i_144255];
            
            // futhark/microgpt.fut:4:11-25
            
            float zp_rhs_140246 = ((float *) wpe_mem_145214.mem)[i_144263 * (int64_t) 16 + i_144255];
            
            // futhark/microgpt.fut:205:68-103
            
            float zp_res_140247 = zp_lhs_140245 + zp_rhs_140246;
            
            // futhark/microgpt.fut:205:86-145
            
            float zt_res_140248 = zp_res_140247 * zp_res_140247;
            
            ((float *) mem_145244)[i_144255] = zt_res_140248;
        }
        // futhark/microgpt.fut:71:13-49
        
        float defunc_0_lifted_lambda_res_140250;
        float r_140252 = 0.0F;
        
        for (int64_t i_140251 = 0; i_140251 < (int64_t) 16; i_140251++) {
            // futhark/microgpt.fut:206:35-43
            
            float lifted_lambda_res_140253 = ((float *) mem_145244)[i_140251];
            
            // futhark/microgpt.fut:71:40-49
            
            float zp_res_140254 = r_140252 + lifted_lambda_res_140253;
            float r_tmp_147162 = zp_res_140254;
            
            r_140252 = r_tmp_147162;
        }
        defunc_0_lifted_lambda_res_140250 = r_140252;
        // futhark/microgpt.fut:206:17-60
        
        float zs_res_140255 = defunc_0_lifted_lambda_res_140250 / 16.0F;
        
        // futhark/microgpt.fut:207:24-55
        
        float zp_res_140256 = 1.0e-5F + zs_res_140255;
        
        // futhark/microgpt.fut:207:16-55
        
        float sqrt_res_140257 = futrts_sqrt32(zp_res_140256);
        
        // futhark/microgpt.fut:208:67-78
        
        float zs_res_140258 = 1.0F / sqrt_res_140257;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144259 = 0; i_144259 < (int64_t) 16; i_144259++) {
            float zp_lhs_140265 = ((float *) mem_145223)[i_144263 * (int64_t) 16 + i_144259];
            
            // futhark/microgpt.fut:4:11-25
            
            float zp_rhs_140266 = ((float *) wpe_mem_145214.mem)[i_144263 * (int64_t) 16 + i_144259];
            
            // futhark/microgpt.fut:208:25-60
            
            float zp_res_140267 = zp_lhs_140265 + zp_rhs_140266;
            
            // futhark/microgpt.fut:208:43-78
            
            float zt_res_140268 = zs_res_140258 * zp_res_140267;
            
            ((float *) mem_145251)[i_144259] = zt_res_140268;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145239, i_144263 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145251, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145262_cached_sizze_147567 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145262, &mem_145262_cached_sizze_147567, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145263_cached_sizze_147568 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145263, &mem_145263_cached_sizze_147568, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145264_cached_sizze_147569 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145264, &mem_145264_cached_sizze_147569, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145277_cached_sizze_147570 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145277, &mem_145277_cached_sizze_147570, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145278_cached_sizze_147571 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145278, &mem_145278_cached_sizze_147571, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145279_cached_sizze_147572 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145279, &mem_145279_cached_sizze_147572, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145289_cached_sizze_147573 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145289, &mem_145289_cached_sizze_147573, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145296_cached_sizze_147574 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145296, &mem_145296_cached_sizze_147574, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145303_cached_sizze_147575 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145303, &mem_145303_cached_sizze_147575, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144293 = 0; i_144293 < (int64_t) 16; i_144293++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144283 = 0; i_144283 < (int64_t) 16; i_144283++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141488;
            float r_141490 = 0.0F;
            
            for (int64_t i_141489 = 0; i_141489 < (int64_t) 16; i_141489++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141491 = ((float *) wqry_mem_145215.mem)[i_144283 * (int64_t) 16 + i_141489];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144267 = 0; i_144267 < (int64_t) 16; i_144267++) {
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_141498 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_144267];
                    
                    // futhark/microgpt.fut:209:128-167
                    
                    float zt_res_141499 = zt_lhs_141498 * zt_lhs_141498;
                    
                    ((float *) mem_145289)[i_144267] = zt_res_141499;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141501;
                float r_141503 = 0.0F;
                
                for (int64_t i_141502 = 0; i_141502 < (int64_t) 16; i_141502++) {
                    // futhark/microgpt.fut:210:35-43
                    
                    float lifted_lambda_res_141504 = ((float *) mem_145289)[i_141502];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141505 = r_141503 + lifted_lambda_res_141504;
                    float r_tmp_147172 = zp_res_141505;
                    
                    r_141503 = r_tmp_147172;
                }
                defunc_0_lifted_lambda_res_141501 = r_141503;
                // futhark/microgpt.fut:210:17-60
                
                float zs_res_141506 = defunc_0_lifted_lambda_res_141501 / 16.0F;
                
                // futhark/microgpt.fut:211:24-55
                
                float zp_res_141507 = 1.0e-5F + zs_res_141506;
                
                // futhark/microgpt.fut:211:16-55
                
                float sqrt_res_141508 = futrts_sqrt32(zp_res_141507);
                
                // futhark/microgpt.fut:199:5-245:83
                
                float zt_lhs_141509 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_141489];
                
                // futhark/microgpt.fut:212:28-39
                
                float zs_res_141510 = 1.0F / sqrt_res_141508;
                
                // futhark/microgpt.fut:212:5-39
                
                float zt_res_141511 = zt_lhs_141509 * zs_res_141510;
                
                // futhark/microgpt.fut:209:78-212:39
                
                float zt_res_141512 = zt_lhs_141491 * zt_res_141511;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141513 = r_141490 + zt_res_141512;
                float r_tmp_147170 = zp_res_141513;
                
                r_141490 = r_tmp_147170;
            }
            defunc_0_lifted_lambda_res_141488 = r_141490;
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141520;
            float r_141522 = 0.0F;
            
            for (int64_t i_141521 = 0; i_141521 < (int64_t) 16; i_141521++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141523 = ((float *) wkey_mem_145212.mem)[i_144283 * (int64_t) 16 + i_141521];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144271 = 0; i_144271 < (int64_t) 16; i_144271++) {
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_141530 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_144271];
                    
                    // futhark/microgpt.fut:213:128-167
                    
                    float zt_res_141531 = zt_lhs_141530 * zt_lhs_141530;
                    
                    ((float *) mem_145296)[i_144271] = zt_res_141531;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141533;
                float r_141535 = 0.0F;
                
                for (int64_t i_141534 = 0; i_141534 < (int64_t) 16; i_141534++) {
                    // futhark/microgpt.fut:214:35-43
                    
                    float lifted_lambda_res_141536 = ((float *) mem_145296)[i_141534];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141537 = r_141535 + lifted_lambda_res_141536;
                    float r_tmp_147175 = zp_res_141537;
                    
                    r_141535 = r_tmp_147175;
                }
                defunc_0_lifted_lambda_res_141533 = r_141535;
                // futhark/microgpt.fut:214:17-60
                
                float zs_res_141538 = defunc_0_lifted_lambda_res_141533 / 16.0F;
                
                // futhark/microgpt.fut:215:24-55
                
                float zp_res_141539 = 1.0e-5F + zs_res_141538;
                
                // futhark/microgpt.fut:215:16-55
                
                float sqrt_res_141540 = futrts_sqrt32(zp_res_141539);
                
                // futhark/microgpt.fut:199:5-245:83
                
                float zt_lhs_141541 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_141521];
                
                // futhark/microgpt.fut:216:28-39
                
                float zs_res_141542 = 1.0F / sqrt_res_141540;
                
                // futhark/microgpt.fut:216:5-39
                
                float zt_res_141543 = zt_lhs_141541 * zs_res_141542;
                
                // futhark/microgpt.fut:213:78-216:39
                
                float zt_res_141544 = zt_lhs_141523 * zt_res_141543;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141545 = r_141522 + zt_res_141544;
                float r_tmp_147173 = zp_res_141545;
                
                r_141522 = r_tmp_147173;
            }
            defunc_0_lifted_lambda_res_141520 = r_141522;
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141555;
            float r_141557 = 0.0F;
            
            for (int64_t i_141556 = 0; i_141556 < (int64_t) 16; i_141556++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_141558 = ((float *) wval_mem_145218.mem)[i_144283 * (int64_t) 16 + i_141556];
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144275 = 0; i_144275 < (int64_t) 16; i_144275++) {
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_141565 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_144275];
                    
                    // futhark/microgpt.fut:217:128-167
                    
                    float zt_res_141566 = zt_lhs_141565 * zt_lhs_141565;
                    
                    ((float *) mem_145303)[i_144275] = zt_res_141566;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_141568;
                float r_141570 = 0.0F;
                
                for (int64_t i_141569 = 0; i_141569 < (int64_t) 16; i_141569++) {
                    // futhark/microgpt.fut:218:35-43
                    
                    float lifted_lambda_res_141571 = ((float *) mem_145303)[i_141569];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_141572 = r_141570 + lifted_lambda_res_141571;
                    float r_tmp_147178 = zp_res_141572;
                    
                    r_141570 = r_tmp_147178;
                }
                defunc_0_lifted_lambda_res_141568 = r_141570;
                // futhark/microgpt.fut:218:17-60
                
                float zs_res_141573 = defunc_0_lifted_lambda_res_141568 / 16.0F;
                
                // futhark/microgpt.fut:219:24-55
                
                float zp_res_141574 = 1.0e-5F + zs_res_141573;
                
                // futhark/microgpt.fut:219:16-55
                
                float sqrt_res_141575 = futrts_sqrt32(zp_res_141574);
                
                // futhark/microgpt.fut:199:5-245:83
                
                float zt_lhs_141576 = ((float *) mem_145239)[i_144293 * (int64_t) 16 + i_141556];
                
                // futhark/microgpt.fut:220:28-39
                
                float zs_res_141577 = 1.0F / sqrt_res_141575;
                
                // futhark/microgpt.fut:220:5-39
                
                float zt_res_141578 = zt_lhs_141576 * zs_res_141577;
                
                // futhark/microgpt.fut:217:78-220:39
                
                float zt_res_141579 = zt_lhs_141558 * zt_res_141578;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141580 = r_141557 + zt_res_141579;
                float r_tmp_147176 = zp_res_141580;
                
                r_141557 = r_tmp_147176;
            }
            defunc_0_lifted_lambda_res_141555 = r_141557;
            ((float *) mem_145277)[i_144283] = defunc_0_lifted_lambda_res_141555;
            ((float *) mem_145278)[i_144283] = defunc_0_lifted_lambda_res_141520;
            ((float *) mem_145279)[i_144283] = defunc_0_lifted_lambda_res_141488;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145262, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145277, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145263, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145278, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145264, i_144293 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145279, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145331_cached_sizze_147576 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145331, &mem_145331_cached_sizze_147576, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145332_cached_sizze_147577 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145332, &mem_145332_cached_sizze_147577, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145333_cached_sizze_147578 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145333, &mem_145333_cached_sizze_147578, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145349_cached_sizze_147579 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145349, &mem_145349_cached_sizze_147579, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145350_cached_sizze_147580 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145350, &mem_145350_cached_sizze_147580, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145351_cached_sizze_147581 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145351, &mem_145351_cached_sizze_147581, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145364_cached_sizze_147582 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145364, &mem_145364_cached_sizze_147582, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145365_cached_sizze_147583 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145365, &mem_145365_cached_sizze_147583, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145366_cached_sizze_147584 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145366, &mem_145366_cached_sizze_147584, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144323 = 0; i_144323 < (int64_t) 16; i_144323++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144313 = 0; i_144313 < (int64_t) 4; i_144313++) {
            // futhark/microgpt.fut:221:94-97
            
            int64_t zp_lhs_141645 = mul64((int64_t) 4, i_144313);
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144303 = 0; i_144303 < (int64_t) 4; i_144303++) {
                // futhark/microgpt.fut:221:99-104
                
                int64_t tmp_141729 = add64(zp_lhs_141645, i_144303);
                
                // futhark/microgpt.fut:221:75-106
                
                bool x_141730 = sle64((int64_t) 0, tmp_141729);
                
                // futhark/microgpt.fut:221:75-106
                
                bool y_141731 = slt64(tmp_141729, (int64_t) 16);
                
                // futhark/microgpt.fut:221:75-106
                
                bool bounds_check_141732 = x_141730 && y_141731;
                
                // futhark/microgpt.fut:221:75-106
                
                bool index_certs_141733;
                
                if (!bounds_check_141732) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_141729, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:221:75-106\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:221:58-107\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:221:40-109\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:9:27-39\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:9:13-40\n   #11 futhark/microgpt.fut:221:15-111\n   #12 futhark/microgpt.fut:424:7-72\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141734 = ((float *) mem_145264)[i_144323 * (int64_t) 16 + tmp_141729];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141742 = ((float *) mem_145263)[i_144323 * (int64_t) 16 + tmp_141729];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141753 = ((float *) mem_145262)[i_144323 * (int64_t) 16 + tmp_141729];
                
                ((float *) mem_145364)[i_144303] = lifted_lambda_res_141753;
                ((float *) mem_145365)[i_144303] = lifted_lambda_res_141742;
                ((float *) mem_145366)[i_144303] = lifted_lambda_res_141734;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145349, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145364, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145350, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145365, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145351, i_144313 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145366, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145331, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145349, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145332, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145350, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145333, i_144323 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145351, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145412_cached_sizze_147585 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145412, &mem_145412_cached_sizze_147585, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145413_cached_sizze_147586 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145413, &mem_145413_cached_sizze_147586, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145414_cached_sizze_147587 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145414, &mem_145414_cached_sizze_147587, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145430_cached_sizze_147588 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145430, &mem_145430_cached_sizze_147588, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145431_cached_sizze_147589 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145431, &mem_145431_cached_sizze_147589, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145432_cached_sizze_147590 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145432, &mem_145432_cached_sizze_147590, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145445_cached_sizze_147591 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145445, &mem_145445_cached_sizze_147591, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145446_cached_sizze_147592 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145446, &mem_145446_cached_sizze_147592, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145447_cached_sizze_147593 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145447, &mem_145447_cached_sizze_147593, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144353 = 0; i_144353 < (int64_t) 4; i_144353++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144343 = 0; i_144343 < (int64_t) 16; i_144343++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144333 = 0; i_144333 < (int64_t) 4; i_144333++) {
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141914 = ((float *) mem_145333)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141921 = ((float *) mem_145332)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141931 = ((float *) mem_145331)[i_144343 * (int64_t) 16 + i_144353 * (int64_t) 4 + i_144333];
                
                ((float *) mem_145445)[i_144333] = lifted_lambda_res_141931;
                ((float *) mem_145446)[i_144333] = lifted_lambda_res_141921;
                ((float *) mem_145447)[i_144333] = lifted_lambda_res_141914;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145430, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145445, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145431, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145446, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145432, i_144343 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145447, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145412, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145430, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145413, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145431, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145414, i_144353 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145432, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145493_cached_sizze_147594 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145493, &mem_145493_cached_sizze_147594, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145499_cached_sizze_147595 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145499, &mem_145499_cached_sizze_147595, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145504_cached_sizze_147596 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145504, &mem_145504_cached_sizze_147596, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144367 = 0; i_144367 < (int64_t) 4; i_144367++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144363 = 0; i_144363 < (int64_t) 16; i_144363++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144359 = 0; i_144359 < (int64_t) 16; i_144359++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140536;
                float r_140538 = 0.0F;
                
                for (int64_t i_140537 = 0; i_140537 < (int64_t) 4; i_140537++) {
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_140539 = ((float *) mem_145414)[i_144367 * (int64_t) 64 + i_144363 * (int64_t) 4 + i_140537];
                    
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_rhs_140540 = ((float *) mem_145413)[i_144367 * (int64_t) 64 + i_144359 * (int64_t) 4 + i_140537];
                    
                    // futhark/microgpt.fut:227:96-145
                    
                    float zt_res_140541 = zt_lhs_140539 * zt_rhs_140540;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140542 = r_140538 + zt_res_140541;
                    float r_tmp_147200 = zp_res_140542;
                    
                    r_140538 = r_tmp_147200;
                }
                defunc_0_lifted_lambda_res_140536 = r_140538;
                ((float *) mem_145504)[i_144359] = defunc_0_lifted_lambda_res_140536;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145499, i_144363 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145504, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145493, i_144367 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145499, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145520_cached_sizze_147597 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145520, &mem_145520_cached_sizze_147597, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145526_cached_sizze_147598 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145526, &mem_145526_cached_sizze_147598, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145531_cached_sizze_147599 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145531, &mem_145531_cached_sizze_147599, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144379 = 0; i_144379 < (int64_t) 4; i_144379++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144375 = 0; i_144375 < (int64_t) 16; i_144375++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144371 = 0; i_144371 < (int64_t) 16; i_144371++) {
                // futhark/microgpt.fut:199:5-245:83
                
                float zs_lhs_140564 = ((float *) mem_145493)[i_144379 * (int64_t) 256 + i_144375 * (int64_t) 16 + i_144371];
                
                // futhark/microgpt.fut:228:79-116
                
                float zs_res_140565 = zs_lhs_140564 / 2.0F;
                
                // futhark/microgpt.fut:4:11-25
                
                float zp_rhs_140566 = ((float *) masks_mem_145221.mem)[i_144375 * (int64_t) 16 + i_144371];
                
                // futhark/microgpt.fut:228:103-141
                
                float zp_res_140567 = zs_res_140565 + zp_rhs_140566;
                
                ((float *) mem_145531)[i_144371] = zp_res_140567;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145526, i_144375 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145531, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145520, i_144379 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145526, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145547_cached_sizze_147600 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145547, &mem_145547_cached_sizze_147600, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145553_cached_sizze_147601 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145553, &mem_145553_cached_sizze_147601, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145558_cached_sizze_147602 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145558, &mem_145558_cached_sizze_147602, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145565_cached_sizze_147603 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145565, &mem_145565_cached_sizze_147603, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145572_cached_sizze_147604 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145572, &mem_145572_cached_sizze_147604, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144401 = 0; i_144401 < (int64_t) 4; i_144401++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144397 = 0; i_144397 < (int64_t) 16; i_144397++) {
            // futhark/microgpt.fut:103:13-33
            
            float defunc_0_reduce_res_142031;
            float redout_144381 = -INFINITY;
            
            for (int64_t i_144382 = 0; i_144382 < (int64_t) 16; i_144382++) {
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_141960 = ((float *) mem_145520)[i_144401 * (int64_t) 256 + i_144397 * (int64_t) 16 + i_144382];
                
                // futhark/microgpt.fut:103:13-33
                
                float max_res_140595 = fmax32(lifted_lambda_res_141960, redout_144381);
                float redout_tmp_147206 = max_res_140595;
                
                redout_144381 = redout_tmp_147206;
            }
            defunc_0_reduce_res_142031 = redout_144381;
            // futhark/microgpt.fut:113:47-56
            
            float neg_res_140596 = -defunc_0_reduce_res_142031;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144385 = 0; i_144385 < (int64_t) 16; i_144385++) {
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_140603 = ((float *) mem_145520)[i_144401 * (int64_t) 256 + i_144397 * (int64_t) 16 + i_144385];
                
                // futhark/microgpt.fut:113:38-56
                
                float zp_res_140604 = neg_res_140596 + lifted_lambda_res_140603;
                
                // futhark/microgpt.fut:113:31-56
                
                float exp_res_140605 = futrts_exp32(zp_res_140604);
                
                ((float *) mem_145558)[i_144385] = exp_res_140605;
            }
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140607;
            float r_140609 = 0.0F;
            
            for (int64_t i_140608 = 0; i_140608 < (int64_t) 16; i_140608++) {
                // futhark/microgpt.fut:114:32-39
                
                float lifted_lambda_res_140610 = ((float *) mem_145558)[i_140608];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140611 = r_140609 + lifted_lambda_res_140610;
                float r_tmp_147208 = zp_res_140611;
                
                r_140609 = r_tmp_147208;
            }
            defunc_0_lifted_lambda_res_140607 = r_140609;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144389 = 0; i_144389 < (int64_t) 16; i_144389++) {
                // futhark/microgpt.fut:115:23-30
                
                float zs_lhs_140618 = ((float *) mem_145558)[i_144389];
                
                // futhark/microgpt.fut:115:23-40
                
                float zs_res_140619 = zs_lhs_140618 / defunc_0_lifted_lambda_res_140607;
                
                ((float *) mem_145565)[i_144389] = zs_res_140619;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144393 = 0; i_144393 < (int64_t) 16; i_144393++) {
                // futhark/microgpt.fut:230:23-31
                
                float lifted_lambda_res_140627 = ((float *) mem_145565)[i_144393];
                
                ((float *) mem_145572)[i_144393] = lifted_lambda_res_140627;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145553, i_144397 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145572, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145547, i_144401 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145553, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145588_cached_sizze_147605 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145588, &mem_145588_cached_sizze_147605, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145594_cached_sizze_147606 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145594, &mem_145594_cached_sizze_147606, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145599_cached_sizze_147607 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145599, &mem_145599_cached_sizze_147607, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144413 = 0; i_144413 < (int64_t) 4; i_144413++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144409 = 0; i_144409 < (int64_t) 16; i_144409++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144405 = 0; i_144405 < (int64_t) 4; i_144405++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140649;
                float r_140651 = 0.0F;
                
                for (int64_t i_140650 = 0; i_140650 < (int64_t) 16; i_140650++) {
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_140652 = ((float *) mem_145547)[i_144413 * (int64_t) 256 + i_144409 * (int64_t) 16 + i_140650];
                    
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_rhs_140653 = ((float *) mem_145412)[i_144413 * (int64_t) 64 + i_140650 * (int64_t) 4 + i_144405];
                    
                    // futhark/microgpt.fut:231:99-153
                    
                    float zt_res_140654 = zt_lhs_140652 * zt_rhs_140653;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140655 = r_140651 + zt_res_140654;
                    float r_tmp_147214 = zp_res_140655;
                    
                    r_140651 = r_tmp_147214;
                }
                defunc_0_lifted_lambda_res_140649 = r_140651;
                ((float *) mem_145599)[i_144405] = defunc_0_lifted_lambda_res_140649;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145594, i_144409 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145599, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145588, i_144413 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145594, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145615_cached_sizze_147608 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145615, &mem_145615_cached_sizze_147608, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145621_cached_sizze_147609 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145621, &mem_145621_cached_sizze_147609, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145626_cached_sizze_147610 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145626, &mem_145626_cached_sizze_147610, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144425 = 0; i_144425 < (int64_t) 16; i_144425++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144421 = 0; i_144421 < (int64_t) 4; i_144421++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144417 = 0; i_144417 < (int64_t) 4; i_144417++) {
                // futhark/microgpt.fut:199:5-245:83
                
                float lifted_lambda_res_140677 = ((float *) mem_145588)[i_144421 * (int64_t) 64 + i_144425 * (int64_t) 4 + i_144417];
                
                ((float *) mem_145626)[i_144417] = lifted_lambda_res_140677;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145621, i_144421 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145626, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
        }
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_145615, i_144425 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145621, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145642_cached_sizze_147611 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145642, &mem_145642_cached_sizze_147611, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145647_cached_sizze_147612 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145647, &mem_145647_cached_sizze_147612, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144433 = 0; i_144433 < (int64_t) 16; i_144433++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144429 = 0; i_144429 < (int64_t) 16; i_144429++) {
            // futhark/microgpt.fut:233:84-87
            
            int64_t tmp_140689 = sdiv64(i_144429, (int64_t) 4);
            
            // futhark/microgpt.fut:233:62-89
            
            bool x_140690 = sle64((int64_t) 0, tmp_140689);
            
            // futhark/microgpt.fut:233:62-89
            
            bool y_140691 = slt64(tmp_140689, (int64_t) 4);
            
            // futhark/microgpt.fut:233:62-89
            
            bool bounds_check_140692 = x_140690 && y_140691;
            
            // futhark/microgpt.fut:233:62-89
            
            bool index_certs_140693;
            
            if (!bounds_check_140692) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140689, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:233:62-89\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:233:43-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:233:16-104\n   #9  futhark/microgpt.fut:424:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:233:96-99
            
            int64_t tmp_140694 = smod64(i_144429, (int64_t) 4);
            
            // futhark/microgpt.fut:233:62-101
            
            bool x_140695 = sle64((int64_t) 0, tmp_140694);
            
            // futhark/microgpt.fut:233:62-101
            
            bool y_140696 = slt64(tmp_140694, (int64_t) 4);
            
            // futhark/microgpt.fut:233:62-101
            
            bool bounds_check_140697 = x_140695 && y_140696;
            
            // futhark/microgpt.fut:233:62-101
            
            bool index_certs_140698;
            
            if (!bounds_check_140697) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140694, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:233:62-101\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:233:43-102\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:233:16-104\n   #9  futhark/microgpt.fut:424:7-72\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:199:5-245:83
            
            float lifted_lambda_res_140699 = ((float *) mem_145615)[i_144433 * (int64_t) 16 + tmp_140689 * (int64_t) 4 + tmp_140694];
            
            ((float *) mem_145647)[i_144429] = lifted_lambda_res_140699;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145642, i_144433 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145647, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145658_cached_sizze_147613 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145658, &mem_145658_cached_sizze_147613, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145663_cached_sizze_147614 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145663, &mem_145663_cached_sizze_147614, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144441 = 0; i_144441 < (int64_t) 16; i_144441++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144437 = 0; i_144437 < (int64_t) 16; i_144437++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140714;
            float r_140716 = 0.0F;
            
            for (int64_t i_140715 = 0; i_140715 < (int64_t) 16; i_140715++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140717 = ((float *) wout_mem_145213.mem)[i_144437 * (int64_t) 16 + i_140715];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float zt_rhs_140718 = ((float *) mem_145642)[i_144441 * (int64_t) 16 + i_140715];
                
                // futhark/microgpt.fut:234:83-125
                
                float zt_res_140719 = zt_lhs_140717 * zt_rhs_140718;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140720 = r_140716 + zt_res_140719;
                float r_tmp_147222 = zp_res_140720;
                
                r_140716 = r_tmp_147222;
            }
            defunc_0_lifted_lambda_res_140714 = r_140716;
            ((float *) mem_145663)[i_144437] = defunc_0_lifted_lambda_res_140714;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145658, i_144441 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145663, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145674_cached_sizze_147615 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145674, &mem_145674_cached_sizze_147615, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145679_cached_sizze_147616 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145679, &mem_145679_cached_sizze_147616, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144449 = 0; i_144449 < (int64_t) 16; i_144449++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144445 = 0; i_144445 < (int64_t) 16; i_144445++) {
            // futhark/microgpt.fut:199:5-245:83
            
            float zp_lhs_140735 = ((float *) mem_145239)[i_144449 * (int64_t) 16 + i_144445];
            
            // futhark/microgpt.fut:199:5-245:83
            
            float zp_rhs_140736 = ((float *) mem_145658)[i_144449 * (int64_t) 16 + i_144445];
            
            // futhark/microgpt.fut:235:51-97
            
            float zp_res_140737 = zp_lhs_140735 + zp_rhs_140736;
            
            ((float *) mem_145679)[i_144445] = zp_res_140737;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145674, i_144449 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145679, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145690_cached_sizze_147617 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145690, &mem_145690_cached_sizze_147617, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145695_cached_sizze_147618 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145695, &mem_145695_cached_sizze_147618, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145699_cached_sizze_147619 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145699, &mem_145699_cached_sizze_147619, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144461 = 0; i_144461 < (int64_t) 16; i_144461++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144457 = 0; i_144457 < (int64_t) 16; i_144457++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140752;
            float r_140754 = 0.0F;
            
            for (int64_t i_140753 = 0; i_140753 < (int64_t) 64; i_140753++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140755 = ((float *) wdown_mem_145211.mem)[i_144457 * (int64_t) 64 + i_140753];
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_140756;
                float r_140758 = 0.0F;
                
                for (int64_t i_140757 = 0; i_140757 < (int64_t) 16; i_140757++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_140759 = ((float *) wup_mem_145217.mem)[i_140753 * (int64_t) 16 + i_140757];
                    
                    // futhark/microgpt.fut:4:11-25
                    for (int64_t i_144453 = 0; i_144453 < (int64_t) 16; i_144453++) {
                        // futhark/microgpt.fut:199:5-245:83
                        
                        float zt_lhs_140766 = ((float *) mem_145674)[i_144461 * (int64_t) 16 + i_144453];
                        
                        // futhark/microgpt.fut:236:186-233
                        
                        float zt_res_140767 = zt_lhs_140766 * zt_lhs_140766;
                        
                        ((float *) mem_145699)[i_144453] = zt_res_140767;
                    }
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_140769;
                    float r_140771 = 0.0F;
                    
                    for (int64_t i_140770 = 0; i_140770 < (int64_t) 16; i_140770++) {
                        // futhark/microgpt.fut:237:37-47
                        
                        float lifted_lambda_res_140772 = ((float *) mem_145699)[i_140770];
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_140773 = r_140771 + lifted_lambda_res_140772;
                        float r_tmp_147230 = zp_res_140773;
                        
                        r_140771 = r_tmp_147230;
                    }
                    defunc_0_lifted_lambda_res_140769 = r_140771;
                    // futhark/microgpt.fut:237:18-64
                    
                    float zs_res_140774 = defunc_0_lifted_lambda_res_140769 / 16.0F;
                    
                    // futhark/microgpt.fut:238:25-57
                    
                    float zp_res_140775 = 1.0e-5F + zs_res_140774;
                    
                    // futhark/microgpt.fut:238:17-57
                    
                    float sqrt_res_140776 = futrts_sqrt32(zp_res_140775);
                    
                    // futhark/microgpt.fut:199:5-245:83
                    
                    float zt_lhs_140777 = ((float *) mem_145674)[i_144461 * (int64_t) 16 + i_140757];
                    
                    // futhark/microgpt.fut:239:32-44
                    
                    float zs_res_140778 = 1.0F / sqrt_res_140776;
                    
                    // futhark/microgpt.fut:239:5-44
                    
                    float zt_res_140779 = zt_lhs_140777 * zs_res_140778;
                    
                    // futhark/microgpt.fut:236:133-239:44
                    
                    float zt_res_140780 = zt_lhs_140759 * zt_res_140779;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_140781 = r_140758 + zt_res_140780;
                    float r_tmp_147228 = zp_res_140781;
                    
                    r_140758 = r_tmp_147228;
                }
                defunc_0_lifted_lambda_res_140756 = r_140758;
                // futhark/microgpt.fut:236:106-239:57
                
                float max_res_140782 = fmax32(0.0F, defunc_0_lifted_lambda_res_140756);
                
                // futhark/microgpt.fut:236:83-239:57
                
                float zt_res_140783 = zt_lhs_140755 * max_res_140782;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140784 = r_140754 + zt_res_140783;
                float r_tmp_147227 = zp_res_140784;
                
                r_140754 = r_tmp_147227;
            }
            defunc_0_lifted_lambda_res_140752 = r_140754;
            ((float *) mem_145695)[i_144457] = defunc_0_lifted_lambda_res_140752;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145690, i_144461 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145695, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145713_cached_sizze_147620 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145713, &mem_145713_cached_sizze_147620, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145718_cached_sizze_147621 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145718, &mem_145718_cached_sizze_147621, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144469 = 0; i_144469 < (int64_t) 16; i_144469++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144465 = 0; i_144465 < (int64_t) 16; i_144465++) {
            // futhark/microgpt.fut:199:5-245:83
            
            float zp_lhs_140799 = ((float *) mem_145674)[i_144469 * (int64_t) 16 + i_144465];
            
            // futhark/microgpt.fut:199:5-245:83
            
            float zp_rhs_140800 = ((float *) mem_145690)[i_144469 * (int64_t) 16 + i_144465];
            
            // futhark/microgpt.fut:240:51-98
            
            float zp_res_140801 = zp_lhs_140799 + zp_rhs_140800;
            
            ((float *) mem_145718)[i_144465] = zp_res_140801;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145713, i_144469 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145718, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145729_cached_sizze_147622 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_145729, &mem_145729_cached_sizze_147622, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145734_cached_sizze_147623 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145734, &mem_145734_cached_sizze_147623, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144477 = 0; i_144477 < (int64_t) 16; i_144477++) {
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144473 = 0; i_144473 < (int64_t) 27; i_144473++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_140817;
            float r_140819 = 0.0F;
            
            for (int64_t i_140818 = 0; i_140818 < (int64_t) 16; i_140818++) {
                // futhark/microgpt.fut:71:46-49
                
                float zt_lhs_140820 = ((float *) wvoc_mem_145219.mem)[i_144473 * (int64_t) 16 + i_140818];
                
                // futhark/microgpt.fut:199:5-245:83
                
                float zt_rhs_140821 = ((float *) mem_145713)[i_144477 * (int64_t) 16 + i_140818];
                
                // futhark/microgpt.fut:241:83-125
                
                float zt_res_140822 = zt_lhs_140820 * zt_rhs_140821;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_140823 = r_140819 + zt_res_140822;
                float r_tmp_147235 = zp_res_140823;
                
                r_140819 = r_tmp_147235;
            }
            defunc_0_lifted_lambda_res_140817 = r_140819;
            ((float *) mem_145734)[i_144473] = defunc_0_lifted_lambda_res_140817;
        }
        lmad_copy_4b(ctx, 1, (uint32_t *) mem_145729, i_144477 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145734, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145745_cached_sizze_147624 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145745, &mem_145745_cached_sizze_147624, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145749_cached_sizze_147625 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145749, &mem_145749_cached_sizze_147625, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145756_cached_sizze_147626 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145756, &mem_145756_cached_sizze_147626, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145763_cached_sizze_147627 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145763, &mem_145763_cached_sizze_147627, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    for (int64_t i_144495 = 0; i_144495 < (int64_t) 16; i_144495++) {
        // futhark/microgpt.fut:103:13-33
        
        float defunc_0_reduce_res_142047;
        float redout_144479 = -INFINITY;
        
        for (int64_t i_144480 = 0; i_144480 < (int64_t) 27; i_144480++) {
            // futhark/microgpt.fut:199:5-245:83
            
            float lifted_lambda_res_141997 = ((float *) mem_145729)[i_144495 * (int64_t) 27 + i_144480];
            
            // futhark/microgpt.fut:103:13-33
            
            float max_res_140844 = fmax32(lifted_lambda_res_141997, redout_144479);
            float redout_tmp_147237 = max_res_140844;
            
            redout_144479 = redout_tmp_147237;
        }
        defunc_0_reduce_res_142047 = redout_144479;
        // futhark/microgpt.fut:113:47-56
        
        float neg_res_140845 = -defunc_0_reduce_res_142047;
        
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144483 = 0; i_144483 < (int64_t) 27; i_144483++) {
            // futhark/microgpt.fut:199:5-245:83
            
            float lifted_lambda_res_140852 = ((float *) mem_145729)[i_144495 * (int64_t) 27 + i_144483];
            
            // futhark/microgpt.fut:113:38-56
            
            float zp_res_140853 = neg_res_140845 + lifted_lambda_res_140852;
            
            // futhark/microgpt.fut:113:31-56
            
            float exp_res_140854 = futrts_exp32(zp_res_140853);
            
            ((float *) mem_145749)[i_144483] = exp_res_140854;
        }
        // futhark/microgpt.fut:71:13-49
        
        float defunc_0_lifted_lambda_res_140856;
        float r_140858 = 0.0F;
        
        for (int64_t i_140857 = 0; i_140857 < (int64_t) 27; i_140857++) {
            // futhark/microgpt.fut:114:32-39
            
            float lifted_lambda_res_140859 = ((float *) mem_145749)[i_140857];
            
            // futhark/microgpt.fut:71:40-49
            
            float zp_res_140860 = r_140858 + lifted_lambda_res_140859;
            float r_tmp_147239 = zp_res_140860;
            
            r_140858 = r_tmp_147239;
        }
        defunc_0_lifted_lambda_res_140856 = r_140858;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144487 = 0; i_144487 < (int64_t) 27; i_144487++) {
            // futhark/microgpt.fut:115:23-30
            
            float zs_lhs_140867 = ((float *) mem_145749)[i_144487];
            
            // futhark/microgpt.fut:115:23-40
            
            float zs_res_140868 = zs_lhs_140867 / defunc_0_lifted_lambda_res_140856;
            
            ((float *) mem_145756)[i_144487] = zs_res_140868;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144491 = 0; i_144491 < (int64_t) 27; i_144491++) {
            // futhark/microgpt.fut:243:4-14
            
            float log_arg0_140876 = ((float *) mem_145756)[i_144491];
            
            // futhark/microgpt.fut:242:75-243:14
            
            float log_res_140877 = futrts_log32(log_arg0_140876);
            
            ((float *) mem_145763)[i_144491] = log_res_140877;
        }
        // futhark/microgpt.fut:71:13-49
        
        float defunc_0_lifted_lambda_res_140879;
        float r_140881 = 0.0F;
        
        for (int64_t i_140880 = 0; i_140880 < (int64_t) 27; i_140880++) {
            // futhark/microgpt.fut:244:32-42
            
            float zt_lhs_140882 = ((float *) mem_145763)[i_140880];
            float zt_rhs_140883 = ((float *) ext_mem_145222.mem)[i_144495 * (int64_t) 27 + i_140880];
            
            // futhark/microgpt.fut:244:32-71
            
            float zt_res_140884 = zt_lhs_140882 * zt_rhs_140883;
            
            // futhark/microgpt.fut:71:40-49
            
            float zp_res_140885 = r_140881 + zt_res_140884;
            float r_tmp_147242 = zp_res_140885;
            
            r_140881 = r_tmp_147242;
        }
        defunc_0_lifted_lambda_res_140879 = r_140881;
        // futhark/microgpt.fut:244:5-73
        
        float neg_res_140886 = -defunc_0_lifted_lambda_res_140879;
        
        ((float *) mem_145745)[i_144495] = neg_res_140886;
    }
    if (memblock_unref(ctx, &ext_mem_145222, "ext_mem_145222") != 0)
        return 1;
    // futhark/microgpt.fut:71:13-49
    
    float defunc_0_lifted_lambda_res_140888;
    float r_140890 = 0.0F;
    
    for (int64_t i_140889 = 0; i_140889 < (int64_t) 16; i_140889++) {
        // futhark/microgpt.fut:199:5-245:83
        
        float lifted_lambda_res_140891 = ((float *) mem_145745)[i_140889];
        
        // futhark/microgpt.fut:71:40-49
        
        float zp_res_140892 = r_140890 + lifted_lambda_res_140891;
        float r_tmp_147243 = zp_res_140892;
        
        r_140890 = r_tmp_147243;
    }
    defunc_0_lifted_lambda_res_140888 = r_140890;
    // futhark/microgpt.fut:245:6-60
    
    float zs_res_140893 = defunc_0_lifted_lambda_res_140888 / 16.0F;
    
    prim_out_147157 = zs_res_140893;
    *out_prim_out_147561 = prim_out_147157;
    
  cleanup:
    {
        free(mem_145223);
        free(mem_145228);
        free(mem_145239);
        free(mem_145244);
        free(mem_145251);
        free(mem_145262);
        free(mem_145263);
        free(mem_145264);
        free(mem_145277);
        free(mem_145278);
        free(mem_145279);
        free(mem_145289);
        free(mem_145296);
        free(mem_145303);
        free(mem_145331);
        free(mem_145332);
        free(mem_145333);
        free(mem_145349);
        free(mem_145350);
        free(mem_145351);
        free(mem_145364);
        free(mem_145365);
        free(mem_145366);
        free(mem_145412);
        free(mem_145413);
        free(mem_145414);
        free(mem_145430);
        free(mem_145431);
        free(mem_145432);
        free(mem_145445);
        free(mem_145446);
        free(mem_145447);
        free(mem_145493);
        free(mem_145499);
        free(mem_145504);
        free(mem_145520);
        free(mem_145526);
        free(mem_145531);
        free(mem_145547);
        free(mem_145553);
        free(mem_145558);
        free(mem_145565);
        free(mem_145572);
        free(mem_145588);
        free(mem_145594);
        free(mem_145599);
        free(mem_145615);
        free(mem_145621);
        free(mem_145626);
        free(mem_145642);
        free(mem_145647);
        free(mem_145658);
        free(mem_145663);
        free(mem_145674);
        free(mem_145679);
        free(mem_145690);
        free(mem_145695);
        free(mem_145699);
        free(mem_145713);
        free(mem_145718);
        free(mem_145729);
        free(mem_145734);
        free(mem_145745);
        free(mem_145749);
        free(mem_145756);
        free(mem_145763);
        if (memblock_unref(ctx, &ext_mem_145222, "ext_mem_145222") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_to_params(struct futhark_context *ctx, struct memblock *mem_out_p_147628, struct memblock *mem_out_p_147629, struct memblock *mem_out_p_147630, struct memblock *mem_out_p_147631, struct memblock *mem_out_p_147632, struct memblock *mem_out_p_147633, struct memblock *mem_out_p_147634, struct memblock *mem_out_p_147635, struct memblock *mem_out_p_147636, struct memblock wte_mem_145211, struct memblock wpe_mem_145212, struct memblock wqry_mem_145213, struct memblock wkey_mem_145214, struct memblock wval_mem_145215, struct memblock wout_mem_145216, struct memblock wup_mem_145217, struct memblock wdown_mem_145218, struct memblock wvoc_mem_145219)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    if (memblock_set(ctx, &mem_out_147157, &wdown_mem_145218, "wdown_mem_145218") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147158, &wkey_mem_145214, "wkey_mem_145214") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147159, &wout_mem_145216, "wout_mem_145216") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147160, &wpe_mem_145212, "wpe_mem_145212") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147161, &wqry_mem_145213, "wqry_mem_145213") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147162, &wte_mem_145211, "wte_mem_145211") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147163, &wup_mem_145217, "wup_mem_145217") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147164, &wval_mem_145215, "wval_mem_145215") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147165, &wvoc_mem_145219, "wvoc_mem_145219") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147628, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147629, &mem_out_147158, "mem_out_147158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147630, &mem_out_147159, "mem_out_147159") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147631, &mem_out_147160, "mem_out_147160") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147632, &mem_out_147161, "mem_out_147161") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147633, &mem_out_147162, "mem_out_147162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147634, &mem_out_147163, "mem_out_147163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147635, &mem_out_147164, "mem_out_147164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147636, &mem_out_147165, "mem_out_147165") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_147165, "mem_out_147165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147164, "mem_out_147164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147163, "mem_out_147163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147162, "mem_out_147162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147161, "mem_out_147161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147160, "mem_out_147160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147159, "mem_out_147159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147158, "mem_out_147158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_train(struct futhark_context *ctx, struct memblock *mem_out_p_147637, struct memblock *mem_out_p_147638, struct memblock *mem_out_p_147639, struct memblock *mem_out_p_147640, struct memblock *mem_out_p_147641, struct memblock *mem_out_p_147642, struct memblock *mem_out_p_147643, struct memblock *mem_out_p_147644, struct memblock *mem_out_p_147645, struct memblock *mem_out_p_147646, struct memblock *mem_out_p_147647, struct memblock *mem_out_p_147648, struct memblock *mem_out_p_147649, struct memblock *mem_out_p_147650, struct memblock *mem_out_p_147651, struct memblock *mem_out_p_147652, struct memblock *mem_out_p_147653, struct memblock *mem_out_p_147654, struct memblock *mem_out_p_147655, struct memblock *mem_out_p_147656, struct memblock *mem_out_p_147657, struct memblock *mem_out_p_147658, struct memblock *mem_out_p_147659, struct memblock *mem_out_p_147660, struct memblock *mem_out_p_147661, struct memblock *mem_out_p_147662, struct memblock *mem_out_p_147663, struct memblock wdown_mem_145211, struct memblock wkey_mem_145212, struct memblock wout_mem_145213, struct memblock wpe_mem_145214, struct memblock wqry_mem_145215, struct memblock wte_mem_145216, struct memblock wup_mem_145217, struct memblock wval_mem_145218, struct memblock wvoc_mem_145219, struct memblock wdown_mem_145220, struct memblock wkey_mem_145221, struct memblock wout_mem_145222, struct memblock wpe_mem_145223, struct memblock wqry_mem_145224, struct memblock wte_mem_145225, struct memblock wup_mem_145226, struct memblock wval_mem_145227, struct memblock wvoc_mem_145228, struct memblock wdown_mem_145229, struct memblock wkey_mem_145230, struct memblock wout_mem_145231, struct memblock wpe_mem_145232, struct memblock wqry_mem_145233, struct memblock wte_mem_145234, struct memblock wup_mem_145235, struct memblock wval_mem_145236, struct memblock wvoc_mem_145237, struct memblock masks_mem_145238, struct memblock dls_mem_145239, struct memblock seqs_mem_145240)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_145352_cached_sizze_147664 = 0;
    unsigned char *mem_145352 = NULL;
    int64_t mem_145353_cached_sizze_147665 = 0;
    unsigned char *mem_145353 = NULL;
    int64_t mem_145362_cached_sizze_147666 = 0;
    unsigned char *mem_145362 = NULL;
    int64_t mem_145369_cached_sizze_147667 = 0;
    unsigned char *mem_145369 = NULL;
    int64_t mem_145384_cached_sizze_147668 = 0;
    unsigned char *mem_145384 = NULL;
    int64_t mem_145389_cached_sizze_147669 = 0;
    unsigned char *mem_145389 = NULL;
    int64_t mem_145400_cached_sizze_147670 = 0;
    unsigned char *mem_145400 = NULL;
    int64_t mem_145405_cached_sizze_147671 = 0;
    unsigned char *mem_145405 = NULL;
    int64_t mem_145416_cached_sizze_147672 = 0;
    unsigned char *mem_145416 = NULL;
    int64_t mem_145423_cached_sizze_147673 = 0;
    unsigned char *mem_145423 = NULL;
    int64_t mem_145430_cached_sizze_147674 = 0;
    unsigned char *mem_145430 = NULL;
    int64_t mem_145435_cached_sizze_147675 = 0;
    unsigned char *mem_145435 = NULL;
    int64_t mem_145446_cached_sizze_147676 = 0;
    unsigned char *mem_145446 = NULL;
    int64_t mem_145451_cached_sizze_147677 = 0;
    unsigned char *mem_145451 = NULL;
    int64_t mem_145462_cached_sizze_147678 = 0;
    unsigned char *mem_145462 = NULL;
    int64_t mem_145469_cached_sizze_147679 = 0;
    unsigned char *mem_145469 = NULL;
    int64_t mem_145476_cached_sizze_147680 = 0;
    unsigned char *mem_145476 = NULL;
    int64_t mem_145481_cached_sizze_147681 = 0;
    unsigned char *mem_145481 = NULL;
    int64_t mem_145492_cached_sizze_147682 = 0;
    unsigned char *mem_145492 = NULL;
    int64_t mem_145493_cached_sizze_147683 = 0;
    unsigned char *mem_145493 = NULL;
    int64_t mem_145494_cached_sizze_147684 = 0;
    unsigned char *mem_145494 = NULL;
    int64_t mem_145507_cached_sizze_147685 = 0;
    unsigned char *mem_145507 = NULL;
    int64_t mem_145508_cached_sizze_147686 = 0;
    unsigned char *mem_145508 = NULL;
    int64_t mem_145509_cached_sizze_147687 = 0;
    unsigned char *mem_145509 = NULL;
    int64_t mem_145540_cached_sizze_147688 = 0;
    unsigned char *mem_145540 = NULL;
    int64_t mem_145541_cached_sizze_147689 = 0;
    unsigned char *mem_145541 = NULL;
    int64_t mem_145542_cached_sizze_147690 = 0;
    unsigned char *mem_145542 = NULL;
    int64_t mem_145558_cached_sizze_147691 = 0;
    unsigned char *mem_145558 = NULL;
    int64_t mem_145559_cached_sizze_147692 = 0;
    unsigned char *mem_145559 = NULL;
    int64_t mem_145560_cached_sizze_147693 = 0;
    unsigned char *mem_145560 = NULL;
    int64_t mem_145573_cached_sizze_147694 = 0;
    unsigned char *mem_145573 = NULL;
    int64_t mem_145574_cached_sizze_147695 = 0;
    unsigned char *mem_145574 = NULL;
    int64_t mem_145575_cached_sizze_147696 = 0;
    unsigned char *mem_145575 = NULL;
    int64_t mem_145621_cached_sizze_147697 = 0;
    unsigned char *mem_145621 = NULL;
    int64_t mem_145622_cached_sizze_147698 = 0;
    unsigned char *mem_145622 = NULL;
    int64_t mem_145623_cached_sizze_147699 = 0;
    unsigned char *mem_145623 = NULL;
    int64_t mem_145639_cached_sizze_147700 = 0;
    unsigned char *mem_145639 = NULL;
    int64_t mem_145640_cached_sizze_147701 = 0;
    unsigned char *mem_145640 = NULL;
    int64_t mem_145641_cached_sizze_147702 = 0;
    unsigned char *mem_145641 = NULL;
    int64_t mem_145654_cached_sizze_147703 = 0;
    unsigned char *mem_145654 = NULL;
    int64_t mem_145655_cached_sizze_147704 = 0;
    unsigned char *mem_145655 = NULL;
    int64_t mem_145656_cached_sizze_147705 = 0;
    unsigned char *mem_145656 = NULL;
    int64_t mem_145702_cached_sizze_147706 = 0;
    unsigned char *mem_145702 = NULL;
    int64_t mem_145708_cached_sizze_147707 = 0;
    unsigned char *mem_145708 = NULL;
    int64_t mem_145713_cached_sizze_147708 = 0;
    unsigned char *mem_145713 = NULL;
    int64_t mem_145729_cached_sizze_147709 = 0;
    unsigned char *mem_145729 = NULL;
    int64_t mem_145735_cached_sizze_147710 = 0;
    unsigned char *mem_145735 = NULL;
    int64_t mem_145740_cached_sizze_147711 = 0;
    unsigned char *mem_145740 = NULL;
    int64_t mem_145756_cached_sizze_147712 = 0;
    unsigned char *mem_145756 = NULL;
    int64_t mem_145762_cached_sizze_147713 = 0;
    unsigned char *mem_145762 = NULL;
    int64_t mem_145767_cached_sizze_147714 = 0;
    unsigned char *mem_145767 = NULL;
    int64_t mem_145774_cached_sizze_147715 = 0;
    unsigned char *mem_145774 = NULL;
    int64_t mem_145781_cached_sizze_147716 = 0;
    unsigned char *mem_145781 = NULL;
    int64_t mem_145797_cached_sizze_147717 = 0;
    unsigned char *mem_145797 = NULL;
    int64_t mem_145803_cached_sizze_147718 = 0;
    unsigned char *mem_145803 = NULL;
    int64_t mem_145808_cached_sizze_147719 = 0;
    unsigned char *mem_145808 = NULL;
    int64_t mem_145824_cached_sizze_147720 = 0;
    unsigned char *mem_145824 = NULL;
    int64_t mem_145830_cached_sizze_147721 = 0;
    unsigned char *mem_145830 = NULL;
    int64_t mem_145835_cached_sizze_147722 = 0;
    unsigned char *mem_145835 = NULL;
    int64_t mem_145851_cached_sizze_147723 = 0;
    unsigned char *mem_145851 = NULL;
    int64_t mem_145856_cached_sizze_147724 = 0;
    unsigned char *mem_145856 = NULL;
    int64_t mem_145867_cached_sizze_147725 = 0;
    unsigned char *mem_145867 = NULL;
    int64_t mem_145872_cached_sizze_147726 = 0;
    unsigned char *mem_145872 = NULL;
    int64_t mem_145883_cached_sizze_147727 = 0;
    unsigned char *mem_145883 = NULL;
    int64_t mem_145888_cached_sizze_147728 = 0;
    unsigned char *mem_145888 = NULL;
    int64_t mem_145899_cached_sizze_147729 = 0;
    unsigned char *mem_145899 = NULL;
    int64_t mem_145904_cached_sizze_147730 = 0;
    unsigned char *mem_145904 = NULL;
    int64_t mem_145915_cached_sizze_147731 = 0;
    unsigned char *mem_145915 = NULL;
    int64_t mem_145922_cached_sizze_147732 = 0;
    unsigned char *mem_145922 = NULL;
    int64_t mem_145929_cached_sizze_147733 = 0;
    unsigned char *mem_145929 = NULL;
    int64_t mem_145934_cached_sizze_147734 = 0;
    unsigned char *mem_145934 = NULL;
    int64_t mem_145945_cached_sizze_147735 = 0;
    unsigned char *mem_145945 = NULL;
    int64_t mem_145950_cached_sizze_147736 = 0;
    unsigned char *mem_145950 = NULL;
    int64_t mem_145961_cached_sizze_147737 = 0;
    unsigned char *mem_145961 = NULL;
    int64_t mem_145966_cached_sizze_147738 = 0;
    unsigned char *mem_145966 = NULL;
    int64_t mem_145977_cached_sizze_147739 = 0;
    unsigned char *mem_145977 = NULL;
    int64_t mem_145982_cached_sizze_147740 = 0;
    unsigned char *mem_145982 = NULL;
    int64_t mem_145993_cached_sizze_147741 = 0;
    unsigned char *mem_145993 = NULL;
    int64_t mem_145998_cached_sizze_147742 = 0;
    unsigned char *mem_145998 = NULL;
    int64_t mem_146009_cached_sizze_147743 = 0;
    unsigned char *mem_146009 = NULL;
    int64_t mem_146014_cached_sizze_147744 = 0;
    unsigned char *mem_146014 = NULL;
    int64_t mem_146025_cached_sizze_147745 = 0;
    unsigned char *mem_146025 = NULL;
    int64_t mem_146026_cached_sizze_147746 = 0;
    unsigned char *mem_146026 = NULL;
    int64_t mem_146035_cached_sizze_147747 = 0;
    unsigned char *mem_146035 = NULL;
    int64_t mem_146036_cached_sizze_147748 = 0;
    unsigned char *mem_146036 = NULL;
    int64_t mem_146049_cached_sizze_147749 = 0;
    unsigned char *mem_146049 = NULL;
    int64_t mem_146050_cached_sizze_147750 = 0;
    unsigned char *mem_146050 = NULL;
    int64_t mem_146063_cached_sizze_147751 = 0;
    unsigned char *mem_146063 = NULL;
    int64_t mem_146064_cached_sizze_147752 = 0;
    unsigned char *mem_146064 = NULL;
    int64_t mem_146085_cached_sizze_147753 = 0;
    unsigned char *mem_146085 = NULL;
    int64_t mem_146092_cached_sizze_147754 = 0;
    unsigned char *mem_146092 = NULL;
    int64_t mem_146097_cached_sizze_147755 = 0;
    unsigned char *mem_146097 = NULL;
    int64_t mem_146108_cached_sizze_147756 = 0;
    unsigned char *mem_146108 = NULL;
    int64_t mem_146113_cached_sizze_147757 = 0;
    unsigned char *mem_146113 = NULL;
    int64_t mem_146124_cached_sizze_147758 = 0;
    unsigned char *mem_146124 = NULL;
    int64_t mem_146125_cached_sizze_147759 = 0;
    unsigned char *mem_146125 = NULL;
    int64_t mem_146134_cached_sizze_147760 = 0;
    unsigned char *mem_146134 = NULL;
    int64_t mem_146135_cached_sizze_147761 = 0;
    unsigned char *mem_146135 = NULL;
    int64_t mem_146156_cached_sizze_147762 = 0;
    unsigned char *mem_146156 = NULL;
    int64_t mem_146161_cached_sizze_147763 = 0;
    unsigned char *mem_146161 = NULL;
    int64_t mem_146172_cached_sizze_147764 = 0;
    unsigned char *mem_146172 = NULL;
    int64_t mem_146177_cached_sizze_147765 = 0;
    unsigned char *mem_146177 = NULL;
    int64_t mem_146188_cached_sizze_147766 = 0;
    unsigned char *mem_146188 = NULL;
    int64_t mem_146195_cached_sizze_147767 = 0;
    unsigned char *mem_146195 = NULL;
    int64_t mem_146202_cached_sizze_147768 = 0;
    unsigned char *mem_146202 = NULL;
    int64_t mem_146212_cached_sizze_147769 = 0;
    unsigned char *mem_146212 = NULL;
    int64_t mem_146217_cached_sizze_147770 = 0;
    unsigned char *mem_146217 = NULL;
    int64_t mem_146228_cached_sizze_147771 = 0;
    unsigned char *mem_146228 = NULL;
    int64_t mem_146229_cached_sizze_147772 = 0;
    unsigned char *mem_146229 = NULL;
    int64_t mem_146238_cached_sizze_147773 = 0;
    unsigned char *mem_146238 = NULL;
    int64_t mem_146239_cached_sizze_147774 = 0;
    unsigned char *mem_146239 = NULL;
    int64_t mem_146260_cached_sizze_147775 = 0;
    unsigned char *mem_146260 = NULL;
    int64_t mem_146266_cached_sizze_147776 = 0;
    unsigned char *mem_146266 = NULL;
    int64_t mem_146271_cached_sizze_147777 = 0;
    unsigned char *mem_146271 = NULL;
    int64_t mem_146287_cached_sizze_147778 = 0;
    unsigned char *mem_146287 = NULL;
    int64_t mem_146293_cached_sizze_147779 = 0;
    unsigned char *mem_146293 = NULL;
    int64_t mem_146298_cached_sizze_147780 = 0;
    unsigned char *mem_146298 = NULL;
    int64_t mem_146314_cached_sizze_147781 = 0;
    unsigned char *mem_146314 = NULL;
    int64_t mem_146315_cached_sizze_147782 = 0;
    unsigned char *mem_146315 = NULL;
    int64_t mem_146326_cached_sizze_147783 = 0;
    unsigned char *mem_146326 = NULL;
    int64_t mem_146327_cached_sizze_147784 = 0;
    unsigned char *mem_146327 = NULL;
    int64_t mem_146336_cached_sizze_147785 = 0;
    unsigned char *mem_146336 = NULL;
    int64_t mem_146343_cached_sizze_147786 = 0;
    unsigned char *mem_146343 = NULL;
    int64_t mem_146368_cached_sizze_147787 = 0;
    unsigned char *mem_146368 = NULL;
    int64_t mem_146374_cached_sizze_147788 = 0;
    unsigned char *mem_146374 = NULL;
    int64_t mem_146379_cached_sizze_147789 = 0;
    unsigned char *mem_146379 = NULL;
    int64_t mem_146395_cached_sizze_147790 = 0;
    unsigned char *mem_146395 = NULL;
    int64_t mem_146400_cached_sizze_147791 = 0;
    unsigned char *mem_146400 = NULL;
    int64_t mem_146411_cached_sizze_147792 = 0;
    unsigned char *mem_146411 = NULL;
    int64_t mem_146417_cached_sizze_147793 = 0;
    unsigned char *mem_146417 = NULL;
    int64_t mem_146422_cached_sizze_147794 = 0;
    unsigned char *mem_146422 = NULL;
    int64_t mem_146438_cached_sizze_147795 = 0;
    unsigned char *mem_146438 = NULL;
    int64_t mem_146444_cached_sizze_147796 = 0;
    unsigned char *mem_146444 = NULL;
    int64_t mem_146449_cached_sizze_147797 = 0;
    unsigned char *mem_146449 = NULL;
    int64_t mem_146465_cached_sizze_147798 = 0;
    unsigned char *mem_146465 = NULL;
    int64_t mem_146466_cached_sizze_147799 = 0;
    unsigned char *mem_146466 = NULL;
    int64_t mem_146477_cached_sizze_147800 = 0;
    unsigned char *mem_146477 = NULL;
    int64_t mem_146478_cached_sizze_147801 = 0;
    unsigned char *mem_146478 = NULL;
    int64_t mem_146487_cached_sizze_147802 = 0;
    unsigned char *mem_146487 = NULL;
    int64_t mem_146488_cached_sizze_147803 = 0;
    unsigned char *mem_146488 = NULL;
    int64_t mem_146519_cached_sizze_147804 = 0;
    unsigned char *mem_146519 = NULL;
    int64_t mem_146520_cached_sizze_147805 = 0;
    unsigned char *mem_146520 = NULL;
    int64_t mem_146521_cached_sizze_147806 = 0;
    unsigned char *mem_146521 = NULL;
    int64_t mem_146537_cached_sizze_147807 = 0;
    unsigned char *mem_146537 = NULL;
    int64_t mem_146538_cached_sizze_147808 = 0;
    unsigned char *mem_146538 = NULL;
    int64_t mem_146539_cached_sizze_147809 = 0;
    unsigned char *mem_146539 = NULL;
    int64_t mem_146552_cached_sizze_147810 = 0;
    unsigned char *mem_146552 = NULL;
    int64_t mem_146553_cached_sizze_147811 = 0;
    unsigned char *mem_146553 = NULL;
    int64_t mem_146554_cached_sizze_147812 = 0;
    unsigned char *mem_146554 = NULL;
    int64_t mem_146600_cached_sizze_147813 = 0;
    unsigned char *mem_146600 = NULL;
    int64_t mem_146601_cached_sizze_147814 = 0;
    unsigned char *mem_146601 = NULL;
    int64_t mem_146602_cached_sizze_147815 = 0;
    unsigned char *mem_146602 = NULL;
    int64_t mem_146615_cached_sizze_147816 = 0;
    unsigned char *mem_146615 = NULL;
    int64_t mem_146616_cached_sizze_147817 = 0;
    unsigned char *mem_146616 = NULL;
    int64_t mem_146617_cached_sizze_147818 = 0;
    unsigned char *mem_146617 = NULL;
    int64_t mem_146648_cached_sizze_147819 = 0;
    unsigned char *mem_146648 = NULL;
    int64_t mem_146649_cached_sizze_147820 = 0;
    unsigned char *mem_146649 = NULL;
    int64_t mem_146650_cached_sizze_147821 = 0;
    unsigned char *mem_146650 = NULL;
    int64_t mem_146651_cached_sizze_147822 = 0;
    unsigned char *mem_146651 = NULL;
    int64_t mem_146668_cached_sizze_147823 = 0;
    unsigned char *mem_146668 = NULL;
    int64_t mem_146669_cached_sizze_147824 = 0;
    unsigned char *mem_146669 = NULL;
    int64_t mem_146670_cached_sizze_147825 = 0;
    unsigned char *mem_146670 = NULL;
    int64_t mem_146671_cached_sizze_147826 = 0;
    unsigned char *mem_146671 = NULL;
    int64_t mem_146712_cached_sizze_147827 = 0;
    unsigned char *mem_146712 = NULL;
    int64_t mem_146719_cached_sizze_147828 = 0;
    unsigned char *mem_146719 = NULL;
    int64_t mem_146726_cached_sizze_147829 = 0;
    unsigned char *mem_146726 = NULL;
    int64_t mem_146736_cached_sizze_147830 = 0;
    unsigned char *mem_146736 = NULL;
    int64_t mem_146741_cached_sizze_147831 = 0;
    unsigned char *mem_146741 = NULL;
    int64_t mem_146752_cached_sizze_147832 = 0;
    unsigned char *mem_146752 = NULL;
    int64_t mem_146759_cached_sizze_147833 = 0;
    unsigned char *mem_146759 = NULL;
    int64_t mem_146766_cached_sizze_147834 = 0;
    unsigned char *mem_146766 = NULL;
    int64_t mem_146776_cached_sizze_147835 = 0;
    unsigned char *mem_146776 = NULL;
    int64_t mem_146781_cached_sizze_147836 = 0;
    unsigned char *mem_146781 = NULL;
    int64_t mem_146792_cached_sizze_147837 = 0;
    unsigned char *mem_146792 = NULL;
    int64_t mem_146793_cached_sizze_147838 = 0;
    unsigned char *mem_146793 = NULL;
    int64_t mem_146802_cached_sizze_147839 = 0;
    unsigned char *mem_146802 = NULL;
    int64_t mem_146803_cached_sizze_147840 = 0;
    unsigned char *mem_146803 = NULL;
    int64_t mem_146824_cached_sizze_147841 = 0;
    unsigned char *mem_146824 = NULL;
    int64_t mem_146829_cached_sizze_147842 = 0;
    unsigned char *mem_146829 = NULL;
    int64_t mem_146840_cached_sizze_147843 = 0;
    unsigned char *mem_146840 = NULL;
    int64_t mem_146841_cached_sizze_147844 = 0;
    unsigned char *mem_146841 = NULL;
    int64_t mem_146850_cached_sizze_147845 = 0;
    unsigned char *mem_146850 = NULL;
    int64_t mem_146851_cached_sizze_147846 = 0;
    unsigned char *mem_146851 = NULL;
    struct memblock mem_param_tmp_147210;
    
    mem_param_tmp_147210.references = NULL;
    
    struct memblock mem_param_tmp_147209;
    
    mem_param_tmp_147209.references = NULL;
    
    struct memblock mem_param_tmp_147208;
    
    mem_param_tmp_147208.references = NULL;
    
    struct memblock mem_param_tmp_147207;
    
    mem_param_tmp_147207.references = NULL;
    
    struct memblock mem_param_tmp_147206;
    
    mem_param_tmp_147206.references = NULL;
    
    struct memblock mem_param_tmp_147205;
    
    mem_param_tmp_147205.references = NULL;
    
    struct memblock mem_param_tmp_147204;
    
    mem_param_tmp_147204.references = NULL;
    
    struct memblock mem_param_tmp_147203;
    
    mem_param_tmp_147203.references = NULL;
    
    struct memblock mem_param_tmp_147202;
    
    mem_param_tmp_147202.references = NULL;
    
    struct memblock mem_param_tmp_147201;
    
    mem_param_tmp_147201.references = NULL;
    
    struct memblock mem_param_tmp_147200;
    
    mem_param_tmp_147200.references = NULL;
    
    struct memblock mem_param_tmp_147199;
    
    mem_param_tmp_147199.references = NULL;
    
    struct memblock mem_param_tmp_147198;
    
    mem_param_tmp_147198.references = NULL;
    
    struct memblock mem_param_tmp_147197;
    
    mem_param_tmp_147197.references = NULL;
    
    struct memblock mem_param_tmp_147196;
    
    mem_param_tmp_147196.references = NULL;
    
    struct memblock mem_param_tmp_147195;
    
    mem_param_tmp_147195.references = NULL;
    
    struct memblock mem_param_tmp_147194;
    
    mem_param_tmp_147194.references = NULL;
    
    struct memblock mem_param_tmp_147193;
    
    mem_param_tmp_147193.references = NULL;
    
    struct memblock mem_param_tmp_147192;
    
    mem_param_tmp_147192.references = NULL;
    
    struct memblock mem_param_tmp_147191;
    
    mem_param_tmp_147191.references = NULL;
    
    struct memblock mem_param_tmp_147190;
    
    mem_param_tmp_147190.references = NULL;
    
    struct memblock mem_param_tmp_147189;
    
    mem_param_tmp_147189.references = NULL;
    
    struct memblock mem_param_tmp_147188;
    
    mem_param_tmp_147188.references = NULL;
    
    struct memblock mem_param_tmp_147187;
    
    mem_param_tmp_147187.references = NULL;
    
    struct memblock mem_param_tmp_147186;
    
    mem_param_tmp_147186.references = NULL;
    
    struct memblock mem_param_tmp_147185;
    
    mem_param_tmp_147185.references = NULL;
    
    struct memblock mem_param_tmp_147184;
    
    mem_param_tmp_147184.references = NULL;
    
    struct memblock ext_mem_146968;
    
    ext_mem_146968.references = NULL;
    
    struct memblock ext_mem_146969;
    
    ext_mem_146969.references = NULL;
    
    struct memblock ext_mem_146970;
    
    ext_mem_146970.references = NULL;
    
    struct memblock mem_146966;
    
    mem_146966.references = NULL;
    
    struct memblock mem_146964;
    
    mem_146964.references = NULL;
    
    struct memblock mem_146962;
    
    mem_146962.references = NULL;
    
    struct memblock mem_146960;
    
    mem_146960.references = NULL;
    
    struct memblock ext_mem_146957;
    
    ext_mem_146957.references = NULL;
    
    struct memblock ext_mem_146958;
    
    ext_mem_146958.references = NULL;
    
    struct memblock ext_mem_146959;
    
    ext_mem_146959.references = NULL;
    
    struct memblock mem_146955;
    
    mem_146955.references = NULL;
    
    struct memblock mem_146953;
    
    mem_146953.references = NULL;
    
    struct memblock mem_146951;
    
    mem_146951.references = NULL;
    
    struct memblock mem_146949;
    
    mem_146949.references = NULL;
    
    struct memblock ext_mem_146946;
    
    ext_mem_146946.references = NULL;
    
    struct memblock ext_mem_146947;
    
    ext_mem_146947.references = NULL;
    
    struct memblock ext_mem_146948;
    
    ext_mem_146948.references = NULL;
    
    struct memblock mem_146944;
    
    mem_146944.references = NULL;
    
    struct memblock mem_146942;
    
    mem_146942.references = NULL;
    
    struct memblock mem_146940;
    
    mem_146940.references = NULL;
    
    struct memblock mem_146938;
    
    mem_146938.references = NULL;
    
    struct memblock ext_mem_146935;
    
    ext_mem_146935.references = NULL;
    
    struct memblock ext_mem_146936;
    
    ext_mem_146936.references = NULL;
    
    struct memblock ext_mem_146937;
    
    ext_mem_146937.references = NULL;
    
    struct memblock mem_146933;
    
    mem_146933.references = NULL;
    
    struct memblock mem_146931;
    
    mem_146931.references = NULL;
    
    struct memblock mem_146929;
    
    mem_146929.references = NULL;
    
    struct memblock mem_146927;
    
    mem_146927.references = NULL;
    
    struct memblock ext_mem_146924;
    
    ext_mem_146924.references = NULL;
    
    struct memblock ext_mem_146925;
    
    ext_mem_146925.references = NULL;
    
    struct memblock ext_mem_146926;
    
    ext_mem_146926.references = NULL;
    
    struct memblock mem_146922;
    
    mem_146922.references = NULL;
    
    struct memblock mem_146920;
    
    mem_146920.references = NULL;
    
    struct memblock mem_146918;
    
    mem_146918.references = NULL;
    
    struct memblock mem_146916;
    
    mem_146916.references = NULL;
    
    struct memblock ext_mem_146913;
    
    ext_mem_146913.references = NULL;
    
    struct memblock ext_mem_146914;
    
    ext_mem_146914.references = NULL;
    
    struct memblock ext_mem_146915;
    
    ext_mem_146915.references = NULL;
    
    struct memblock mem_146911;
    
    mem_146911.references = NULL;
    
    struct memblock mem_146909;
    
    mem_146909.references = NULL;
    
    struct memblock mem_146907;
    
    mem_146907.references = NULL;
    
    struct memblock mem_146905;
    
    mem_146905.references = NULL;
    
    struct memblock ext_mem_146902;
    
    ext_mem_146902.references = NULL;
    
    struct memblock ext_mem_146903;
    
    ext_mem_146903.references = NULL;
    
    struct memblock ext_mem_146904;
    
    ext_mem_146904.references = NULL;
    
    struct memblock mem_146900;
    
    mem_146900.references = NULL;
    
    struct memblock mem_146898;
    
    mem_146898.references = NULL;
    
    struct memblock mem_146896;
    
    mem_146896.references = NULL;
    
    struct memblock mem_146894;
    
    mem_146894.references = NULL;
    
    struct memblock ext_mem_146891;
    
    ext_mem_146891.references = NULL;
    
    struct memblock ext_mem_146892;
    
    ext_mem_146892.references = NULL;
    
    struct memblock ext_mem_146893;
    
    ext_mem_146893.references = NULL;
    
    struct memblock mem_146889;
    
    mem_146889.references = NULL;
    
    struct memblock mem_146887;
    
    mem_146887.references = NULL;
    
    struct memblock mem_146885;
    
    mem_146885.references = NULL;
    
    struct memblock mem_146883;
    
    mem_146883.references = NULL;
    
    struct memblock ext_mem_146880;
    
    ext_mem_146880.references = NULL;
    
    struct memblock ext_mem_146881;
    
    ext_mem_146881.references = NULL;
    
    struct memblock ext_mem_146882;
    
    ext_mem_146882.references = NULL;
    
    struct memblock mem_146878;
    
    mem_146878.references = NULL;
    
    struct memblock mem_146876;
    
    mem_146876.references = NULL;
    
    struct memblock mem_146874;
    
    mem_146874.references = NULL;
    
    struct memblock mem_146872;
    
    mem_146872.references = NULL;
    
    struct memblock ext_mem_145351;
    
    ext_mem_145351.references = NULL;
    
    struct memblock mem_param_145348;
    
    mem_param_145348.references = NULL;
    
    struct memblock mem_param_145344;
    
    mem_param_145344.references = NULL;
    
    struct memblock mem_param_145340;
    
    mem_param_145340.references = NULL;
    
    struct memblock mem_param_145336;
    
    mem_param_145336.references = NULL;
    
    struct memblock mem_param_145332;
    
    mem_param_145332.references = NULL;
    
    struct memblock mem_param_145328;
    
    mem_param_145328.references = NULL;
    
    struct memblock mem_param_145324;
    
    mem_param_145324.references = NULL;
    
    struct memblock mem_param_145320;
    
    mem_param_145320.references = NULL;
    
    struct memblock mem_param_145316;
    
    mem_param_145316.references = NULL;
    
    struct memblock mem_param_145312;
    
    mem_param_145312.references = NULL;
    
    struct memblock mem_param_145308;
    
    mem_param_145308.references = NULL;
    
    struct memblock mem_param_145304;
    
    mem_param_145304.references = NULL;
    
    struct memblock mem_param_145300;
    
    mem_param_145300.references = NULL;
    
    struct memblock mem_param_145296;
    
    mem_param_145296.references = NULL;
    
    struct memblock mem_param_145292;
    
    mem_param_145292.references = NULL;
    
    struct memblock mem_param_145288;
    
    mem_param_145288.references = NULL;
    
    struct memblock mem_param_145284;
    
    mem_param_145284.references = NULL;
    
    struct memblock mem_param_145280;
    
    mem_param_145280.references = NULL;
    
    struct memblock mem_param_145276;
    
    mem_param_145276.references = NULL;
    
    struct memblock mem_param_145272;
    
    mem_param_145272.references = NULL;
    
    struct memblock mem_param_145268;
    
    mem_param_145268.references = NULL;
    
    struct memblock mem_param_145264;
    
    mem_param_145264.references = NULL;
    
    struct memblock mem_param_145260;
    
    mem_param_145260.references = NULL;
    
    struct memblock mem_param_145256;
    
    mem_param_145256.references = NULL;
    
    struct memblock mem_param_145252;
    
    mem_param_145252.references = NULL;
    
    struct memblock mem_param_145248;
    
    mem_param_145248.references = NULL;
    
    struct memblock mem_param_145244;
    
    mem_param_145244.references = NULL;
    
    struct memblock ext_mem_147052;
    
    ext_mem_147052.references = NULL;
    
    struct memblock ext_mem_147053;
    
    ext_mem_147053.references = NULL;
    
    struct memblock ext_mem_147054;
    
    ext_mem_147054.references = NULL;
    
    struct memblock ext_mem_147055;
    
    ext_mem_147055.references = NULL;
    
    struct memblock ext_mem_147056;
    
    ext_mem_147056.references = NULL;
    
    struct memblock ext_mem_147057;
    
    ext_mem_147057.references = NULL;
    
    struct memblock ext_mem_147058;
    
    ext_mem_147058.references = NULL;
    
    struct memblock ext_mem_147059;
    
    ext_mem_147059.references = NULL;
    
    struct memblock ext_mem_147060;
    
    ext_mem_147060.references = NULL;
    
    struct memblock ext_mem_147061;
    
    ext_mem_147061.references = NULL;
    
    struct memblock ext_mem_147062;
    
    ext_mem_147062.references = NULL;
    
    struct memblock ext_mem_147063;
    
    ext_mem_147063.references = NULL;
    
    struct memblock ext_mem_147064;
    
    ext_mem_147064.references = NULL;
    
    struct memblock ext_mem_147065;
    
    ext_mem_147065.references = NULL;
    
    struct memblock ext_mem_147066;
    
    ext_mem_147066.references = NULL;
    
    struct memblock ext_mem_147067;
    
    ext_mem_147067.references = NULL;
    
    struct memblock ext_mem_147068;
    
    ext_mem_147068.references = NULL;
    
    struct memblock ext_mem_147069;
    
    ext_mem_147069.references = NULL;
    
    struct memblock ext_mem_147070;
    
    ext_mem_147070.references = NULL;
    
    struct memblock ext_mem_147071;
    
    ext_mem_147071.references = NULL;
    
    struct memblock ext_mem_147072;
    
    ext_mem_147072.references = NULL;
    
    struct memblock ext_mem_147073;
    
    ext_mem_147073.references = NULL;
    
    struct memblock ext_mem_147074;
    
    ext_mem_147074.references = NULL;
    
    struct memblock ext_mem_147075;
    
    ext_mem_147075.references = NULL;
    
    struct memblock ext_mem_147076;
    
    ext_mem_147076.references = NULL;
    
    struct memblock ext_mem_147077;
    
    ext_mem_147077.references = NULL;
    
    struct memblock ext_mem_147078;
    
    ext_mem_147078.references = NULL;
    
    struct memblock mem_145349;
    
    mem_145349.references = NULL;
    
    struct memblock mem_out_147183;
    
    mem_out_147183.references = NULL;
    
    struct memblock mem_out_147182;
    
    mem_out_147182.references = NULL;
    
    struct memblock mem_out_147181;
    
    mem_out_147181.references = NULL;
    
    struct memblock mem_out_147180;
    
    mem_out_147180.references = NULL;
    
    struct memblock mem_out_147179;
    
    mem_out_147179.references = NULL;
    
    struct memblock mem_out_147178;
    
    mem_out_147178.references = NULL;
    
    struct memblock mem_out_147177;
    
    mem_out_147177.references = NULL;
    
    struct memblock mem_out_147176;
    
    mem_out_147176.references = NULL;
    
    struct memblock mem_out_147175;
    
    mem_out_147175.references = NULL;
    
    struct memblock mem_out_147174;
    
    mem_out_147174.references = NULL;
    
    struct memblock mem_out_147173;
    
    mem_out_147173.references = NULL;
    
    struct memblock mem_out_147172;
    
    mem_out_147172.references = NULL;
    
    struct memblock mem_out_147171;
    
    mem_out_147171.references = NULL;
    
    struct memblock mem_out_147170;
    
    mem_out_147170.references = NULL;
    
    struct memblock mem_out_147169;
    
    mem_out_147169.references = NULL;
    
    struct memblock mem_out_147168;
    
    mem_out_147168.references = NULL;
    
    struct memblock mem_out_147167;
    
    mem_out_147167.references = NULL;
    
    struct memblock mem_out_147166;
    
    mem_out_147166.references = NULL;
    
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    // futhark/microgpt.fut:482:32-53
    if (memblock_alloc(ctx, &mem_145349, (int64_t) 128, "mem_145349")) {
        err = 1;
        goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145352_cached_sizze_147664 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_145352, &mem_145352_cached_sizze_147664, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145353_cached_sizze_147665 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145353, &mem_145353_cached_sizze_147665, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145362_cached_sizze_147666 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145362, &mem_145362_cached_sizze_147666, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145369_cached_sizze_147667 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_145369, &mem_145369_cached_sizze_147667, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145384_cached_sizze_147668 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145384, &mem_145384_cached_sizze_147668, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145389_cached_sizze_147669 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145389, &mem_145389_cached_sizze_147669, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145400_cached_sizze_147670 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145400, &mem_145400_cached_sizze_147670, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145405_cached_sizze_147671 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145405, &mem_145405_cached_sizze_147671, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145416_cached_sizze_147672 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145416, &mem_145416_cached_sizze_147672, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145423_cached_sizze_147673 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145423, &mem_145423_cached_sizze_147673, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145430_cached_sizze_147674 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145430, &mem_145430_cached_sizze_147674, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145435_cached_sizze_147675 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145435, &mem_145435_cached_sizze_147675, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145446_cached_sizze_147676 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145446, &mem_145446_cached_sizze_147676, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145451_cached_sizze_147677 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145451, &mem_145451_cached_sizze_147677, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145462_cached_sizze_147678 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145462, &mem_145462_cached_sizze_147678, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145469_cached_sizze_147679 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145469, &mem_145469_cached_sizze_147679, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145476_cached_sizze_147680 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145476, &mem_145476_cached_sizze_147680, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145481_cached_sizze_147681 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145481, &mem_145481_cached_sizze_147681, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145492_cached_sizze_147682 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145492, &mem_145492_cached_sizze_147682, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145493_cached_sizze_147683 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145493, &mem_145493_cached_sizze_147683, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145494_cached_sizze_147684 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145494, &mem_145494_cached_sizze_147684, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145507_cached_sizze_147685 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145507, &mem_145507_cached_sizze_147685, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145508_cached_sizze_147686 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145508, &mem_145508_cached_sizze_147686, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145509_cached_sizze_147687 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145509, &mem_145509_cached_sizze_147687, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145540_cached_sizze_147688 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145540, &mem_145540_cached_sizze_147688, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145541_cached_sizze_147689 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145541, &mem_145541_cached_sizze_147689, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145542_cached_sizze_147690 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145542, &mem_145542_cached_sizze_147690, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145558_cached_sizze_147691 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145558, &mem_145558_cached_sizze_147691, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145559_cached_sizze_147692 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145559, &mem_145559_cached_sizze_147692, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145560_cached_sizze_147693 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145560, &mem_145560_cached_sizze_147693, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145573_cached_sizze_147694 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145573, &mem_145573_cached_sizze_147694, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145574_cached_sizze_147695 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145574, &mem_145574_cached_sizze_147695, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145575_cached_sizze_147696 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145575, &mem_145575_cached_sizze_147696, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145621_cached_sizze_147697 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145621, &mem_145621_cached_sizze_147697, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145622_cached_sizze_147698 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145622, &mem_145622_cached_sizze_147698, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145623_cached_sizze_147699 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145623, &mem_145623_cached_sizze_147699, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145639_cached_sizze_147700 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145639, &mem_145639_cached_sizze_147700, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145640_cached_sizze_147701 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145640, &mem_145640_cached_sizze_147701, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145641_cached_sizze_147702 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145641, &mem_145641_cached_sizze_147702, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145654_cached_sizze_147703 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145654, &mem_145654_cached_sizze_147703, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145655_cached_sizze_147704 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145655, &mem_145655_cached_sizze_147704, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145656_cached_sizze_147705 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145656, &mem_145656_cached_sizze_147705, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145702_cached_sizze_147706 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145702, &mem_145702_cached_sizze_147706, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145708_cached_sizze_147707 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145708, &mem_145708_cached_sizze_147707, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145713_cached_sizze_147708 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145713, &mem_145713_cached_sizze_147708, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145729_cached_sizze_147709 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145729, &mem_145729_cached_sizze_147709, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145735_cached_sizze_147710 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145735, &mem_145735_cached_sizze_147710, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145740_cached_sizze_147711 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145740, &mem_145740_cached_sizze_147711, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145756_cached_sizze_147712 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145756, &mem_145756_cached_sizze_147712, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145762_cached_sizze_147713 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145762, &mem_145762_cached_sizze_147713, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145767_cached_sizze_147714 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145767, &mem_145767_cached_sizze_147714, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145774_cached_sizze_147715 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145774, &mem_145774_cached_sizze_147715, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145781_cached_sizze_147716 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145781, &mem_145781_cached_sizze_147716, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145797_cached_sizze_147717 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145797, &mem_145797_cached_sizze_147717, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145803_cached_sizze_147718 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145803, &mem_145803_cached_sizze_147718, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145808_cached_sizze_147719 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145808, &mem_145808_cached_sizze_147719, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145824_cached_sizze_147720 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145824, &mem_145824_cached_sizze_147720, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145830_cached_sizze_147721 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145830, &mem_145830_cached_sizze_147721, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145835_cached_sizze_147722 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_145835, &mem_145835_cached_sizze_147722, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145851_cached_sizze_147723 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145851, &mem_145851_cached_sizze_147723, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145856_cached_sizze_147724 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145856, &mem_145856_cached_sizze_147724, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145867_cached_sizze_147725 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145867, &mem_145867_cached_sizze_147725, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145872_cached_sizze_147726 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145872, &mem_145872_cached_sizze_147726, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145883_cached_sizze_147727 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145883, &mem_145883_cached_sizze_147727, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145888_cached_sizze_147728 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145888, &mem_145888_cached_sizze_147728, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145899_cached_sizze_147729 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145899, &mem_145899_cached_sizze_147729, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145904_cached_sizze_147730 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145904, &mem_145904_cached_sizze_147730, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145915_cached_sizze_147731 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145915, &mem_145915_cached_sizze_147731, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145922_cached_sizze_147732 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145922, &mem_145922_cached_sizze_147732, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145929_cached_sizze_147733 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145929, &mem_145929_cached_sizze_147733, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145934_cached_sizze_147734 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145934, &mem_145934_cached_sizze_147734, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145945_cached_sizze_147735 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145945, &mem_145945_cached_sizze_147735, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145950_cached_sizze_147736 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145950, &mem_145950_cached_sizze_147736, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145961_cached_sizze_147737 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_145961, &mem_145961_cached_sizze_147737, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145966_cached_sizze_147738 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_145966, &mem_145966_cached_sizze_147738, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145977_cached_sizze_147739 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145977, &mem_145977_cached_sizze_147739, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145982_cached_sizze_147740 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145982, &mem_145982_cached_sizze_147740, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145993_cached_sizze_147741 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_145993, &mem_145993_cached_sizze_147741, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_145998_cached_sizze_147742 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_145998, &mem_145998_cached_sizze_147742, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146009_cached_sizze_147743 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146009, &mem_146009_cached_sizze_147743, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146014_cached_sizze_147744 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146014, &mem_146014_cached_sizze_147744, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146025_cached_sizze_147745 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146025, &mem_146025_cached_sizze_147745, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146026_cached_sizze_147746 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146026, &mem_146026_cached_sizze_147746, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146035_cached_sizze_147747 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146035, &mem_146035_cached_sizze_147747, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146036_cached_sizze_147748 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146036, &mem_146036_cached_sizze_147748, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146049_cached_sizze_147749 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146049, &mem_146049_cached_sizze_147749, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146050_cached_sizze_147750 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146050, &mem_146050_cached_sizze_147750, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146063_cached_sizze_147751 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146063, &mem_146063_cached_sizze_147751, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146064_cached_sizze_147752 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146064, &mem_146064_cached_sizze_147752, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146085_cached_sizze_147753 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146085, &mem_146085_cached_sizze_147753, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146092_cached_sizze_147754 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146092, &mem_146092_cached_sizze_147754, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146097_cached_sizze_147755 < (int64_t) 108) {
        err = lexical_realloc(ctx, &mem_146097, &mem_146097_cached_sizze_147755, (int64_t) 108);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146108_cached_sizze_147756 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146108, &mem_146108_cached_sizze_147756, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146113_cached_sizze_147757 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146113, &mem_146113_cached_sizze_147757, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146124_cached_sizze_147758 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146124, &mem_146124_cached_sizze_147758, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146125_cached_sizze_147759 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146125, &mem_146125_cached_sizze_147759, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146134_cached_sizze_147760 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146134, &mem_146134_cached_sizze_147760, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146135_cached_sizze_147761 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146135, &mem_146135_cached_sizze_147761, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146156_cached_sizze_147762 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146156, &mem_146156_cached_sizze_147762, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146161_cached_sizze_147763 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146161, &mem_146161_cached_sizze_147763, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146172_cached_sizze_147764 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146172, &mem_146172_cached_sizze_147764, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146177_cached_sizze_147765 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146177, &mem_146177_cached_sizze_147765, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146188_cached_sizze_147766 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146188, &mem_146188_cached_sizze_147766, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146195_cached_sizze_147767 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146195, &mem_146195_cached_sizze_147767, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146202_cached_sizze_147768 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146202, &mem_146202_cached_sizze_147768, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146212_cached_sizze_147769 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146212, &mem_146212_cached_sizze_147769, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146217_cached_sizze_147770 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146217, &mem_146217_cached_sizze_147770, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146228_cached_sizze_147771 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146228, &mem_146228_cached_sizze_147771, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146229_cached_sizze_147772 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146229, &mem_146229_cached_sizze_147772, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146238_cached_sizze_147773 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146238, &mem_146238_cached_sizze_147773, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146239_cached_sizze_147774 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146239, &mem_146239_cached_sizze_147774, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146260_cached_sizze_147775 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146260, &mem_146260_cached_sizze_147775, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146266_cached_sizze_147776 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146266, &mem_146266_cached_sizze_147776, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146271_cached_sizze_147777 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146271, &mem_146271_cached_sizze_147777, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146287_cached_sizze_147778 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146287, &mem_146287_cached_sizze_147778, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146293_cached_sizze_147779 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146293, &mem_146293_cached_sizze_147779, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146298_cached_sizze_147780 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146298, &mem_146298_cached_sizze_147780, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146314_cached_sizze_147781 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146314, &mem_146314_cached_sizze_147781, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146315_cached_sizze_147782 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146315, &mem_146315_cached_sizze_147782, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146326_cached_sizze_147783 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146326, &mem_146326_cached_sizze_147783, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146327_cached_sizze_147784 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146327, &mem_146327_cached_sizze_147784, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146336_cached_sizze_147785 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146336, &mem_146336_cached_sizze_147785, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146343_cached_sizze_147786 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146343, &mem_146343_cached_sizze_147786, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146368_cached_sizze_147787 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146368, &mem_146368_cached_sizze_147787, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146374_cached_sizze_147788 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146374, &mem_146374_cached_sizze_147788, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146379_cached_sizze_147789 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146379, &mem_146379_cached_sizze_147789, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146395_cached_sizze_147790 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146395, &mem_146395_cached_sizze_147790, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146400_cached_sizze_147791 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146400, &mem_146400_cached_sizze_147791, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146411_cached_sizze_147792 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146411, &mem_146411_cached_sizze_147792, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146417_cached_sizze_147793 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146417, &mem_146417_cached_sizze_147793, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146422_cached_sizze_147794 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146422, &mem_146422_cached_sizze_147794, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146438_cached_sizze_147795 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146438, &mem_146438_cached_sizze_147795, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146444_cached_sizze_147796 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146444, &mem_146444_cached_sizze_147796, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146449_cached_sizze_147797 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146449, &mem_146449_cached_sizze_147797, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146465_cached_sizze_147798 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146465, &mem_146465_cached_sizze_147798, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146466_cached_sizze_147799 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146466, &mem_146466_cached_sizze_147799, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146477_cached_sizze_147800 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146477, &mem_146477_cached_sizze_147800, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146478_cached_sizze_147801 < (int64_t) 256) {
        err = lexical_realloc(ctx, &mem_146478, &mem_146478_cached_sizze_147801, (int64_t) 256);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146487_cached_sizze_147802 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146487, &mem_146487_cached_sizze_147802, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146488_cached_sizze_147803 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146488, &mem_146488_cached_sizze_147803, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146519_cached_sizze_147804 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146519, &mem_146519_cached_sizze_147804, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146520_cached_sizze_147805 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146520, &mem_146520_cached_sizze_147805, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146521_cached_sizze_147806 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146521, &mem_146521_cached_sizze_147806, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146537_cached_sizze_147807 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146537, &mem_146537_cached_sizze_147807, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146538_cached_sizze_147808 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146538, &mem_146538_cached_sizze_147808, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146539_cached_sizze_147809 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146539, &mem_146539_cached_sizze_147809, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146552_cached_sizze_147810 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146552, &mem_146552_cached_sizze_147810, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146553_cached_sizze_147811 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146553, &mem_146553_cached_sizze_147811, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146554_cached_sizze_147812 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_146554, &mem_146554_cached_sizze_147812, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146600_cached_sizze_147813 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146600, &mem_146600_cached_sizze_147813, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146601_cached_sizze_147814 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146601, &mem_146601_cached_sizze_147814, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146602_cached_sizze_147815 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146602, &mem_146602_cached_sizze_147815, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146615_cached_sizze_147816 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146615, &mem_146615_cached_sizze_147816, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146616_cached_sizze_147817 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146616, &mem_146616_cached_sizze_147817, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146617_cached_sizze_147818 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146617, &mem_146617_cached_sizze_147818, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146648_cached_sizze_147819 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146648, &mem_146648_cached_sizze_147819, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146649_cached_sizze_147820 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146649, &mem_146649_cached_sizze_147820, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146650_cached_sizze_147821 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146650, &mem_146650_cached_sizze_147821, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146651_cached_sizze_147822 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146651, &mem_146651_cached_sizze_147822, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146668_cached_sizze_147823 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146668, &mem_146668_cached_sizze_147823, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146669_cached_sizze_147824 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146669, &mem_146669_cached_sizze_147824, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146670_cached_sizze_147825 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146670, &mem_146670_cached_sizze_147825, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146671_cached_sizze_147826 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146671, &mem_146671_cached_sizze_147826, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146712_cached_sizze_147827 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146712, &mem_146712_cached_sizze_147827, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146719_cached_sizze_147828 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146719, &mem_146719_cached_sizze_147828, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146726_cached_sizze_147829 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146726, &mem_146726_cached_sizze_147829, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146736_cached_sizze_147830 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146736, &mem_146736_cached_sizze_147830, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146741_cached_sizze_147831 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146741, &mem_146741_cached_sizze_147831, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146752_cached_sizze_147832 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146752, &mem_146752_cached_sizze_147832, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146759_cached_sizze_147833 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146759, &mem_146759_cached_sizze_147833, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146766_cached_sizze_147834 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146766, &mem_146766_cached_sizze_147834, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146776_cached_sizze_147835 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146776, &mem_146776_cached_sizze_147835, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146781_cached_sizze_147836 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146781, &mem_146781_cached_sizze_147836, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146792_cached_sizze_147837 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146792, &mem_146792_cached_sizze_147837, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146793_cached_sizze_147838 < (int64_t) 1024) {
        err = lexical_realloc(ctx, &mem_146793, &mem_146793_cached_sizze_147838, (int64_t) 1024);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146802_cached_sizze_147839 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146802, &mem_146802_cached_sizze_147839, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146803_cached_sizze_147840 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146803, &mem_146803_cached_sizze_147840, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146824_cached_sizze_147841 < (int64_t) 4096) {
        err = lexical_realloc(ctx, &mem_146824, &mem_146824_cached_sizze_147841, (int64_t) 4096);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146829_cached_sizze_147842 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146829, &mem_146829_cached_sizze_147842, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146840_cached_sizze_147843 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146840, &mem_146840_cached_sizze_147843, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146841_cached_sizze_147844 < (int64_t) 1728) {
        err = lexical_realloc(ctx, &mem_146841, &mem_146841_cached_sizze_147844, (int64_t) 1728);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146850_cached_sizze_147845 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146850, &mem_146850_cached_sizze_147845, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:4:11-25
    if (mem_146851_cached_sizze_147846 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_146851, &mem_146851_cached_sizze_147846, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // futhark/microgpt.fut:509:5-514:51
    if (memblock_set(ctx, &mem_param_145244, &wdown_mem_145211, "wdown_mem_145211") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145248, &wkey_mem_145212, "wkey_mem_145212") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145252, &wout_mem_145213, "wout_mem_145213") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145256, &wpe_mem_145214, "wpe_mem_145214") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145260, &wqry_mem_145215, "wqry_mem_145215") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145264, &wte_mem_145216, "wte_mem_145216") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145268, &wup_mem_145217, "wup_mem_145217") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145272, &wval_mem_145218, "wval_mem_145218") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145276, &wvoc_mem_145219, "wvoc_mem_145219") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145280, &wdown_mem_145220, "wdown_mem_145220") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145284, &wkey_mem_145221, "wkey_mem_145221") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145288, &wout_mem_145222, "wout_mem_145222") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145292, &wpe_mem_145223, "wpe_mem_145223") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145296, &wqry_mem_145224, "wqry_mem_145224") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145300, &wte_mem_145225, "wte_mem_145225") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145304, &wup_mem_145226, "wup_mem_145226") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145308, &wval_mem_145227, "wval_mem_145227") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145312, &wvoc_mem_145228, "wvoc_mem_145228") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145316, &wdown_mem_145229, "wdown_mem_145229") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145320, &wkey_mem_145230, "wkey_mem_145230") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145324, &wout_mem_145231, "wout_mem_145231") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145328, &wpe_mem_145232, "wpe_mem_145232") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145332, &wqry_mem_145233, "wqry_mem_145233") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145336, &wte_mem_145234, "wte_mem_145234") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145340, &wup_mem_145235, "wup_mem_145235") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145344, &wval_mem_145236, "wval_mem_145236") != 0)
        return 1;
    if (memblock_set(ctx, &mem_param_145348, &wvoc_mem_145237, "wvoc_mem_145237") != 0)
        return 1;
    for (int64_t step_137620 = 0; step_137620 < (int64_t) 2; step_137620++) {
        // futhark/microgpt.fut:511:16-25
        
        int64_t dl_137648 = ((int64_t *) dls_mem_145239.mem)[step_137620];
        
        // futhark/microgpt.fut:482:32-53
        // futhark/microgpt.fut:482:32-53
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_145349.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) seqs_mem_145240.mem, step_137620 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        // futhark/microgpt.fut:482:32-53
        if (futrts_cal_target_9281(ctx, &ext_mem_145351, mem_145349, dl_137648) != 0) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144257 = 0; i_144257 < (int64_t) 16; i_144257++) {
            int64_t tmp_140913 = ((int64_t *) seqs_mem_145240.mem)[step_137620 * (int64_t) 16 + i_144257];
            
            // futhark/microgpt.fut:484:38-53
            
            bool x_140914 = sle64((int64_t) 0, tmp_140913);
            
            // futhark/microgpt.fut:484:38-53
            
            bool y_140915 = slt64(tmp_140913, (int64_t) 27);
            
            // futhark/microgpt.fut:484:38-53
            
            bool bounds_check_140916 = x_140914 && y_140915;
            
            // futhark/microgpt.fut:484:38-53
            
            bool index_certs_140917;
            
            if (!bounds_check_140916) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_140913, "] out of bounds for array of shape [", (long long) (int64_t) 27, "].", "-> #0  futhark/microgpt.fut:484:38-53\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:9:27-39\n   #3  futhark/microgpt.fut:4:11-25\n   #4  futhark/microgpt.fut:9:13-40\n   #5  futhark/microgpt.fut:15:29-44\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:15:15-45\n   #8  futhark/microgpt.fut:484:13-57\n   #9  futhark/microgpt.fut:492:26-498:29\n   #10 futhark/microgpt.fut:514:11-50\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144247 = 0; i_144247 < (int64_t) 16; i_144247++) {
                // futhark/microgpt.fut:4:11-25
                
                float lifted_lambda_res_140924 = ((float *) mem_param_145264.mem)[tmp_140913 * (int64_t) 16 + i_144247];
                
                ((float *) mem_145362)[i_144247] = lifted_lambda_res_140924;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144251 = 0; i_144251 < (int64_t) 27; i_144251++) {
                // futhark/microgpt.fut:514:11-50
                
                float zt_rhs_140938 = ((float *) ext_mem_145351.mem)[i_144257 * (int64_t) 27 + i_144251];
                
                // futhark/microgpt.fut:318:61-113
                
                float zt_res_140939 = -6.25e-2F * zt_rhs_140938;
                
                ((float *) mem_145369)[i_144251] = zt_res_140939;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145352, i_144257 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145369, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145353, i_144257 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145362, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        if (memblock_unref(ctx, &ext_mem_145351, "ext_mem_145351") != 0)
            return 1;
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144266 = 0; i_144266 < (int64_t) 16; i_144266++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144262 = 0; i_144262 < (int64_t) 16; i_144262++) {
                // futhark/microgpt.fut:514:11-50
                
                float zp_lhs_137686 = ((float *) mem_145353)[i_144266 * (int64_t) 16 + i_144262];
                
                // futhark/microgpt.fut:4:11-25
                
                float zp_rhs_137687 = ((float *) mem_param_145256.mem)[i_144266 * (int64_t) 16 + i_144262];
                
                // futhark/microgpt.fut:279:52-82
                
                float zp_res_137688 = zp_lhs_137686 + zp_rhs_137687;
                
                ((float *) mem_145389)[i_144262] = zp_res_137688;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145384, i_144266 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145389, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144274 = 0; i_144274 < (int64_t) 16; i_144274++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144270 = 0; i_144270 < (int64_t) 16; i_144270++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_137703 = ((float *) mem_145384)[i_144274 * (int64_t) 16 + i_144270];
                
                // futhark/microgpt.fut:280:40-73
                
                float zt_res_137704 = zt_lhs_137703 * zt_lhs_137703;
                
                ((float *) mem_145405)[i_144270] = zt_res_137704;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145400, i_144274 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145405, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144278 = 0; i_144278 < (int64_t) 16; i_144278++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_137713;
            float r_137715 = 0.0F;
            
            for (int64_t i_137714 = 0; i_137714 < (int64_t) 16; i_137714++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_137716 = ((float *) mem_145400)[i_144278 * (int64_t) 16 + i_137714];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_137717 = r_137715 + lifted_lambda_res_137716;
                float r_tmp_147247 = zp_res_137717;
                
                r_137715 = r_tmp_147247;
            }
            defunc_0_lifted_lambda_res_137713 = r_137715;
            // futhark/microgpt.fut:281:36-87
            
            float zs_res_137718 = defunc_0_lifted_lambda_res_137713 / 16.0F;
            
            ((float *) mem_145416)[i_144278] = zs_res_137718;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144282 = 0; i_144282 < (int64_t) 16; i_144282++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zp_lhs_137726 = ((float *) mem_145416)[i_144282];
            
            // futhark/microgpt.fut:282:45-85
            
            float zp_res_137727 = 1.0e-5F + zp_lhs_137726;
            
            // futhark/microgpt.fut:282:37-85
            
            float sqrt_res_137728 = futrts_sqrt32(zp_res_137727);
            
            ((float *) mem_145423)[i_144282] = sqrt_res_137728;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144290 = 0; i_144290 < (int64_t) 16; i_144290++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_137736 = ((float *) mem_145423)[i_144290];
            
            // futhark/microgpt.fut:283:79-100
            
            float zs_res_137737 = 1.0F / zs_rhs_137736;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144286 = 0; i_144286 < (int64_t) 16; i_144286++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_137744 = ((float *) mem_145384)[i_144290 * (int64_t) 16 + i_144286];
                
                // futhark/microgpt.fut:283:56-100
                
                float zt_res_137745 = zs_res_137737 * zt_lhs_137744;
                
                ((float *) mem_145435)[i_144286] = zt_res_137745;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145430, i_144290 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145435, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144298 = 0; i_144298 < (int64_t) 16; i_144298++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144294 = 0; i_144294 < (int64_t) 16; i_144294++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_137760 = ((float *) mem_145430)[i_144298 * (int64_t) 16 + i_144294];
                
                // futhark/microgpt.fut:284:44-85
                
                float zt_res_137761 = zt_lhs_137760 * zt_lhs_137760;
                
                ((float *) mem_145451)[i_144294] = zt_res_137761;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145446, i_144298 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145451, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144302 = 0; i_144302 < (int64_t) 16; i_144302++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_137770;
            float r_137772 = 0.0F;
            
            for (int64_t i_137771 = 0; i_137771 < (int64_t) 16; i_137771++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_137773 = ((float *) mem_145446)[i_144302 * (int64_t) 16 + i_137771];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_137774 = r_137772 + lifted_lambda_res_137773;
                float r_tmp_147254 = zp_res_137774;
                
                r_137772 = r_tmp_147254;
            }
            defunc_0_lifted_lambda_res_137770 = r_137772;
            // futhark/microgpt.fut:285:38-91
            
            float zs_res_137775 = defunc_0_lifted_lambda_res_137770 / 16.0F;
            
            ((float *) mem_145462)[i_144302] = zs_res_137775;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144306 = 0; i_144306 < (int64_t) 16; i_144306++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zp_lhs_137783 = ((float *) mem_145462)[i_144306];
            
            // futhark/microgpt.fut:286:45-86
            
            float zp_res_137784 = 1.0e-5F + zp_lhs_137783;
            
            // futhark/microgpt.fut:286:37-86
            
            float sqrt_res_137785 = futrts_sqrt32(zp_res_137784);
            
            ((float *) mem_145469)[i_144306] = sqrt_res_137785;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144314 = 0; i_144314 < (int64_t) 16; i_144314++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_137793 = ((float *) mem_145469)[i_144314];
            
            // futhark/microgpt.fut:287:80-101
            
            float zs_res_137794 = 1.0F / zs_rhs_137793;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144310 = 0; i_144310 < (int64_t) 16; i_144310++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_137801 = ((float *) mem_145430)[i_144314 * (int64_t) 16 + i_144310];
                
                // futhark/microgpt.fut:287:56-101
                
                float zt_res_137802 = zs_res_137794 * zt_lhs_137801;
                
                ((float *) mem_145481)[i_144310] = zt_res_137802;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145476, i_144314 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145481, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144332 = 0; i_144332 < (int64_t) 16; i_144332++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144322 = 0; i_144322 < (int64_t) 16; i_144322++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_142409;
                float r_142411 = 0.0F;
                
                for (int64_t i_142410 = 0; i_142410 < (int64_t) 16; i_142410++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_142412 = ((float *) mem_param_145260.mem)[i_144322 * (int64_t) 16 + i_142410];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_142413 = ((float *) mem_145476)[i_144332 * (int64_t) 16 + i_142410];
                    
                    // futhark/microgpt.fut:288:75-112
                    
                    float zt_res_142414 = zt_lhs_142412 * zt_rhs_142413;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_142415 = r_142411 + zt_res_142414;
                    float r_tmp_147264 = zp_res_142415;
                    
                    r_142411 = r_tmp_147264;
                }
                defunc_0_lifted_lambda_res_142409 = r_142411;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_142422;
                float r_142424 = 0.0F;
                
                for (int64_t i_142423 = 0; i_142423 < (int64_t) 16; i_142423++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_142425 = ((float *) mem_param_145248.mem)[i_144322 * (int64_t) 16 + i_142423];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_142426 = ((float *) mem_145476)[i_144332 * (int64_t) 16 + i_142423];
                    
                    // futhark/microgpt.fut:289:75-112
                    
                    float zt_res_142427 = zt_lhs_142425 * zt_rhs_142426;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_142428 = r_142424 + zt_res_142427;
                    float r_tmp_147265 = zp_res_142428;
                    
                    r_142424 = r_tmp_147265;
                }
                defunc_0_lifted_lambda_res_142422 = r_142424;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_142438;
                float r_142440 = 0.0F;
                
                for (int64_t i_142439 = 0; i_142439 < (int64_t) 16; i_142439++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_142441 = ((float *) mem_param_145272.mem)[i_144322 * (int64_t) 16 + i_142439];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_142442 = ((float *) mem_145476)[i_144332 * (int64_t) 16 + i_142439];
                    
                    // futhark/microgpt.fut:290:75-112
                    
                    float zt_res_142443 = zt_lhs_142441 * zt_rhs_142442;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_142444 = r_142440 + zt_res_142443;
                    float r_tmp_147266 = zp_res_142444;
                    
                    r_142440 = r_tmp_147266;
                }
                defunc_0_lifted_lambda_res_142438 = r_142440;
                ((float *) mem_145507)[i_144322] = defunc_0_lifted_lambda_res_142438;
                ((float *) mem_145508)[i_144322] = defunc_0_lifted_lambda_res_142422;
                ((float *) mem_145509)[i_144322] = defunc_0_lifted_lambda_res_142409;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145492, i_144332 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145507, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145493, i_144332 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145508, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145494, i_144332 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145509, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144362 = 0; i_144362 < (int64_t) 16; i_144362++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144352 = 0; i_144352 < (int64_t) 4; i_144352++) {
                // futhark/microgpt.fut:291:92-95
                
                int64_t zp_lhs_142503 = mul64((int64_t) 4, i_144352);
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144342 = 0; i_144342 < (int64_t) 4; i_144342++) {
                    // futhark/microgpt.fut:291:97-102
                    
                    int64_t tmp_142587 = add64(zp_lhs_142503, i_144342);
                    
                    // futhark/microgpt.fut:291:72-104
                    
                    bool x_142588 = sle64((int64_t) 0, tmp_142587);
                    
                    // futhark/microgpt.fut:291:72-104
                    
                    bool y_142589 = slt64(tmp_142587, (int64_t) 16);
                    
                    // futhark/microgpt.fut:291:72-104
                    
                    bool bounds_check_142590 = x_142588 && y_142589;
                    
                    // futhark/microgpt.fut:291:72-104
                    
                    bool index_certs_142591;
                    
                    if (!bounds_check_142590) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_142587, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:291:72-104\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:291:55-105\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:291:37-107\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:9:27-39\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:9:13-40\n   #11 futhark/microgpt.fut:291:12-109\n   #12 futhark/microgpt.fut:487:5-74\n   #13 futhark/microgpt.fut:492:26-498:29\n   #14 futhark/microgpt.fut:514:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142592 = ((float *) mem_145494)[i_144362 * (int64_t) 16 + tmp_142587];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142600 = ((float *) mem_145493)[i_144362 * (int64_t) 16 + tmp_142587];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142611 = ((float *) mem_145492)[i_144362 * (int64_t) 16 + tmp_142587];
                    
                    ((float *) mem_145573)[i_144342] = lifted_lambda_res_142611;
                    ((float *) mem_145574)[i_144342] = lifted_lambda_res_142600;
                    ((float *) mem_145575)[i_144342] = lifted_lambda_res_142592;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145558, i_144352 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145573, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145559, i_144352 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145574, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145560, i_144352 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145575, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145540, i_144362 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145558, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145541, i_144362 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145559, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145542, i_144362 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145560, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144392 = 0; i_144392 < (int64_t) 4; i_144392++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144382 = 0; i_144382 < (int64_t) 16; i_144382++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144372 = 0; i_144372 < (int64_t) 4; i_144372++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142772 = ((float *) mem_145542)[i_144382 * (int64_t) 16 + i_144392 * (int64_t) 4 + i_144372];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142779 = ((float *) mem_145541)[i_144382 * (int64_t) 16 + i_144392 * (int64_t) 4 + i_144372];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142789 = ((float *) mem_145540)[i_144382 * (int64_t) 16 + i_144392 * (int64_t) 4 + i_144372];
                    
                    ((float *) mem_145654)[i_144372] = lifted_lambda_res_142789;
                    ((float *) mem_145655)[i_144372] = lifted_lambda_res_142779;
                    ((float *) mem_145656)[i_144372] = lifted_lambda_res_142772;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145639, i_144382 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145654, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145640, i_144382 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145655, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145641, i_144382 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145621, i_144392 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145639, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145622, i_144392 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145640, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145623, i_144392 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145641, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144406 = 0; i_144406 < (int64_t) 4; i_144406++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144402 = 0; i_144402 < (int64_t) 16; i_144402++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144398 = 0; i_144398 < (int64_t) 16; i_144398++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_138013;
                    float r_138015 = 0.0F;
                    
                    for (int64_t i_138014 = 0; i_138014 < (int64_t) 4; i_138014++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_138016 = ((float *) mem_145623)[i_144406 * (int64_t) 64 + i_144402 * (int64_t) 4 + i_138014];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_138017 = ((float *) mem_145622)[i_144406 * (int64_t) 64 + i_144398 * (int64_t) 4 + i_138014];
                        
                        // futhark/microgpt.fut:297:92-143
                        
                        float zt_res_138018 = zt_lhs_138016 * zt_rhs_138017;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_138019 = r_138015 + zt_res_138018;
                        float r_tmp_147288 = zp_res_138019;
                        
                        r_138015 = r_tmp_147288;
                    }
                    defunc_0_lifted_lambda_res_138013 = r_138015;
                    ((float *) mem_145713)[i_144398] = defunc_0_lifted_lambda_res_138013;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145708, i_144402 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145713, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145702, i_144406 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145708, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144418 = 0; i_144418 < (int64_t) 4; i_144418++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144414 = 0; i_144414 < (int64_t) 16; i_144414++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144410 = 0; i_144410 < (int64_t) 16; i_144410++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zs_lhs_138041 = ((float *) mem_145702)[i_144418 * (int64_t) 256 + i_144414 * (int64_t) 16 + i_144410];
                    
                    // futhark/microgpt.fut:298:75-112
                    
                    float zs_res_138042 = zs_lhs_138041 / 2.0F;
                    float zp_rhs_138043 = ((float *) masks_mem_145238.mem)[step_137620 * (int64_t) 256 + i_144414 * (int64_t) 16 + i_144410];
                    
                    // futhark/microgpt.fut:298:99-137
                    
                    float zp_res_138044 = zs_res_138042 + zp_rhs_138043;
                    
                    ((float *) mem_145740)[i_144410] = zp_res_138044;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145735, i_144414 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145740, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145729, i_144418 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145735, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144440 = 0; i_144440 < (int64_t) 4; i_144440++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144436 = 0; i_144436 < (int64_t) 16; i_144436++) {
                // futhark/microgpt.fut:103:13-33
                
                float defunc_0_reduce_res_144080;
                float redout_144420 = -INFINITY;
                
                for (int64_t i_144421 = 0; i_144421 < (int64_t) 16; i_144421++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_142818 = ((float *) mem_145729)[i_144440 * (int64_t) 256 + i_144436 * (int64_t) 16 + i_144421];
                    
                    // futhark/microgpt.fut:103:13-33
                    
                    float max_res_138072 = fmax32(lifted_lambda_res_142818, redout_144420);
                    float redout_tmp_147294 = max_res_138072;
                    
                    redout_144420 = redout_tmp_147294;
                }
                defunc_0_reduce_res_144080 = redout_144420;
                // futhark/microgpt.fut:113:47-56
                
                float neg_res_138073 = -defunc_0_reduce_res_144080;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144424 = 0; i_144424 < (int64_t) 16; i_144424++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_138080 = ((float *) mem_145729)[i_144440 * (int64_t) 256 + i_144436 * (int64_t) 16 + i_144424];
                    
                    // futhark/microgpt.fut:113:38-56
                    
                    float zp_res_138081 = neg_res_138073 + lifted_lambda_res_138080;
                    
                    // futhark/microgpt.fut:113:31-56
                    
                    float exp_res_138082 = futrts_exp32(zp_res_138081);
                    
                    ((float *) mem_145767)[i_144424] = exp_res_138082;
                }
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138084;
                float r_138086 = 0.0F;
                
                for (int64_t i_138085 = 0; i_138085 < (int64_t) 16; i_138085++) {
                    // futhark/microgpt.fut:114:32-39
                    
                    float lifted_lambda_res_138087 = ((float *) mem_145767)[i_138085];
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138088 = r_138086 + lifted_lambda_res_138087;
                    float r_tmp_147296 = zp_res_138088;
                    
                    r_138086 = r_tmp_147296;
                }
                defunc_0_lifted_lambda_res_138084 = r_138086;
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144428 = 0; i_144428 < (int64_t) 16; i_144428++) {
                    // futhark/microgpt.fut:115:23-30
                    
                    float zs_lhs_138095 = ((float *) mem_145767)[i_144428];
                    
                    // futhark/microgpt.fut:115:23-40
                    
                    float zs_res_138096 = zs_lhs_138095 / defunc_0_lifted_lambda_res_138084;
                    
                    ((float *) mem_145774)[i_144428] = zs_res_138096;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144432 = 0; i_144432 < (int64_t) 16; i_144432++) {
                    // futhark/microgpt.fut:300:23-31
                    
                    float lifted_lambda_res_138104 = ((float *) mem_145774)[i_144432];
                    
                    ((float *) mem_145781)[i_144432] = lifted_lambda_res_138104;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145762, i_144436 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145756, i_144440 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_145762, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144452 = 0; i_144452 < (int64_t) 4; i_144452++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144448 = 0; i_144448 < (int64_t) 16; i_144448++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144444 = 0; i_144444 < (int64_t) 4; i_144444++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_138126;
                    float r_138128 = 0.0F;
                    
                    for (int64_t i_138127 = 0; i_138127 < (int64_t) 16; i_138127++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_138129 = ((float *) mem_145756)[i_144452 * (int64_t) 256 + i_144448 * (int64_t) 16 + i_138127];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_138130 = ((float *) mem_145621)[i_144452 * (int64_t) 64 + i_138127 * (int64_t) 4 + i_144444];
                        
                        // futhark/microgpt.fut:301:95-150
                        
                        float zt_res_138131 = zt_lhs_138129 * zt_rhs_138130;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_138132 = r_138128 + zt_res_138131;
                        float r_tmp_147302 = zp_res_138132;
                        
                        r_138128 = r_tmp_147302;
                    }
                    defunc_0_lifted_lambda_res_138126 = r_138128;
                    ((float *) mem_145808)[i_144444] = defunc_0_lifted_lambda_res_138126;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145803, i_144448 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145808, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145797, i_144452 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145803, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144464 = 0; i_144464 < (int64_t) 16; i_144464++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144460 = 0; i_144460 < (int64_t) 4; i_144460++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144456 = 0; i_144456 < (int64_t) 4; i_144456++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_138154 = ((float *) mem_145797)[i_144460 * (int64_t) 64 + i_144464 * (int64_t) 4 + i_144456];
                    
                    ((float *) mem_145835)[i_144456] = lifted_lambda_res_138154;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_145830, i_144460 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145835, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_145824, i_144464 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_145830, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144472 = 0; i_144472 < (int64_t) 16; i_144472++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144468 = 0; i_144468 < (int64_t) 16; i_144468++) {
                // futhark/microgpt.fut:303:82-85
                
                int64_t tmp_138166 = sdiv64(i_144468, (int64_t) 4);
                
                // futhark/microgpt.fut:303:59-87
                
                bool x_138167 = sle64((int64_t) 0, tmp_138166);
                
                // futhark/microgpt.fut:303:59-87
                
                bool y_138168 = slt64(tmp_138166, (int64_t) 4);
                
                // futhark/microgpt.fut:303:59-87
                
                bool bounds_check_138169 = x_138167 && y_138168;
                
                // futhark/microgpt.fut:303:59-87
                
                bool index_certs_138170;
                
                if (!bounds_check_138169) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_138166, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:303:59-87\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:303:40-100\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:303:13-102\n   #9  futhark/microgpt.fut:487:5-74\n   #10 futhark/microgpt.fut:492:26-498:29\n   #11 futhark/microgpt.fut:514:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:303:94-97
                
                int64_t tmp_138171 = smod64(i_144468, (int64_t) 4);
                
                // futhark/microgpt.fut:303:59-99
                
                bool x_138172 = sle64((int64_t) 0, tmp_138171);
                
                // futhark/microgpt.fut:303:59-99
                
                bool y_138173 = slt64(tmp_138171, (int64_t) 4);
                
                // futhark/microgpt.fut:303:59-99
                
                bool bounds_check_138174 = x_138172 && y_138173;
                
                // futhark/microgpt.fut:303:59-99
                
                bool index_certs_138175;
                
                if (!bounds_check_138174) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_138171, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:303:59-99\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:303:40-100\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:303:13-102\n   #9  futhark/microgpt.fut:487:5-74\n   #10 futhark/microgpt.fut:492:26-498:29\n   #11 futhark/microgpt.fut:514:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_138176 = ((float *) mem_145824)[i_144472 * (int64_t) 16 + tmp_138166 * (int64_t) 4 + tmp_138171];
                
                ((float *) mem_145856)[i_144468] = lifted_lambda_res_138176;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145851, i_144472 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145856, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144480 = 0; i_144480 < (int64_t) 16; i_144480++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144476 = 0; i_144476 < (int64_t) 16; i_144476++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138191;
                float r_138193 = 0.0F;
                
                for (int64_t i_138192 = 0; i_138192 < (int64_t) 16; i_138192++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138194 = ((float *) mem_param_145252.mem)[i_144476 * (int64_t) 16 + i_138192];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138195 = ((float *) mem_145851)[i_144480 * (int64_t) 16 + i_138192];
                    
                    // futhark/microgpt.fut:304:80-123
                    
                    float zt_res_138196 = zt_lhs_138194 * zt_rhs_138195;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138197 = r_138193 + zt_res_138196;
                    float r_tmp_147310 = zp_res_138197;
                    
                    r_138193 = r_tmp_147310;
                }
                defunc_0_lifted_lambda_res_138191 = r_138193;
                ((float *) mem_145872)[i_144476] = defunc_0_lifted_lambda_res_138191;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145867, i_144480 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145872, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144488 = 0; i_144488 < (int64_t) 16; i_144488++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144484 = 0; i_144484 < (int64_t) 16; i_144484++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_lhs_138212 = ((float *) mem_145476)[i_144488 * (int64_t) 16 + i_144484];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_rhs_138213 = ((float *) mem_145867)[i_144488 * (int64_t) 16 + i_144484];
                
                // futhark/microgpt.fut:305:48-96
                
                float zp_res_138214 = zp_lhs_138212 + zp_rhs_138213;
                
                ((float *) mem_145888)[i_144484] = zp_res_138214;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145883, i_144488 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145888, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144496 = 0; i_144496 < (int64_t) 16; i_144496++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144492 = 0; i_144492 < (int64_t) 16; i_144492++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138229 = ((float *) mem_145883)[i_144496 * (int64_t) 16 + i_144492];
                
                // futhark/microgpt.fut:306:48-97
                
                float zt_res_138230 = zt_lhs_138229 * zt_lhs_138229;
                
                ((float *) mem_145904)[i_144492] = zt_res_138230;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145899, i_144496 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145904, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144500 = 0; i_144500 < (int64_t) 16; i_144500++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_138239;
            float r_138241 = 0.0F;
            
            for (int64_t i_138240 = 0; i_138240 < (int64_t) 16; i_138240++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_138242 = ((float *) mem_145899)[i_144500 * (int64_t) 16 + i_138240];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_138243 = r_138241 + lifted_lambda_res_138242;
                float r_tmp_147316 = zp_res_138243;
                
                r_138241 = r_tmp_147316;
            }
            defunc_0_lifted_lambda_res_138239 = r_138241;
            // futhark/microgpt.fut:307:41-99
            
            float zs_res_138244 = defunc_0_lifted_lambda_res_138239 / 16.0F;
            
            ((float *) mem_145915)[i_144500] = zs_res_138244;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144504 = 0; i_144504 < (int64_t) 16; i_144504++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zp_lhs_138252 = ((float *) mem_145915)[i_144504];
            
            // futhark/microgpt.fut:308:48-92
            
            float zp_res_138253 = 1.0e-5F + zp_lhs_138252;
            
            // futhark/microgpt.fut:308:40-92
            
            float sqrt_res_138254 = futrts_sqrt32(zp_res_138253);
            
            ((float *) mem_145922)[i_144504] = sqrt_res_138254;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144512 = 0; i_144512 < (int64_t) 16; i_144512++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_138262 = ((float *) mem_145922)[i_144512];
            
            // futhark/microgpt.fut:309:88-112
            
            float zs_res_138263 = 1.0F / zs_rhs_138262;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144508 = 0; i_144508 < (int64_t) 16; i_144508++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138270 = ((float *) mem_145883)[i_144512 * (int64_t) 16 + i_144508];
                
                // futhark/microgpt.fut:309:60-112
                
                float zt_res_138271 = zs_res_138263 * zt_lhs_138270;
                
                ((float *) mem_145934)[i_144508] = zt_res_138271;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145929, i_144512 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145934, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144520 = 0; i_144520 < (int64_t) 16; i_144520++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144516 = 0; i_144516 < (int64_t) 64; i_144516++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138287;
                float r_138289 = 0.0F;
                
                for (int64_t i_138288 = 0; i_138288 < (int64_t) 16; i_138288++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138290 = ((float *) mem_param_145268.mem)[i_144516 * (int64_t) 16 + i_138288];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138291 = ((float *) mem_145929)[i_144520 * (int64_t) 16 + i_138288];
                    
                    // futhark/microgpt.fut:310:80-122
                    
                    float zt_res_138292 = zt_lhs_138290 * zt_rhs_138291;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138293 = r_138289 + zt_res_138292;
                    float r_tmp_147322 = zp_res_138293;
                    
                    r_138289 = r_tmp_147322;
                }
                defunc_0_lifted_lambda_res_138287 = r_138289;
                ((float *) mem_145950)[i_144516] = defunc_0_lifted_lambda_res_138287;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145945, i_144520 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145950, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144528 = 0; i_144528 < (int64_t) 16; i_144528++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144524 = 0; i_144524 < (int64_t) 64; i_144524++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float max_arg0_138308 = ((float *) mem_145945)[i_144528 * (int64_t) 64 + i_144524];
                
                // futhark/microgpt.fut:311:48-81
                
                float max_res_138309 = fmax32(0.0F, max_arg0_138308);
                
                ((float *) mem_145966)[i_144524] = max_res_138309;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145961, i_144528 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145966, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144536 = 0; i_144536 < (int64_t) 16; i_144536++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144532 = 0; i_144532 < (int64_t) 16; i_144532++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138324;
                float r_138326 = 0.0F;
                
                for (int64_t i_138325 = 0; i_138325 < (int64_t) 64; i_138325++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138327 = ((float *) mem_param_145244.mem)[i_144532 * (int64_t) 64 + i_138325];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138328 = ((float *) mem_145961)[i_144536 * (int64_t) 64 + i_138325];
                    
                    // futhark/microgpt.fut:312:80-124
                    
                    float zt_res_138329 = zt_lhs_138327 * zt_rhs_138328;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138330 = r_138326 + zt_res_138329;
                    float r_tmp_147327 = zp_res_138330;
                    
                    r_138326 = r_tmp_147327;
                }
                defunc_0_lifted_lambda_res_138324 = r_138326;
                ((float *) mem_145982)[i_144532] = defunc_0_lifted_lambda_res_138324;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145977, i_144536 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145982, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144544 = 0; i_144544 < (int64_t) 16; i_144544++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144540 = 0; i_144540 < (int64_t) 16; i_144540++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_lhs_138345 = ((float *) mem_145883)[i_144544 * (int64_t) 16 + i_144540];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_rhs_138346 = ((float *) mem_145977)[i_144544 * (int64_t) 16 + i_144540];
                
                // futhark/microgpt.fut:313:48-97
                
                float zp_res_138347 = zp_lhs_138345 + zp_rhs_138346;
                
                ((float *) mem_145998)[i_144540] = zp_res_138347;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_145993, i_144544 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_145998, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144552 = 0; i_144552 < (int64_t) 16; i_144552++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144548 = 0; i_144548 < (int64_t) 27; i_144548++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138363;
                float r_138365 = 0.0F;
                
                for (int64_t i_138364 = 0; i_138364 < (int64_t) 16; i_138364++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138366 = ((float *) mem_param_145276.mem)[i_144548 * (int64_t) 16 + i_138364];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138367 = ((float *) mem_145993)[i_144552 * (int64_t) 16 + i_138364];
                    
                    // futhark/microgpt.fut:314:80-123
                    
                    float zt_res_138368 = zt_lhs_138366 * zt_rhs_138367;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138369 = r_138365 + zt_res_138368;
                    float r_tmp_147332 = zp_res_138369;
                    
                    r_138365 = r_tmp_147332;
                }
                defunc_0_lifted_lambda_res_138363 = r_138365;
                ((float *) mem_146014)[i_144548] = defunc_0_lifted_lambda_res_138363;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146009, i_144552 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146014, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144582 = 0; i_144582 < (int64_t) 16; i_144582++) {
            // futhark/microgpt.fut:103:13-33
            
            float defunc_0_reduce_res_144099;
            float defunc_0_reduce_res_144100;
            float redout_144554;
            float redout_144555;
            
            redout_144554 = -INFINITY;
            redout_144555 = -INFINITY;
            for (int64_t i_144556 = 0; i_144556 < (int64_t) 27; i_144556++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_142904 = ((float *) mem_146009)[i_144582 * (int64_t) 27 + i_144556];
                
                // futhark/microgpt.fut:103:13-33
                
                float max_res_141366 = fmax32(lifted_lambda_res_142904, redout_144554);
                
                // futhark/microgpt.fut:103:13-33
                
                float max_res_141418 = fmax32(lifted_lambda_res_142904, redout_144555);
                float redout_tmp_147335 = max_res_141366;
                float redout_tmp_147336 = max_res_141418;
                
                redout_144554 = redout_tmp_147335;
                redout_144555 = redout_tmp_147336;
            }
            defunc_0_reduce_res_144099 = redout_144554;
            defunc_0_reduce_res_144100 = redout_144555;
            // futhark/microgpt.fut:113:47-56
            
            float neg_res_141367 = -defunc_0_reduce_res_144099;
            
            // futhark/microgpt.fut:113:47-56
            
            float neg_res_141419 = -defunc_0_reduce_res_144100;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144561 = 0; i_144561 < (int64_t) 27; i_144561++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_142943 = ((float *) mem_146009)[i_144582 * (int64_t) 27 + i_144561];
                
                // futhark/microgpt.fut:113:38-56
                
                float zp_res_142944 = neg_res_141367 + lifted_lambda_res_142943;
                
                // futhark/microgpt.fut:113:31-56
                
                float exp_res_142945 = futrts_exp32(zp_res_142944);
                
                // futhark/microgpt.fut:113:38-56
                
                float zp_res_142953 = neg_res_141419 + lifted_lambda_res_142943;
                
                // futhark/microgpt.fut:113:31-56
                
                float exp_res_142954 = futrts_exp32(zp_res_142953);
                
                ((float *) mem_146035)[i_144561] = exp_res_142954;
                ((float *) mem_146036)[i_144561] = exp_res_142945;
            }
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141378;
            float r_141380 = 0.0F;
            
            for (int64_t i_141379 = 0; i_141379 < (int64_t) 27; i_141379++) {
                // futhark/microgpt.fut:114:32-39
                
                float lifted_lambda_res_141381 = ((float *) mem_146036)[i_141379];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141382 = r_141380 + lifted_lambda_res_141381;
                float r_tmp_147339 = zp_res_141382;
                
                r_141380 = r_tmp_147339;
            }
            defunc_0_lifted_lambda_res_141378 = r_141380;
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_141430;
            float r_141432 = 0.0F;
            
            for (int64_t i_141431 = 0; i_141431 < (int64_t) 27; i_141431++) {
                // futhark/microgpt.fut:114:32-39
                
                float lifted_lambda_res_141433 = ((float *) mem_146035)[i_141431];
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_141434 = r_141432 + lifted_lambda_res_141433;
                float r_tmp_147340 = zp_res_141434;
                
                r_141432 = r_tmp_147340;
            }
            defunc_0_lifted_lambda_res_141430 = r_141432;
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144568 = 0; i_144568 < (int64_t) 27; i_144568++) {
                // futhark/microgpt.fut:115:23-30
                
                float zs_lhs_142972 = ((float *) mem_146036)[i_144568];
                
                // futhark/microgpt.fut:115:23-40
                
                float zs_res_142973 = zs_lhs_142972 / defunc_0_lifted_lambda_res_141378;
                
                // futhark/microgpt.fut:115:23-30
                
                float zs_lhs_142980 = ((float *) mem_146035)[i_144568];
                
                // futhark/microgpt.fut:115:23-40
                
                float zs_res_142981 = zs_lhs_142980 / defunc_0_lifted_lambda_res_141430;
                
                ((float *) mem_146049)[i_144568] = zs_res_142981;
                ((float *) mem_146050)[i_144568] = zs_res_142973;
            }
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144575 = 0; i_144575 < (int64_t) 27; i_144575++) {
                // futhark/microgpt.fut:320:24-34
                
                float lifted_lambda_res_142999 = ((float *) mem_146050)[i_144575];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_143006 = ((float *) mem_145352)[i_144582 * (int64_t) 27 + i_144575];
                
                // futhark/microgpt.fut:322:4-14
                
                float zs_rhs_143007 = ((float *) mem_146049)[i_144575];
                
                // futhark/microgpt.fut:321:88-322:14
                
                float zs_res_143008 = 1.0F / zs_rhs_143007;
                
                // futhark/microgpt.fut:321:60-322:14
                
                float zt_res_143009 = zt_lhs_143006 * zs_res_143008;
                
                ((float *) mem_146063)[i_144575] = zt_res_143009;
                ((float *) mem_146064)[i_144575] = lifted_lambda_res_142999;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146025, i_144582 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146063, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146026, i_144582 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146064, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144587 = 0; i_144587 < (int64_t) 16; i_144587++) {
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_138503;
            float r_138505 = 0.0F;
            
            for (int64_t i_138504 = 0; i_138504 < (int64_t) 27; i_138504++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138506 = ((float *) mem_146025)[i_144587 * (int64_t) 27 + i_138504];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_138507 = ((float *) mem_146026)[i_144587 * (int64_t) 27 + i_138504];
                
                // futhark/microgpt.fut:323:60-109
                
                float zt_res_138508 = zt_lhs_138506 * zt_rhs_138507;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_138509 = r_138505 + zt_res_138508;
                float r_tmp_147346 = zp_res_138509;
                
                r_138505 = r_tmp_147346;
            }
            defunc_0_lifted_lambda_res_138503 = r_138505;
            ((float *) mem_146085)[i_144587] = defunc_0_lifted_lambda_res_138503;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144595 = 0; i_144595 < (int64_t) 16; i_144595++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float neg_arg0_138517 = ((float *) mem_146085)[i_144595];
            
            // futhark/microgpt.fut:324:116-138
            
            float neg_res_138518 = -neg_arg0_138517;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144591 = 0; i_144591 < (int64_t) 27; i_144591++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138525 = ((float *) mem_146026)[i_144595 * (int64_t) 27 + i_144591];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_lhs_138526 = ((float *) mem_146025)[i_144595 * (int64_t) 27 + i_144591];
                
                // futhark/microgpt.fut:324:88-138
                
                float zp_res_138527 = neg_res_138518 + zp_lhs_138526;
                
                // futhark/microgpt.fut:324:60-138
                
                float zt_res_138528 = zt_lhs_138525 * zp_res_138527;
                
                ((float *) mem_146097)[i_144591] = zt_res_138528;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146092, i_144595 * (int64_t) 27, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146097, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 27});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144603 = 0; i_144603 < (int64_t) 16; i_144603++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144599 = 0; i_144599 < (int64_t) 16; i_144599++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138543;
                float r_138545 = 0.0F;
                
                for (int64_t i_138544 = 0; i_138544 < (int64_t) 27; i_138544++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138546 = ((float *) mem_param_145276.mem)[i_138544 * (int64_t) 16 + i_144599];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138547 = ((float *) mem_146092)[i_144603 * (int64_t) 27 + i_138544];
                    
                    // futhark/microgpt.fut:325:80-123
                    
                    float zt_res_138548 = zt_lhs_138546 * zt_rhs_138547;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138549 = r_138545 + zt_res_138548;
                    float r_tmp_147351 = zp_res_138549;
                    
                    r_138545 = r_tmp_147351;
                }
                defunc_0_lifted_lambda_res_138543 = r_138545;
                ((float *) mem_146113)[i_144599] = defunc_0_lifted_lambda_res_138543;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146108, i_144603 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146113, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144616 = 0; i_144616 < (int64_t) 16; i_144616++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144609 = 0; i_144609 < (int64_t) 64; i_144609++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143037;
                float r_143039 = 0.0F;
                
                for (int64_t i_143038 = 0; i_143038 < (int64_t) 16; i_143038++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_143040 = ((float *) mem_param_145244.mem)[i_143038 * (int64_t) 64 + i_144609];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143041 = ((float *) mem_146108)[i_144616 * (int64_t) 16 + i_143038];
                    
                    // futhark/microgpt.fut:326:80-124
                    
                    float zt_res_143042 = zt_lhs_143040 * zt_rhs_143041;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143043 = r_143039 + zt_res_143042;
                    float r_tmp_147356 = zp_res_143043;
                    
                    r_143039 = r_tmp_147356;
                }
                defunc_0_lifted_lambda_res_143037 = r_143039;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143050;
                float r_143052 = 0.0F;
                
                for (int64_t i_143051 = 0; i_143051 < (int64_t) 16; i_143051++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143053 = ((float *) mem_146108)[i_143051 * (int64_t) 16 + i_144616];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143054 = ((float *) mem_145961)[i_143051 * (int64_t) 64 + i_144609];
                    
                    // futhark/microgpt.fut:366:82-131
                    
                    float zt_res_143055 = zt_lhs_143053 * zt_rhs_143054;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143056 = r_143052 + zt_res_143055;
                    float r_tmp_147357 = zp_res_143056;
                    
                    r_143052 = r_tmp_147357;
                }
                defunc_0_lifted_lambda_res_143050 = r_143052;
                ((float *) mem_146134)[i_144609] = defunc_0_lifted_lambda_res_143050;
                ((float *) mem_146135)[i_144609] = defunc_0_lifted_lambda_res_143037;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146124, i_144616 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146134, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146125, i_144616 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146135, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144625 = 0; i_144625 < (int64_t) 16; i_144625++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144621 = 0; i_144621 < (int64_t) 64; i_144621++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float indicatorp_arg0_138585 = ((float *) mem_145945)[i_144625 * (int64_t) 64 + i_144621];
                
                // futhark/microgpt.fut:125:42-54
                
                float max_res_138586 = fmax32(0.0F, indicatorp_arg0_138585);
                
                // futhark/microgpt.fut:125:35-54
                
                float sgn_res_138587 = fsignum32(max_res_138586);
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_138588 = ((float *) mem_146125)[i_144625 * (int64_t) 64 + i_144621];
                
                // futhark/microgpt.fut:327:49-110
                
                float zt_res_138589 = sgn_res_138587 * zt_rhs_138588;
                
                ((float *) mem_146161)[i_144621] = zt_res_138589;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146156, i_144625 * (int64_t) 64, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146161, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 64});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144633 = 0; i_144633 < (int64_t) 16; i_144633++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144629 = 0; i_144629 < (int64_t) 16; i_144629++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138604;
                float r_138606 = 0.0F;
                
                for (int64_t i_138605 = 0; i_138605 < (int64_t) 64; i_138605++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_138607 = ((float *) mem_param_145268.mem)[i_138605 * (int64_t) 16 + i_144629];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138608 = ((float *) mem_146156)[i_144633 * (int64_t) 64 + i_138605];
                    
                    // futhark/microgpt.fut:328:80-122
                    
                    float zt_res_138609 = zt_lhs_138607 * zt_rhs_138608;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138610 = r_138606 + zt_res_138609;
                    float r_tmp_147362 = zp_res_138610;
                    
                    r_138606 = r_tmp_147362;
                }
                defunc_0_lifted_lambda_res_138604 = r_138606;
                ((float *) mem_146177)[i_144629] = defunc_0_lifted_lambda_res_138604;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146172, i_144633 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146177, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144637 = 0; i_144637 < (int64_t) 16; i_144637++) {
            float f_elem_138615 = ((float *) mem_145922)[i_144637];
            
            // futhark/microgpt.fut:329:68-92
            
            float zs_res_138620 = 1.0F / f_elem_138615;
            
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_138621;
            float r_138623 = 0.0F;
            
            for (int64_t i_138622 = 0; i_138622 < (int64_t) 16; i_138622++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138624 = ((float *) mem_145883)[i_144637 * (int64_t) 16 + i_138622];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_138625 = ((float *) mem_146172)[i_144637 * (int64_t) 16 + i_138622];
                
                // futhark/microgpt.fut:329:100-149
                
                float zt_res_138626 = zt_lhs_138624 * zt_rhs_138625;
                
                // futhark/microgpt.fut:329:123-180
                
                float zt_res_138627 = zs_res_138620 * zt_res_138626;
                
                // futhark/microgpt.fut:329:72-180
                
                float zt_res_138628 = zs_res_138620 * zt_res_138627;
                
                // futhark/microgpt.fut:329:60-180
                
                float neg_res_138629 = -zt_res_138628;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_138630 = r_138623 + neg_res_138629;
                float r_tmp_147364 = zp_res_138630;
                
                r_138623 = r_tmp_147364;
            }
            defunc_0_lifted_lambda_res_138621 = r_138623;
            ((float *) mem_146188)[i_144637] = defunc_0_lifted_lambda_res_138621;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144641 = 0; i_144641 < (int64_t) 16; i_144641++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_lhs_138638 = ((float *) mem_146188)[i_144641];
            
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_rhs_138639 = ((float *) mem_145922)[i_144641];
            
            // futhark/microgpt.fut:330:76-105
            
            float zt_res_138640 = 2.0F * zt_rhs_138639;
            
            // futhark/microgpt.fut:330:62-105
            
            float zs_res_138641 = 1.0F / zt_res_138640;
            
            // futhark/microgpt.fut:330:40-105
            
            float zt_res_138642 = zt_lhs_138638 * zs_res_138641;
            
            ((float *) mem_146195)[i_144641] = zt_res_138642;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144645 = 0; i_144645 < (int64_t) 16; i_144645++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_lhs_138650 = ((float *) mem_146195)[i_144645];
            
            // futhark/microgpt.fut:331:60-91
            
            float zs_res_138651 = zs_lhs_138650 / 16.0F;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_147367 = 0; nest_i_147367 < (int64_t) 16; nest_i_147367++) {
                ((float *) mem_146202)[i_144645 * (int64_t) 16 + nest_i_147367] = zs_res_138651;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144653 = 0; i_144653 < (int64_t) 16; i_144653++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_138660 = ((float *) mem_145922)[i_144653];
            
            // futhark/microgpt.fut:332:118-142
            
            float zs_res_138661 = 1.0F / zs_rhs_138660;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144649 = 0; i_144649 < (int64_t) 16; i_144649++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_lhs_138668 = ((float *) mem_146108)[i_144653 * (int64_t) 16 + i_144649];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138669 = ((float *) mem_146172)[i_144653 * (int64_t) 16 + i_144649];
                
                // futhark/microgpt.fut:332:90-142
                
                float zt_res_138670 = zs_res_138661 * zt_lhs_138669;
                
                // futhark/microgpt.fut:332:62-142
                
                float zp_res_138671 = zp_lhs_138668 + zt_res_138670;
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_138672 = ((float *) mem_145883)[i_144653 * (int64_t) 16 + i_144649];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_138673 = ((float *) mem_146202)[i_144653 * (int64_t) 16 + i_144649];
                
                // futhark/microgpt.fut:332:151-200
                
                float zt_res_138674 = zt_lhs_138672 * zt_rhs_138673;
                
                // futhark/microgpt.fut:332:85-200
                
                float zp_res_138675 = zp_res_138671 + zt_res_138674;
                
                // futhark/microgpt.fut:332:146-257
                
                float zp_res_138676 = zt_res_138674 + zp_res_138675;
                
                ((float *) mem_146217)[i_144649] = zp_res_138676;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146212, i_144653 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146217, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144666 = 0; i_144666 < (int64_t) 16; i_144666++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144659 = 0; i_144659 < (int64_t) 16; i_144659++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143079;
                float r_143081 = 0.0F;
                
                for (int64_t i_143080 = 0; i_143080 < (int64_t) 16; i_143080++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_143082 = ((float *) mem_param_145252.mem)[i_143080 * (int64_t) 16 + i_144659];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143083 = ((float *) mem_146212)[i_144666 * (int64_t) 16 + i_143080];
                    
                    // futhark/microgpt.fut:333:80-123
                    
                    float zt_res_143084 = zt_lhs_143082 * zt_rhs_143083;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143085 = r_143081 + zt_res_143084;
                    float r_tmp_147374 = zp_res_143085;
                    
                    r_143081 = r_tmp_147374;
                }
                defunc_0_lifted_lambda_res_143079 = r_143081;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143092;
                float r_143094 = 0.0F;
                
                for (int64_t i_143093 = 0; i_143093 < (int64_t) 16; i_143093++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143095 = ((float *) mem_146212)[i_143093 * (int64_t) 16 + i_144666];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143096 = ((float *) mem_145851)[i_143093 * (int64_t) 16 + i_144659];
                    
                    // futhark/microgpt.fut:364:81-130
                    
                    float zt_res_143097 = zt_lhs_143095 * zt_rhs_143096;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143098 = r_143094 + zt_res_143097;
                    float r_tmp_147375 = zp_res_143098;
                    
                    r_143094 = r_tmp_147375;
                }
                defunc_0_lifted_lambda_res_143092 = r_143094;
                ((float *) mem_146238)[i_144659] = defunc_0_lifted_lambda_res_143092;
                ((float *) mem_146239)[i_144659] = defunc_0_lifted_lambda_res_143079;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146228, i_144666 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146238, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146229, i_144666 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146239, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144679 = 0; i_144679 < (int64_t) 16; i_144679++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144675 = 0; i_144675 < (int64_t) 4; i_144675++) {
                // futhark/microgpt.fut:334:101-104
                
                int64_t zp_lhs_138708 = mul64((int64_t) 4, i_144675);
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144671 = 0; i_144671 < (int64_t) 4; i_144671++) {
                    // futhark/microgpt.fut:334:106-112
                    
                    int64_t tmp_138711 = add64(zp_lhs_138708, i_144671);
                    
                    // futhark/microgpt.fut:334:77-114
                    
                    bool x_138712 = sle64((int64_t) 0, tmp_138711);
                    
                    // futhark/microgpt.fut:334:77-114
                    
                    bool y_138713 = slt64(tmp_138711, (int64_t) 16);
                    
                    // futhark/microgpt.fut:334:77-114
                    
                    bool bounds_check_138714 = x_138712 && y_138713;
                    
                    // futhark/microgpt.fut:334:77-114
                    
                    bool index_certs_138715;
                    
                    if (!bounds_check_138714) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_138711, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  futhark/microgpt.fut:334:77-114\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:334:59-115\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:6:13-17\n   #6  futhark/microgpt.fut:334:40-117\n   #7  futhark/microgpt.fut:4:11-25\n   #8  futhark/microgpt.fut:9:27-39\n   #9  futhark/microgpt.fut:4:11-25\n   #10 futhark/microgpt.fut:9:13-40\n   #11 futhark/microgpt.fut:334:13-119\n   #12 futhark/microgpt.fut:487:5-74\n   #13 futhark/microgpt.fut:492:26-498:29\n   #14 futhark/microgpt.fut:514:11-50\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_138716 = ((float *) mem_146229)[i_144679 * (int64_t) 16 + tmp_138711];
                    
                    ((float *) mem_146271)[i_144671] = lifted_lambda_res_138716;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146266, i_144675 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146271, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146260, i_144679 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146266, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144691 = 0; i_144691 < (int64_t) 4; i_144691++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144687 = 0; i_144687 < (int64_t) 16; i_144687++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144683 = 0; i_144683 < (int64_t) 4; i_144683++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_138738 = ((float *) mem_146260)[i_144687 * (int64_t) 16 + i_144691 * (int64_t) 4 + i_144683];
                    
                    ((float *) mem_146298)[i_144683] = lifted_lambda_res_138738;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146293, i_144687 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146298, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146287, i_144691 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146293, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144712 = 0; i_144712 < (int64_t) 4; i_144712++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144705 = 0; i_144705 < (int64_t) 16; i_144705++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144695 = 0; i_144695 < (int64_t) 16; i_144695++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_143128;
                    float r_143130 = 0.0F;
                    
                    for (int64_t i_143129 = 0; i_143129 < (int64_t) 4; i_143129++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_143131 = ((float *) mem_146287)[i_144712 * (int64_t) 64 + i_144705 * (int64_t) 4 + i_143129];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_143132 = ((float *) mem_145621)[i_144712 * (int64_t) 64 + i_144695 * (int64_t) 4 + i_143129];
                        
                        // futhark/microgpt.fut:336:98-158
                        
                        float zt_res_143133 = zt_lhs_143131 * zt_rhs_143132;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_143134 = r_143130 + zt_res_143133;
                        float r_tmp_147387 = zp_res_143134;
                        
                        r_143130 = r_tmp_147387;
                    }
                    defunc_0_lifted_lambda_res_143128 = r_143130;
                    ((float *) mem_146336)[i_144695] = defunc_0_lifted_lambda_res_143128;
                }
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144699 = 0; i_144699 < (int64_t) 4; i_144699++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_143148;
                    float r_143150 = 0.0F;
                    
                    for (int64_t i_143149 = 0; i_143149 < (int64_t) 16; i_143149++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_143151 = ((float *) mem_145756)[i_144712 * (int64_t) 256 + i_143149 * (int64_t) 16 + i_144705];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_143152 = ((float *) mem_146287)[i_144712 * (int64_t) 64 + i_143149 * (int64_t) 4 + i_144699];
                        
                        // futhark/microgpt.fut:341:98-158
                        
                        float zt_res_143153 = zt_lhs_143151 * zt_rhs_143152;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_143154 = r_143150 + zt_res_143153;
                        float r_tmp_147389 = zp_res_143154;
                        
                        r_143150 = r_tmp_147389;
                    }
                    defunc_0_lifted_lambda_res_143148 = r_143150;
                    ((float *) mem_146343)[i_144699] = defunc_0_lifted_lambda_res_143148;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146326, i_144705 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146343, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146327, i_144705 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146336, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146314, i_144712 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146326, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146315, i_144712 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146327, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144725 = 0; i_144725 < (int64_t) 4; i_144725++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144721 = 0; i_144721 < (int64_t) 16; i_144721++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144717 = 0; i_144717 < (int64_t) 16; i_144717++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_138788 = ((float *) mem_146315)[i_144725 * (int64_t) 256 + i_144721 * (int64_t) 16 + i_144717];
                    
                    ((float *) mem_146379)[i_144717] = lifted_lambda_res_138788;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146374, i_144721 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146379, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146368, i_144725 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146374, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144733 = 0; i_144733 < (int64_t) 4; i_144733++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144729 = 0; i_144729 < (int64_t) 16; i_144729++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_138804;
                float r_138806 = 0.0F;
                
                for (int64_t i_138805 = 0; i_138805 < (int64_t) 16; i_138805++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_138807 = ((float *) mem_146368)[i_144733 * (int64_t) 256 + i_144729 * (int64_t) 16 + i_138805];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_138808 = ((float *) mem_145756)[i_144733 * (int64_t) 256 + i_144729 * (int64_t) 16 + i_138805];
                    
                    // futhark/microgpt.fut:338:79-139
                    
                    float zt_res_138809 = zt_lhs_138807 * zt_rhs_138808;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_138810 = r_138806 + zt_res_138809;
                    float r_tmp_147395 = zp_res_138810;
                    
                    r_138806 = r_tmp_147395;
                }
                defunc_0_lifted_lambda_res_138804 = r_138806;
                ((float *) mem_146400)[i_144729] = defunc_0_lifted_lambda_res_138804;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146395, i_144733 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146400, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144745 = 0; i_144745 < (int64_t) 4; i_144745++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144741 = 0; i_144741 < (int64_t) 16; i_144741++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float neg_arg0_138825 = ((float *) mem_146395)[i_144745 * (int64_t) 16 + i_144741];
                
                // futhark/microgpt.fut:339:146-174
                
                float neg_res_138826 = -neg_arg0_138825;
                
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144737 = 0; i_144737 < (int64_t) 16; i_144737++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_138833 = ((float *) mem_145756)[i_144745 * (int64_t) 256 + i_144741 * (int64_t) 16 + i_144737];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zp_lhs_138834 = ((float *) mem_146368)[i_144745 * (int64_t) 256 + i_144741 * (int64_t) 16 + i_144737];
                    
                    // futhark/microgpt.fut:339:112-174
                    
                    float zp_res_138835 = neg_res_138826 + zp_lhs_138834;
                    
                    // futhark/microgpt.fut:339:79-174
                    
                    float zt_res_138836 = zt_lhs_138833 * zp_res_138835;
                    
                    ((float *) mem_146422)[i_144737] = zt_res_138836;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146417, i_144741 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146422, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146411, i_144745 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146417, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144757 = 0; i_144757 < (int64_t) 4; i_144757++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144753 = 0; i_144753 < (int64_t) 16; i_144753++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144749 = 0; i_144749 < (int64_t) 16; i_144749++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zs_lhs_138858 = ((float *) mem_146411)[i_144757 * (int64_t) 256 + i_144753 * (int64_t) 16 + i_144749];
                    
                    // futhark/microgpt.fut:340:55-97
                    
                    float zs_res_138859 = zs_lhs_138858 / 2.0F;
                    
                    ((float *) mem_146449)[i_144749] = zs_res_138859;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146444, i_144753 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146449, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146438, i_144757 * (int64_t) 256, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146444, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144777 = 0; i_144777 < (int64_t) 4; i_144777++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144770 = 0; i_144770 < (int64_t) 16; i_144770++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144763 = 0; i_144763 < (int64_t) 4; i_144763++) {
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_143237;
                    float r_143239 = 0.0F;
                    
                    for (int64_t i_143238 = 0; i_143238 < (int64_t) 16; i_143238++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_143240 = ((float *) mem_145623)[i_144777 * (int64_t) 64 + i_143238 * (int64_t) 4 + i_144763];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_143241 = ((float *) mem_146438)[i_144777 * (int64_t) 256 + i_143238 * (int64_t) 16 + i_144770];
                        
                        // futhark/microgpt.fut:342:98-158
                        
                        float zt_res_143242 = zt_lhs_143240 * zt_rhs_143241;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_143243 = r_143239 + zt_res_143242;
                        float r_tmp_147408 = zp_res_143243;
                        
                        r_143239 = r_tmp_147408;
                    }
                    defunc_0_lifted_lambda_res_143237 = r_143239;
                    // futhark/microgpt.fut:71:13-49
                    
                    float defunc_0_lifted_lambda_res_143250;
                    float r_143252 = 0.0F;
                    
                    for (int64_t i_143251 = 0; i_143251 < (int64_t) 16; i_143251++) {
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_lhs_143253 = ((float *) mem_146438)[i_144777 * (int64_t) 256 + i_144770 * (int64_t) 16 + i_143251];
                        
                        // futhark/microgpt.fut:272:5-372:64
                        
                        float zt_rhs_143254 = ((float *) mem_145622)[i_144777 * (int64_t) 64 + i_143251 * (int64_t) 4 + i_144763];
                        
                        // futhark/microgpt.fut:343:98-158
                        
                        float zt_res_143255 = zt_lhs_143253 * zt_rhs_143254;
                        
                        // futhark/microgpt.fut:71:40-49
                        
                        float zp_res_143256 = r_143252 + zt_res_143255;
                        float r_tmp_147409 = zp_res_143256;
                        
                        r_143252 = r_tmp_147409;
                    }
                    defunc_0_lifted_lambda_res_143250 = r_143252;
                    ((float *) mem_146487)[i_144763] = defunc_0_lifted_lambda_res_143250;
                    ((float *) mem_146488)[i_144763] = defunc_0_lifted_lambda_res_143237;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146477, i_144770 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146487, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146478, i_144770 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146488, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146465, i_144777 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146477, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146466, i_144777 * (int64_t) 64, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146478, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144806 = 0; i_144806 < (int64_t) 16; i_144806++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144796 = 0; i_144796 < (int64_t) 4; i_144796++) {
                // futhark/microgpt.fut:4:11-25
                for (int64_t i_144786 = 0; i_144786 < (int64_t) 4; i_144786++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_143413 = ((float *) mem_146314)[i_144796 * (int64_t) 64 + i_144806 * (int64_t) 4 + i_144786];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_143420 = ((float *) mem_146466)[i_144796 * (int64_t) 64 + i_144806 * (int64_t) 4 + i_144786];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float lifted_lambda_res_143430 = ((float *) mem_146465)[i_144796 * (int64_t) 64 + i_144806 * (int64_t) 4 + i_144786];
                    
                    ((float *) mem_146552)[i_144786] = lifted_lambda_res_143430;
                    ((float *) mem_146553)[i_144786] = lifted_lambda_res_143420;
                    ((float *) mem_146554)[i_144786] = lifted_lambda_res_143413;
                }
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146537, i_144796 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146552, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146538, i_144796 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146553, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
                lmad_copy_4b(ctx, 1, (uint32_t *) mem_146539, i_144796 * (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146554, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
            }
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146519, i_144806 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146537, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146520, i_144806 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146538, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
            lmad_copy_4b(ctx, 2, (uint32_t *) mem_146521, i_144806 * (int64_t) 16, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint32_t *) mem_146539, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (int64_t []) {(int64_t) 4, (int64_t) 4});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144826 = 0; i_144826 < (int64_t) 16; i_144826++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144816 = 0; i_144816 < (int64_t) 16; i_144816++) {
                // futhark/microgpt.fut:347:82-85
                
                int64_t tmp_143497 = sdiv64(i_144816, (int64_t) 4);
                
                // futhark/microgpt.fut:347:59-87
                
                bool x_143498 = sle64((int64_t) 0, tmp_143497);
                
                // futhark/microgpt.fut:347:59-87
                
                bool y_143499 = slt64(tmp_143497, (int64_t) 4);
                
                // futhark/microgpt.fut:347:59-87
                
                bool bounds_check_143500 = x_143498 && y_143499;
                
                // futhark/microgpt.fut:347:59-87
                
                bool index_certs_143501;
                
                if (!bounds_check_143500) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_143497, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:347:59-87\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:347:40-100\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:347:13-102\n   #9  futhark/microgpt.fut:487:5-74\n   #10 futhark/microgpt.fut:492:26-498:29\n   #11 futhark/microgpt.fut:514:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:347:94-97
                
                int64_t tmp_143502 = smod64(i_144816, (int64_t) 4);
                
                // futhark/microgpt.fut:347:59-99
                
                bool x_143503 = sle64((int64_t) 0, tmp_143502);
                
                // futhark/microgpt.fut:347:59-99
                
                bool y_143504 = slt64(tmp_143502, (int64_t) 4);
                
                // futhark/microgpt.fut:347:59-99
                
                bool bounds_check_143505 = x_143503 && y_143504;
                
                // futhark/microgpt.fut:347:59-99
                
                bool index_certs_143506;
                
                if (!bounds_check_143505) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_143502, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  futhark/microgpt.fut:347:59-99\n   #1  futhark/microgpt.fut:4:11-25\n   #2  futhark/microgpt.fut:6:13-17\n   #3  futhark/microgpt.fut:347:40-100\n   #4  futhark/microgpt.fut:4:11-25\n   #5  futhark/microgpt.fut:9:27-39\n   #6  futhark/microgpt.fut:4:11-25\n   #7  futhark/microgpt.fut:9:13-40\n   #8  futhark/microgpt.fut:347:13-102\n   #9  futhark/microgpt.fut:487:5-74\n   #10 futhark/microgpt.fut:492:26-498:29\n   #11 futhark/microgpt.fut:514:11-50\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_143507 = ((float *) mem_146521)[i_144826 * (int64_t) 16 + tmp_143497 * (int64_t) 4 + tmp_143502];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_143520 = ((float *) mem_146520)[i_144826 * (int64_t) 16 + tmp_143497 * (int64_t) 4 + tmp_143502];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_143536 = ((float *) mem_146519)[i_144826 * (int64_t) 16 + tmp_143497 * (int64_t) 4 + tmp_143502];
                
                ((float *) mem_146615)[i_144816] = lifted_lambda_res_143536;
                ((float *) mem_146616)[i_144816] = lifted_lambda_res_143520;
                ((float *) mem_146617)[i_144816] = lifted_lambda_res_143507;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146600, i_144826 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146615, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146601, i_144826 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146616, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146602, i_144826 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146617, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144851 = 0; i_144851 < (int64_t) 16; i_144851++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144838 = 0; i_144838 < (int64_t) 16; i_144838++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zp_lhs_143703 = ((float *) mem_146212)[i_144851 * (int64_t) 16 + i_144838];
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143704;
                float r_143706 = 0.0F;
                
                for (int64_t i_143705 = 0; i_143705 < (int64_t) 16; i_143705++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_143707 = ((float *) mem_param_145272.mem)[i_143705 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143708 = ((float *) mem_146602)[i_144851 * (int64_t) 16 + i_143705];
                    
                    // futhark/microgpt.fut:350:110-153
                    
                    float zt_res_143709 = zt_lhs_143707 * zt_rhs_143708;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143710 = r_143706 + zt_res_143709;
                    float r_tmp_147433 = zp_res_143710;
                    
                    r_143706 = r_tmp_147433;
                }
                defunc_0_lifted_lambda_res_143704 = r_143706;
                // futhark/microgpt.fut:350:62-155
                
                float zp_res_143711 = zp_lhs_143703 + defunc_0_lifted_lambda_res_143704;
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143712;
                float r_143714 = 0.0F;
                
                for (int64_t i_143713 = 0; i_143713 < (int64_t) 16; i_143713++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_143715 = ((float *) mem_param_145248.mem)[i_143713 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143716 = ((float *) mem_146601)[i_144851 * (int64_t) 16 + i_143713];
                    
                    // futhark/microgpt.fut:350:183-226
                    
                    float zt_res_143717 = zt_lhs_143715 * zt_rhs_143716;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143718 = r_143714 + zt_res_143717;
                    float r_tmp_147434 = zp_res_143718;
                    
                    r_143714 = r_tmp_147434;
                }
                defunc_0_lifted_lambda_res_143712 = r_143714;
                // futhark/microgpt.fut:350:85-228
                
                float zp_res_143719 = zp_res_143711 + defunc_0_lifted_lambda_res_143712;
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143720;
                float r_143722 = 0.0F;
                
                for (int64_t i_143721 = 0; i_143721 < (int64_t) 16; i_143721++) {
                    // futhark/microgpt.fut:71:46-49
                    
                    float zt_lhs_143723 = ((float *) mem_param_145260.mem)[i_143721 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143724 = ((float *) mem_146600)[i_144851 * (int64_t) 16 + i_143721];
                    
                    // futhark/microgpt.fut:350:256-299
                    
                    float zt_res_143725 = zt_lhs_143723 * zt_rhs_143724;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143726 = r_143722 + zt_res_143725;
                    float r_tmp_147435 = zp_res_143726;
                    
                    r_143722 = r_tmp_147435;
                }
                defunc_0_lifted_lambda_res_143720 = r_143722;
                // futhark/microgpt.fut:350:158-301
                
                float zp_res_143727 = zp_res_143719 + defunc_0_lifted_lambda_res_143720;
                
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143734;
                float r_143736 = 0.0F;
                
                for (int64_t i_143735 = 0; i_143735 < (int64_t) 16; i_143735++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143737 = ((float *) mem_146600)[i_143735 * (int64_t) 16 + i_144851];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143738 = ((float *) mem_145476)[i_143735 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:361:81-129
                    
                    float zt_res_143739 = zt_lhs_143737 * zt_rhs_143738;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143740 = r_143736 + zt_res_143739;
                    float r_tmp_147436 = zp_res_143740;
                    
                    r_143736 = r_tmp_147436;
                }
                defunc_0_lifted_lambda_res_143734 = r_143736;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143750;
                float r_143752 = 0.0F;
                
                for (int64_t i_143751 = 0; i_143751 < (int64_t) 16; i_143751++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143753 = ((float *) mem_146601)[i_143751 * (int64_t) 16 + i_144851];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143754 = ((float *) mem_145476)[i_143751 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:362:81-129
                    
                    float zt_res_143755 = zt_lhs_143753 * zt_rhs_143754;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143756 = r_143752 + zt_res_143755;
                    float r_tmp_147437 = zp_res_143756;
                    
                    r_143752 = r_tmp_147437;
                }
                defunc_0_lifted_lambda_res_143750 = r_143752;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143768;
                float r_143770 = 0.0F;
                
                for (int64_t i_143769 = 0; i_143769 < (int64_t) 16; i_143769++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143771 = ((float *) mem_146602)[i_143769 * (int64_t) 16 + i_144851];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143772 = ((float *) mem_145476)[i_143769 * (int64_t) 16 + i_144838];
                    
                    // futhark/microgpt.fut:363:81-129
                    
                    float zt_res_143773 = zt_lhs_143771 * zt_rhs_143772;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143774 = r_143770 + zt_res_143773;
                    float r_tmp_147438 = zp_res_143774;
                    
                    r_143770 = r_tmp_147438;
                }
                defunc_0_lifted_lambda_res_143768 = r_143770;
                ((float *) mem_146668)[i_144838] = defunc_0_lifted_lambda_res_143768;
                ((float *) mem_146669)[i_144838] = defunc_0_lifted_lambda_res_143750;
                ((float *) mem_146670)[i_144838] = defunc_0_lifted_lambda_res_143734;
                ((float *) mem_146671)[i_144838] = zp_res_143727;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146648, i_144851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146668, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146649, i_144851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146669, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146650, i_144851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146670, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146651, i_144851 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146671, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144858 = 0; i_144858 < (int64_t) 16; i_144858++) {
            float f_elem_139117 = ((float *) mem_145469)[i_144858];
            
            // futhark/microgpt.fut:351:68-91
            
            float zs_res_139122 = 1.0F / f_elem_139117;
            
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_139123;
            float r_139125 = 0.0F;
            
            for (int64_t i_139124 = 0; i_139124 < (int64_t) 16; i_139124++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139126 = ((float *) mem_145430)[i_144858 * (int64_t) 16 + i_139124];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_139127 = ((float *) mem_146651)[i_144858 * (int64_t) 16 + i_139124];
                
                // futhark/microgpt.fut:351:99-147
                
                float zt_res_139128 = zt_lhs_139126 * zt_rhs_139127;
                
                // futhark/microgpt.fut:351:121-177
                
                float zt_res_139129 = zs_res_139122 * zt_res_139128;
                
                // futhark/microgpt.fut:351:72-177
                
                float zt_res_139130 = zs_res_139122 * zt_res_139129;
                
                // futhark/microgpt.fut:351:60-177
                
                float neg_res_139131 = -zt_res_139130;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_139132 = r_139125 + neg_res_139131;
                float r_tmp_147440 = zp_res_139132;
                
                r_139125 = r_tmp_147440;
            }
            defunc_0_lifted_lambda_res_139123 = r_139125;
            ((float *) mem_146712)[i_144858] = defunc_0_lifted_lambda_res_139123;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144862 = 0; i_144862 < (int64_t) 16; i_144862++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_lhs_139140 = ((float *) mem_146712)[i_144862];
            
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_rhs_139141 = ((float *) mem_145469)[i_144862];
            
            // futhark/microgpt.fut:352:76-104
            
            float zt_res_139142 = 2.0F * zt_rhs_139141;
            
            // futhark/microgpt.fut:352:62-104
            
            float zs_res_139143 = 1.0F / zt_res_139142;
            
            // futhark/microgpt.fut:352:40-104
            
            float zt_res_139144 = zt_lhs_139140 * zs_res_139143;
            
            ((float *) mem_146719)[i_144862] = zt_res_139144;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144866 = 0; i_144866 < (int64_t) 16; i_144866++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_lhs_139152 = ((float *) mem_146719)[i_144866];
            
            // futhark/microgpt.fut:353:60-91
            
            float zs_res_139153 = zs_lhs_139152 / 16.0F;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_147443 = 0; nest_i_147443 < (int64_t) 16; nest_i_147443++) {
                ((float *) mem_146726)[i_144866 * (int64_t) 16 + nest_i_147443] = zs_res_139153;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144874 = 0; i_144874 < (int64_t) 16; i_144874++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_139162 = ((float *) mem_145469)[i_144874];
            
            // futhark/microgpt.fut:354:90-113
            
            float zs_res_139163 = 1.0F / zs_rhs_139162;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144870 = 0; i_144870 < (int64_t) 16; i_144870++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139170 = ((float *) mem_146651)[i_144874 * (int64_t) 16 + i_144870];
                
                // futhark/microgpt.fut:354:62-113
                
                float zt_res_139171 = zs_res_139163 * zt_lhs_139170;
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139172 = ((float *) mem_145430)[i_144874 * (int64_t) 16 + i_144870];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_139173 = ((float *) mem_146726)[i_144874 * (int64_t) 16 + i_144870];
                
                // futhark/microgpt.fut:354:121-169
                
                float zt_res_139174 = zt_lhs_139172 * zt_rhs_139173;
                
                // futhark/microgpt.fut:354:85-169
                
                float zp_res_139175 = zt_res_139171 + zt_res_139174;
                
                // futhark/microgpt.fut:354:116-225
                
                float zp_res_139176 = zt_res_139174 + zp_res_139175;
                
                ((float *) mem_146741)[i_144870] = zp_res_139176;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146736, i_144874 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146741, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144878 = 0; i_144878 < (int64_t) 16; i_144878++) {
            float f_elem_139181 = ((float *) mem_145423)[i_144878];
            
            // futhark/microgpt.fut:355:68-91
            
            float zs_res_139186 = 1.0F / f_elem_139181;
            
            // futhark/microgpt.fut:71:13-49
            
            float defunc_0_lifted_lambda_res_139187;
            float r_139189 = 0.0F;
            
            for (int64_t i_139188 = 0; i_139188 < (int64_t) 16; i_139188++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139190 = ((float *) mem_145384)[i_144878 * (int64_t) 16 + i_139188];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_139191 = ((float *) mem_146736)[i_144878 * (int64_t) 16 + i_139188];
                
                // futhark/microgpt.fut:355:99-146
                
                float zt_res_139192 = zt_lhs_139190 * zt_rhs_139191;
                
                // futhark/microgpt.fut:355:120-176
                
                float zt_res_139193 = zs_res_139186 * zt_res_139192;
                
                // futhark/microgpt.fut:355:72-176
                
                float zt_res_139194 = zs_res_139186 * zt_res_139193;
                
                // futhark/microgpt.fut:355:60-176
                
                float neg_res_139195 = -zt_res_139194;
                
                // futhark/microgpt.fut:71:40-49
                
                float zp_res_139196 = r_139189 + neg_res_139195;
                float r_tmp_147447 = zp_res_139196;
                
                r_139189 = r_tmp_147447;
            }
            defunc_0_lifted_lambda_res_139187 = r_139189;
            ((float *) mem_146752)[i_144878] = defunc_0_lifted_lambda_res_139187;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144882 = 0; i_144882 < (int64_t) 16; i_144882++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_lhs_139204 = ((float *) mem_146752)[i_144882];
            
            // futhark/microgpt.fut:272:5-372:64
            
            float zt_rhs_139205 = ((float *) mem_145423)[i_144882];
            
            // futhark/microgpt.fut:356:76-104
            
            float zt_res_139206 = 2.0F * zt_rhs_139205;
            
            // futhark/microgpt.fut:356:62-104
            
            float zs_res_139207 = 1.0F / zt_res_139206;
            
            // futhark/microgpt.fut:356:40-104
            
            float zt_res_139208 = zt_lhs_139204 * zs_res_139207;
            
            ((float *) mem_146759)[i_144882] = zt_res_139208;
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144886 = 0; i_144886 < (int64_t) 16; i_144886++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_lhs_139216 = ((float *) mem_146759)[i_144886];
            
            // futhark/microgpt.fut:357:60-91
            
            float zs_res_139217 = zs_lhs_139216 / 16.0F;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t nest_i_147450 = 0; nest_i_147450 < (int64_t) 16; nest_i_147450++) {
                ((float *) mem_146766)[i_144886 * (int64_t) 16 + nest_i_147450] = zs_res_139217;
            }
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144894 = 0; i_144894 < (int64_t) 16; i_144894++) {
            // futhark/microgpt.fut:272:5-372:64
            
            float zs_rhs_139226 = ((float *) mem_145423)[i_144894];
            
            // futhark/microgpt.fut:358:90-113
            
            float zs_res_139227 = 1.0F / zs_rhs_139226;
            
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144890 = 0; i_144890 < (int64_t) 16; i_144890++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139234 = ((float *) mem_146736)[i_144894 * (int64_t) 16 + i_144890];
                
                // futhark/microgpt.fut:358:62-113
                
                float zt_res_139235 = zs_res_139227 * zt_lhs_139234;
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_lhs_139236 = ((float *) mem_145384)[i_144894 * (int64_t) 16 + i_144890];
                
                // futhark/microgpt.fut:272:5-372:64
                
                float zt_rhs_139237 = ((float *) mem_146766)[i_144894 * (int64_t) 16 + i_144890];
                
                // futhark/microgpt.fut:358:121-168
                
                float zt_res_139238 = zt_lhs_139236 * zt_rhs_139237;
                
                // futhark/microgpt.fut:358:85-168
                
                float zp_res_139239 = zt_res_139235 + zt_res_139238;
                
                // futhark/microgpt.fut:358:116-223
                
                float zp_res_139240 = zt_res_139238 + zp_res_139239;
                
                ((float *) mem_146781)[i_144890] = zp_res_139240;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146776, i_144894 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146781, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144907 = 0; i_144907 < (int64_t) 16; i_144907++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144900 = 0; i_144900 < (int64_t) 16; i_144900++) {
                // futhark/microgpt.fut:272:5-372:64
                
                float lifted_lambda_res_143800 = ((float *) mem_146776)[i_144907 * (int64_t) 16 + i_144900];
                
                ((float *) mem_146802)[i_144900] = lifted_lambda_res_143800;
                ((float *) mem_146803)[i_144900] = lifted_lambda_res_143800;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146792, i_144907 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146802, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146793, i_144907 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146803, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144916 = 0; i_144916 < (int64_t) 64; i_144916++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144912 = 0; i_144912 < (int64_t) 16; i_144912++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_139354;
                float r_139356 = 0.0F;
                
                for (int64_t i_139355 = 0; i_139355 < (int64_t) 16; i_139355++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_139357 = ((float *) mem_146156)[i_139355 * (int64_t) 64 + i_144916];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_139358 = ((float *) mem_145929)[i_139355 * (int64_t) 16 + i_144912];
                    
                    // futhark/microgpt.fut:365:80-129
                    
                    float zt_res_139359 = zt_lhs_139357 * zt_rhs_139358;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_139360 = r_139356 + zt_res_139359;
                    float r_tmp_147459 = zp_res_139360;
                    
                    r_139356 = r_tmp_147459;
                }
                defunc_0_lifted_lambda_res_139354 = r_139356;
                ((float *) mem_146829)[i_144912] = defunc_0_lifted_lambda_res_139354;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146824, i_144916 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146829, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:4:11-25
        for (int64_t i_144929 = 0; i_144929 < (int64_t) 27; i_144929++) {
            // futhark/microgpt.fut:4:11-25
            for (int64_t i_144922 = 0; i_144922 < (int64_t) 16; i_144922++) {
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143828;
                float r_143830 = 0.0F;
                
                for (int64_t i_143829 = 0; i_143829 < (int64_t) 16; i_143829++) {
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_lhs_143831 = ((float *) mem_146092)[i_143829 * (int64_t) 27 + i_144929];
                    
                    // futhark/microgpt.fut:272:5-372:64
                    
                    float zt_rhs_143832 = ((float *) mem_145993)[i_143829 * (int64_t) 16 + i_144922];
                    
                    // futhark/microgpt.fut:367:81-130
                    
                    float zt_res_143833 = zt_lhs_143831 * zt_rhs_143832;
                    
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143834 = r_143830 + zt_res_143833;
                    float r_tmp_147464 = zp_res_143834;
                    
                    r_143830 = r_tmp_147464;
                }
                defunc_0_lifted_lambda_res_143828 = r_143830;
                // futhark/microgpt.fut:71:13-49
                
                float defunc_0_lifted_lambda_res_143837;
                float r_143839 = 0.0F;
                
                for (int64_t i_143838 = 0; i_143838 < (int64_t) 16; i_143838++) {
                    int64_t zeze_lhs_143840 = ((int64_t *) seqs_mem_145240.mem)[step_137620 * (int64_t) 16 + i_143838];
                    
                    // futhark/microgpt.fut:489:61-114
                    
                    bool cond_143841 = zeze_lhs_143840 == i_144929;
                    
                    // futhark/microgpt.fut:489:61-114
                    
                    float lifted_lambda_res_143842;
                    
                    if (cond_143841) {
                        // futhark/microgpt.fut:514:11-50
                        
                        float lifted_lambda_res_t_res_144157 = ((float *) mem_146792)[i_143838 * (int64_t) 16 + i_144922];
                        
                        lifted_lambda_res_143842 = lifted_lambda_res_t_res_144157;
                    } else {
                        lifted_lambda_res_143842 = 0.0F;
                    }
                    // futhark/microgpt.fut:71:40-49
                    
                    float zp_res_143848 = r_143839 + lifted_lambda_res_143842;
                    float r_tmp_147465 = zp_res_143848;
                    
                    r_143839 = r_tmp_147465;
                }
                defunc_0_lifted_lambda_res_143837 = r_143839;
                ((float *) mem_146850)[i_144922] = defunc_0_lifted_lambda_res_143837;
                ((float *) mem_146851)[i_144922] = defunc_0_lifted_lambda_res_143828;
            }
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146840, i_144929 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146850, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
            lmad_copy_4b(ctx, 1, (uint32_t *) mem_146841, i_144929 * (int64_t) 16, (int64_t []) {(int64_t) 1}, (uint32_t *) mem_146851, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 16});
        }
        // futhark/microgpt.fut:66:26-45
        
        float i64_res_139438 = sitofp_i64_f32(step_137620);
        
        // futhark/microgpt.fut:444:46-84
        
        float zm_rhs_139439 = i64_res_139438 / 2.0F;
        
        // futhark/microgpt.fut:444:24-84
        
        float zt_rhs_139440 = 1.0F - zm_rhs_139439;
        
        // futhark/microgpt.fut:444:19-84
        
        float lt_r_139441 = 1.0e-2F * zt_rhs_139440;
        
        // futhark/microgpt.fut:446:5-52
        if (memblock_alloc(ctx, &mem_146872, (int64_t) 1728, "mem_146872")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:446:5-52
        // futhark/microgpt.fut:446:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146872.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145264.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:446:5-52
        if (memblock_alloc(ctx, &mem_146874, (int64_t) 1728, "mem_146874")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:446:5-52
        // futhark/microgpt.fut:446:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146874.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145300.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:446:5-52
        if (memblock_alloc(ctx, &mem_146876, (int64_t) 1728, "mem_146876")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:446:5-52
        // futhark/microgpt.fut:446:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146876.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145336.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:446:5-52
        if (memblock_alloc(ctx, &mem_146878, (int64_t) 1728, "mem_146878")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:446:5-52
        // futhark/microgpt.fut:446:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146878.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146840, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:446:5-52
        if (futrts_adam_opt_w_12790(ctx, &ext_mem_146882, &ext_mem_146881, &ext_mem_146880, mem_146872, mem_146874, mem_146876, mem_146878, (int64_t) 27, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146872, "mem_146872") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146874, "mem_146874") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146876, "mem_146876") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146878, "mem_146878") != 0)
            return 1;
        // futhark/microgpt.fut:448:5-52
        if (memblock_alloc(ctx, &mem_146883, (int64_t) 1024, "mem_146883")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:448:5-52
        // futhark/microgpt.fut:448:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146883.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145256.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:448:5-52
        if (memblock_alloc(ctx, &mem_146885, (int64_t) 1024, "mem_146885")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:448:5-52
        // futhark/microgpt.fut:448:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146885.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145292.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:448:5-52
        if (memblock_alloc(ctx, &mem_146887, (int64_t) 1024, "mem_146887")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:448:5-52
        // futhark/microgpt.fut:448:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146887.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145328.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:448:5-52
        if (memblock_alloc(ctx, &mem_146889, (int64_t) 1024, "mem_146889")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:448:5-52
        // futhark/microgpt.fut:448:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146889.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146793, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:448:5-52
        if (futrts_adam_opt_w_12791(ctx, &ext_mem_146893, &ext_mem_146892, &ext_mem_146891, mem_146883, mem_146885, mem_146887, mem_146889, (int64_t) 16, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146883, "mem_146883") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146885, "mem_146885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146887, "mem_146887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146889, "mem_146889") != 0)
            return 1;
        // futhark/microgpt.fut:450:5-56
        if (memblock_alloc(ctx, &mem_146894, (int64_t) 1024, "mem_146894")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:450:5-56
        // futhark/microgpt.fut:450:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146894.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145260.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:450:5-56
        if (memblock_alloc(ctx, &mem_146896, (int64_t) 1024, "mem_146896")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:450:5-56
        // futhark/microgpt.fut:450:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146896.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145296.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:450:5-56
        if (memblock_alloc(ctx, &mem_146898, (int64_t) 1024, "mem_146898")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:450:5-56
        // futhark/microgpt.fut:450:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146898.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145332.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:450:5-56
        if (memblock_alloc(ctx, &mem_146900, (int64_t) 1024, "mem_146900")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:450:5-56
        // futhark/microgpt.fut:450:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146900.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146650, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:450:5-56
        if (futrts_adam_opt_w_12791(ctx, &ext_mem_146904, &ext_mem_146903, &ext_mem_146902, mem_146894, mem_146896, mem_146898, mem_146900, (int64_t) 16, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146894, "mem_146894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146896, "mem_146896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146898, "mem_146898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146900, "mem_146900") != 0)
            return 1;
        // futhark/microgpt.fut:452:5-56
        if (memblock_alloc(ctx, &mem_146905, (int64_t) 1024, "mem_146905")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:452:5-56
        // futhark/microgpt.fut:452:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146905.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145248.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:452:5-56
        if (memblock_alloc(ctx, &mem_146907, (int64_t) 1024, "mem_146907")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:452:5-56
        // futhark/microgpt.fut:452:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146907.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145284.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:452:5-56
        if (memblock_alloc(ctx, &mem_146909, (int64_t) 1024, "mem_146909")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:452:5-56
        // futhark/microgpt.fut:452:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146909.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145320.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:452:5-56
        if (memblock_alloc(ctx, &mem_146911, (int64_t) 1024, "mem_146911")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:452:5-56
        // futhark/microgpt.fut:452:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146911.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146649, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:452:5-56
        if (futrts_adam_opt_w_12791(ctx, &ext_mem_146915, &ext_mem_146914, &ext_mem_146913, mem_146905, mem_146907, mem_146909, mem_146911, (int64_t) 16, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146905, "mem_146905") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146907, "mem_146907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146909, "mem_146909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146911, "mem_146911") != 0)
            return 1;
        // futhark/microgpt.fut:454:5-56
        if (memblock_alloc(ctx, &mem_146916, (int64_t) 1024, "mem_146916")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:454:5-56
        // futhark/microgpt.fut:454:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146916.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145272.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:454:5-56
        if (memblock_alloc(ctx, &mem_146918, (int64_t) 1024, "mem_146918")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:454:5-56
        // futhark/microgpt.fut:454:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146918.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145308.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:454:5-56
        if (memblock_alloc(ctx, &mem_146920, (int64_t) 1024, "mem_146920")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:454:5-56
        // futhark/microgpt.fut:454:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146920.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145344.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:454:5-56
        if (memblock_alloc(ctx, &mem_146922, (int64_t) 1024, "mem_146922")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:454:5-56
        // futhark/microgpt.fut:454:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146922.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146648, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:454:5-56
        if (futrts_adam_opt_w_12791(ctx, &ext_mem_146926, &ext_mem_146925, &ext_mem_146924, mem_146916, mem_146918, mem_146920, mem_146922, (int64_t) 16, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146916, "mem_146916") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146918, "mem_146918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146920, "mem_146920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146922, "mem_146922") != 0)
            return 1;
        // futhark/microgpt.fut:456:5-56
        if (memblock_alloc(ctx, &mem_146927, (int64_t) 1024, "mem_146927")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:456:5-56
        // futhark/microgpt.fut:456:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146927.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145252.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:456:5-56
        if (memblock_alloc(ctx, &mem_146929, (int64_t) 1024, "mem_146929")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:456:5-56
        // futhark/microgpt.fut:456:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146929.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145288.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:456:5-56
        if (memblock_alloc(ctx, &mem_146931, (int64_t) 1024, "mem_146931")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:456:5-56
        // futhark/microgpt.fut:456:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146931.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145324.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:456:5-56
        if (memblock_alloc(ctx, &mem_146933, (int64_t) 1024, "mem_146933")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:456:5-56
        // futhark/microgpt.fut:456:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146933.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146228, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 16});
        // futhark/microgpt.fut:456:5-56
        if (futrts_adam_opt_w_12791(ctx, &ext_mem_146937, &ext_mem_146936, &ext_mem_146935, mem_146927, mem_146929, mem_146931, mem_146933, (int64_t) 16, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146927, "mem_146927") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146929, "mem_146929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146931, "mem_146931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146933, "mem_146933") != 0)
            return 1;
        // futhark/microgpt.fut:458:5-52
        if (memblock_alloc(ctx, &mem_146938, (int64_t) 4096, "mem_146938")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:458:5-52
        // futhark/microgpt.fut:458:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146938.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145268.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:458:5-52
        if (memblock_alloc(ctx, &mem_146940, (int64_t) 4096, "mem_146940")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:458:5-52
        // futhark/microgpt.fut:458:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146940.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145304.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:458:5-52
        if (memblock_alloc(ctx, &mem_146942, (int64_t) 4096, "mem_146942")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:458:5-52
        // futhark/microgpt.fut:458:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146942.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145340.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:458:5-52
        if (memblock_alloc(ctx, &mem_146944, (int64_t) 4096, "mem_146944")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:458:5-52
        // futhark/microgpt.fut:458:5-52
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146944.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146824, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 64, (int64_t) 16});
        // futhark/microgpt.fut:458:5-52
        if (futrts_adam_opt_w_12790(ctx, &ext_mem_146948, &ext_mem_146947, &ext_mem_146946, mem_146938, mem_146940, mem_146942, mem_146944, (int64_t) 64, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146938, "mem_146938") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146940, "mem_146940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146942, "mem_146942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146944, "mem_146944") != 0)
            return 1;
        // futhark/microgpt.fut:460:5-60
        if (memblock_alloc(ctx, &mem_146949, (int64_t) 4096, "mem_146949")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:460:5-60
        // futhark/microgpt.fut:460:5-60
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146949.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint32_t *) mem_param_145244.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:460:5-60
        if (memblock_alloc(ctx, &mem_146951, (int64_t) 4096, "mem_146951")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:460:5-60
        // futhark/microgpt.fut:460:5-60
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146951.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint32_t *) mem_param_145280.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:460:5-60
        if (memblock_alloc(ctx, &mem_146953, (int64_t) 4096, "mem_146953")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:460:5-60
        // futhark/microgpt.fut:460:5-60
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146953.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint32_t *) mem_param_145316.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:460:5-60
        if (memblock_alloc(ctx, &mem_146955, (int64_t) 4096, "mem_146955")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:460:5-60
        // futhark/microgpt.fut:460:5-60
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146955.mem, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (uint32_t *) mem_146124, (int64_t) 0, (int64_t []) {(int64_t) 64, (int64_t) 1}, (int64_t []) {(int64_t) 16, (int64_t) 64});
        // futhark/microgpt.fut:460:5-60
        if (futrts_adam_opt_w_12790(ctx, &ext_mem_146959, &ext_mem_146958, &ext_mem_146957, mem_146949, mem_146951, mem_146953, mem_146955, (int64_t) 16, (int64_t) 64, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146949, "mem_146949") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146951, "mem_146951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146953, "mem_146953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146955, "mem_146955") != 0)
            return 1;
        // futhark/microgpt.fut:462:5-56
        if (memblock_alloc(ctx, &mem_146960, (int64_t) 1728, "mem_146960")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:462:5-56
        // futhark/microgpt.fut:462:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146960.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145276.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:462:5-56
        if (memblock_alloc(ctx, &mem_146962, (int64_t) 1728, "mem_146962")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:462:5-56
        // futhark/microgpt.fut:462:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146962.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145312.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:462:5-56
        if (memblock_alloc(ctx, &mem_146964, (int64_t) 1728, "mem_146964")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:462:5-56
        // futhark/microgpt.fut:462:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146964.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_param_145348.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:462:5-56
        if (memblock_alloc(ctx, &mem_146966, (int64_t) 1728, "mem_146966")) {
            err = 1;
            goto cleanup;
        }
        // futhark/microgpt.fut:462:5-56
        // futhark/microgpt.fut:462:5-56
        lmad_copy_4b(ctx, 2, (uint32_t *) mem_146966.mem, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (uint32_t *) mem_146841, (int64_t) 0, (int64_t []) {(int64_t) 16, (int64_t) 1}, (int64_t []) {(int64_t) 27, (int64_t) 16});
        // futhark/microgpt.fut:462:5-56
        if (futrts_adam_opt_w_12790(ctx, &ext_mem_146970, &ext_mem_146969, &ext_mem_146968, mem_146960, mem_146962, mem_146964, mem_146966, (int64_t) 27, (int64_t) 16, step_137620, lt_r_139441) != 0) {
            err = 1;
            goto cleanup;
        }
        if (memblock_unref(ctx, &mem_146960, "mem_146960") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146962, "mem_146962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146964, "mem_146964") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146966, "mem_146966") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147184, &ext_mem_146959, "ext_mem_146959") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147185, &ext_mem_146915, "ext_mem_146915") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147186, &ext_mem_146937, "ext_mem_146937") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147187, &ext_mem_146893, "ext_mem_146893") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147188, &ext_mem_146904, "ext_mem_146904") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147189, &ext_mem_146882, "ext_mem_146882") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147190, &ext_mem_146948, "ext_mem_146948") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147191, &ext_mem_146926, "ext_mem_146926") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147192, &ext_mem_146970, "ext_mem_146970") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147193, &ext_mem_146958, "ext_mem_146958") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147194, &ext_mem_146914, "ext_mem_146914") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147195, &ext_mem_146936, "ext_mem_146936") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147196, &ext_mem_146892, "ext_mem_146892") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147197, &ext_mem_146903, "ext_mem_146903") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147198, &ext_mem_146881, "ext_mem_146881") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147199, &ext_mem_146947, "ext_mem_146947") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147200, &ext_mem_146925, "ext_mem_146925") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147201, &ext_mem_146969, "ext_mem_146969") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147202, &ext_mem_146957, "ext_mem_146957") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147203, &ext_mem_146913, "ext_mem_146913") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147204, &ext_mem_146935, "ext_mem_146935") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147205, &ext_mem_146891, "ext_mem_146891") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147206, &ext_mem_146902, "ext_mem_146902") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147207, &ext_mem_146880, "ext_mem_146880") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147208, &ext_mem_146946, "ext_mem_146946") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147209, &ext_mem_146924, "ext_mem_146924") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_tmp_147210, &ext_mem_146968, "ext_mem_146968") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145244, &mem_param_tmp_147184, "mem_param_tmp_147184") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145248, &mem_param_tmp_147185, "mem_param_tmp_147185") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145252, &mem_param_tmp_147186, "mem_param_tmp_147186") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145256, &mem_param_tmp_147187, "mem_param_tmp_147187") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145260, &mem_param_tmp_147188, "mem_param_tmp_147188") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145264, &mem_param_tmp_147189, "mem_param_tmp_147189") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145268, &mem_param_tmp_147190, "mem_param_tmp_147190") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145272, &mem_param_tmp_147191, "mem_param_tmp_147191") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145276, &mem_param_tmp_147192, "mem_param_tmp_147192") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145280, &mem_param_tmp_147193, "mem_param_tmp_147193") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145284, &mem_param_tmp_147194, "mem_param_tmp_147194") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145288, &mem_param_tmp_147195, "mem_param_tmp_147195") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145292, &mem_param_tmp_147196, "mem_param_tmp_147196") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145296, &mem_param_tmp_147197, "mem_param_tmp_147197") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145300, &mem_param_tmp_147198, "mem_param_tmp_147198") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145304, &mem_param_tmp_147199, "mem_param_tmp_147199") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145308, &mem_param_tmp_147200, "mem_param_tmp_147200") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145312, &mem_param_tmp_147201, "mem_param_tmp_147201") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145316, &mem_param_tmp_147202, "mem_param_tmp_147202") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145320, &mem_param_tmp_147203, "mem_param_tmp_147203") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145324, &mem_param_tmp_147204, "mem_param_tmp_147204") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145328, &mem_param_tmp_147205, "mem_param_tmp_147205") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145332, &mem_param_tmp_147206, "mem_param_tmp_147206") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145336, &mem_param_tmp_147207, "mem_param_tmp_147207") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145340, &mem_param_tmp_147208, "mem_param_tmp_147208") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145344, &mem_param_tmp_147209, "mem_param_tmp_147209") != 0)
            return 1;
        if (memblock_set(ctx, &mem_param_145348, &mem_param_tmp_147210, "mem_param_tmp_147210") != 0)
            return 1;
    }
    if (memblock_set(ctx, &ext_mem_147078, &mem_param_145244, "mem_param_145244") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147077, &mem_param_145248, "mem_param_145248") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147076, &mem_param_145252, "mem_param_145252") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147075, &mem_param_145256, "mem_param_145256") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147074, &mem_param_145260, "mem_param_145260") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147073, &mem_param_145264, "mem_param_145264") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147072, &mem_param_145268, "mem_param_145268") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147071, &mem_param_145272, "mem_param_145272") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147070, &mem_param_145276, "mem_param_145276") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147069, &mem_param_145280, "mem_param_145280") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147068, &mem_param_145284, "mem_param_145284") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147067, &mem_param_145288, "mem_param_145288") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147066, &mem_param_145292, "mem_param_145292") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147065, &mem_param_145296, "mem_param_145296") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147064, &mem_param_145300, "mem_param_145300") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147063, &mem_param_145304, "mem_param_145304") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147062, &mem_param_145308, "mem_param_145308") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147061, &mem_param_145312, "mem_param_145312") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147060, &mem_param_145316, "mem_param_145316") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147059, &mem_param_145320, "mem_param_145320") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147058, &mem_param_145324, "mem_param_145324") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147057, &mem_param_145328, "mem_param_145328") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147056, &mem_param_145332, "mem_param_145332") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147055, &mem_param_145336, "mem_param_145336") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147054, &mem_param_145340, "mem_param_145340") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147053, &mem_param_145344, "mem_param_145344") != 0)
        return 1;
    if (memblock_set(ctx, &ext_mem_147052, &mem_param_145348, "mem_param_145348") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_145349, "mem_145349") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147157, &ext_mem_147073, "ext_mem_147073") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147158, &ext_mem_147075, "ext_mem_147075") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147159, &ext_mem_147074, "ext_mem_147074") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147160, &ext_mem_147077, "ext_mem_147077") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147161, &ext_mem_147071, "ext_mem_147071") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147162, &ext_mem_147076, "ext_mem_147076") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147163, &ext_mem_147072, "ext_mem_147072") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147164, &ext_mem_147078, "ext_mem_147078") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147165, &ext_mem_147070, "ext_mem_147070") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147166, &ext_mem_147064, "ext_mem_147064") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147167, &ext_mem_147066, "ext_mem_147066") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147168, &ext_mem_147065, "ext_mem_147065") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147169, &ext_mem_147068, "ext_mem_147068") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147170, &ext_mem_147062, "ext_mem_147062") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147171, &ext_mem_147067, "ext_mem_147067") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147172, &ext_mem_147063, "ext_mem_147063") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147173, &ext_mem_147069, "ext_mem_147069") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147174, &ext_mem_147061, "ext_mem_147061") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147175, &ext_mem_147055, "ext_mem_147055") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147176, &ext_mem_147057, "ext_mem_147057") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147177, &ext_mem_147056, "ext_mem_147056") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147178, &ext_mem_147059, "ext_mem_147059") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147179, &ext_mem_147053, "ext_mem_147053") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147180, &ext_mem_147058, "ext_mem_147058") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147181, &ext_mem_147054, "ext_mem_147054") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147182, &ext_mem_147060, "ext_mem_147060") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147183, &ext_mem_147052, "ext_mem_147052") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147637, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147638, &mem_out_147158, "mem_out_147158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147639, &mem_out_147159, "mem_out_147159") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147640, &mem_out_147160, "mem_out_147160") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147641, &mem_out_147161, "mem_out_147161") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147642, &mem_out_147162, "mem_out_147162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147643, &mem_out_147163, "mem_out_147163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147644, &mem_out_147164, "mem_out_147164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147645, &mem_out_147165, "mem_out_147165") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147646, &mem_out_147166, "mem_out_147166") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147647, &mem_out_147167, "mem_out_147167") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147648, &mem_out_147168, "mem_out_147168") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147649, &mem_out_147169, "mem_out_147169") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147650, &mem_out_147170, "mem_out_147170") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147651, &mem_out_147171, "mem_out_147171") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147652, &mem_out_147172, "mem_out_147172") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147653, &mem_out_147173, "mem_out_147173") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147654, &mem_out_147174, "mem_out_147174") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147655, &mem_out_147175, "mem_out_147175") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147656, &mem_out_147176, "mem_out_147176") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147657, &mem_out_147177, "mem_out_147177") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147658, &mem_out_147178, "mem_out_147178") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147659, &mem_out_147179, "mem_out_147179") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147660, &mem_out_147180, "mem_out_147180") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147661, &mem_out_147181, "mem_out_147181") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147662, &mem_out_147182, "mem_out_147182") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147663, &mem_out_147183, "mem_out_147183") != 0)
        return 1;
    
  cleanup:
    {
        free(mem_145352);
        free(mem_145353);
        free(mem_145362);
        free(mem_145369);
        free(mem_145384);
        free(mem_145389);
        free(mem_145400);
        free(mem_145405);
        free(mem_145416);
        free(mem_145423);
        free(mem_145430);
        free(mem_145435);
        free(mem_145446);
        free(mem_145451);
        free(mem_145462);
        free(mem_145469);
        free(mem_145476);
        free(mem_145481);
        free(mem_145492);
        free(mem_145493);
        free(mem_145494);
        free(mem_145507);
        free(mem_145508);
        free(mem_145509);
        free(mem_145540);
        free(mem_145541);
        free(mem_145542);
        free(mem_145558);
        free(mem_145559);
        free(mem_145560);
        free(mem_145573);
        free(mem_145574);
        free(mem_145575);
        free(mem_145621);
        free(mem_145622);
        free(mem_145623);
        free(mem_145639);
        free(mem_145640);
        free(mem_145641);
        free(mem_145654);
        free(mem_145655);
        free(mem_145656);
        free(mem_145702);
        free(mem_145708);
        free(mem_145713);
        free(mem_145729);
        free(mem_145735);
        free(mem_145740);
        free(mem_145756);
        free(mem_145762);
        free(mem_145767);
        free(mem_145774);
        free(mem_145781);
        free(mem_145797);
        free(mem_145803);
        free(mem_145808);
        free(mem_145824);
        free(mem_145830);
        free(mem_145835);
        free(mem_145851);
        free(mem_145856);
        free(mem_145867);
        free(mem_145872);
        free(mem_145883);
        free(mem_145888);
        free(mem_145899);
        free(mem_145904);
        free(mem_145915);
        free(mem_145922);
        free(mem_145929);
        free(mem_145934);
        free(mem_145945);
        free(mem_145950);
        free(mem_145961);
        free(mem_145966);
        free(mem_145977);
        free(mem_145982);
        free(mem_145993);
        free(mem_145998);
        free(mem_146009);
        free(mem_146014);
        free(mem_146025);
        free(mem_146026);
        free(mem_146035);
        free(mem_146036);
        free(mem_146049);
        free(mem_146050);
        free(mem_146063);
        free(mem_146064);
        free(mem_146085);
        free(mem_146092);
        free(mem_146097);
        free(mem_146108);
        free(mem_146113);
        free(mem_146124);
        free(mem_146125);
        free(mem_146134);
        free(mem_146135);
        free(mem_146156);
        free(mem_146161);
        free(mem_146172);
        free(mem_146177);
        free(mem_146188);
        free(mem_146195);
        free(mem_146202);
        free(mem_146212);
        free(mem_146217);
        free(mem_146228);
        free(mem_146229);
        free(mem_146238);
        free(mem_146239);
        free(mem_146260);
        free(mem_146266);
        free(mem_146271);
        free(mem_146287);
        free(mem_146293);
        free(mem_146298);
        free(mem_146314);
        free(mem_146315);
        free(mem_146326);
        free(mem_146327);
        free(mem_146336);
        free(mem_146343);
        free(mem_146368);
        free(mem_146374);
        free(mem_146379);
        free(mem_146395);
        free(mem_146400);
        free(mem_146411);
        free(mem_146417);
        free(mem_146422);
        free(mem_146438);
        free(mem_146444);
        free(mem_146449);
        free(mem_146465);
        free(mem_146466);
        free(mem_146477);
        free(mem_146478);
        free(mem_146487);
        free(mem_146488);
        free(mem_146519);
        free(mem_146520);
        free(mem_146521);
        free(mem_146537);
        free(mem_146538);
        free(mem_146539);
        free(mem_146552);
        free(mem_146553);
        free(mem_146554);
        free(mem_146600);
        free(mem_146601);
        free(mem_146602);
        free(mem_146615);
        free(mem_146616);
        free(mem_146617);
        free(mem_146648);
        free(mem_146649);
        free(mem_146650);
        free(mem_146651);
        free(mem_146668);
        free(mem_146669);
        free(mem_146670);
        free(mem_146671);
        free(mem_146712);
        free(mem_146719);
        free(mem_146726);
        free(mem_146736);
        free(mem_146741);
        free(mem_146752);
        free(mem_146759);
        free(mem_146766);
        free(mem_146776);
        free(mem_146781);
        free(mem_146792);
        free(mem_146793);
        free(mem_146802);
        free(mem_146803);
        free(mem_146824);
        free(mem_146829);
        free(mem_146840);
        free(mem_146841);
        free(mem_146850);
        free(mem_146851);
        if (memblock_unref(ctx, &mem_param_tmp_147210, "mem_param_tmp_147210") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147209, "mem_param_tmp_147209") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147208, "mem_param_tmp_147208") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147207, "mem_param_tmp_147207") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147206, "mem_param_tmp_147206") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147205, "mem_param_tmp_147205") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147204, "mem_param_tmp_147204") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147203, "mem_param_tmp_147203") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147202, "mem_param_tmp_147202") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147201, "mem_param_tmp_147201") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147200, "mem_param_tmp_147200") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147199, "mem_param_tmp_147199") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147198, "mem_param_tmp_147198") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147197, "mem_param_tmp_147197") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147196, "mem_param_tmp_147196") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147195, "mem_param_tmp_147195") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147194, "mem_param_tmp_147194") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147193, "mem_param_tmp_147193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147192, "mem_param_tmp_147192") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147191, "mem_param_tmp_147191") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147190, "mem_param_tmp_147190") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147189, "mem_param_tmp_147189") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147188, "mem_param_tmp_147188") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147187, "mem_param_tmp_147187") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147186, "mem_param_tmp_147186") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147185, "mem_param_tmp_147185") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_tmp_147184, "mem_param_tmp_147184") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146968, "ext_mem_146968") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146969, "ext_mem_146969") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146970, "ext_mem_146970") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146966, "mem_146966") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146964, "mem_146964") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146962, "mem_146962") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146960, "mem_146960") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146957, "ext_mem_146957") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146958, "ext_mem_146958") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146959, "ext_mem_146959") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146955, "mem_146955") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146953, "mem_146953") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146951, "mem_146951") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146949, "mem_146949") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146946, "ext_mem_146946") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146947, "ext_mem_146947") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146948, "ext_mem_146948") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146944, "mem_146944") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146942, "mem_146942") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146940, "mem_146940") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146938, "mem_146938") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146935, "ext_mem_146935") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146936, "ext_mem_146936") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146937, "ext_mem_146937") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146933, "mem_146933") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146931, "mem_146931") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146929, "mem_146929") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146927, "mem_146927") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146924, "ext_mem_146924") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146925, "ext_mem_146925") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146926, "ext_mem_146926") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146922, "mem_146922") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146920, "mem_146920") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146918, "mem_146918") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146916, "mem_146916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146913, "ext_mem_146913") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146914, "ext_mem_146914") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146915, "ext_mem_146915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146911, "mem_146911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146909, "mem_146909") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146907, "mem_146907") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146905, "mem_146905") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146902, "ext_mem_146902") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146903, "ext_mem_146903") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146904, "ext_mem_146904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146900, "mem_146900") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146898, "mem_146898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146896, "mem_146896") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146894, "mem_146894") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146891, "ext_mem_146891") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146892, "ext_mem_146892") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146893, "ext_mem_146893") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146889, "mem_146889") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146887, "mem_146887") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146885, "mem_146885") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146883, "mem_146883") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146880, "ext_mem_146880") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146881, "ext_mem_146881") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_146882, "ext_mem_146882") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146878, "mem_146878") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146876, "mem_146876") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146874, "mem_146874") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_146872, "mem_146872") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_145351, "ext_mem_145351") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145348, "mem_param_145348") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145344, "mem_param_145344") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145340, "mem_param_145340") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145336, "mem_param_145336") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145332, "mem_param_145332") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145328, "mem_param_145328") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145324, "mem_param_145324") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145320, "mem_param_145320") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145316, "mem_param_145316") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145312, "mem_param_145312") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145308, "mem_param_145308") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145304, "mem_param_145304") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145300, "mem_param_145300") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145296, "mem_param_145296") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145292, "mem_param_145292") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145288, "mem_param_145288") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145284, "mem_param_145284") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145280, "mem_param_145280") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145276, "mem_param_145276") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145272, "mem_param_145272") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145268, "mem_param_145268") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145264, "mem_param_145264") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145260, "mem_param_145260") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145256, "mem_param_145256") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145252, "mem_param_145252") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145248, "mem_param_145248") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_145244, "mem_param_145244") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147052, "ext_mem_147052") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147053, "ext_mem_147053") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147054, "ext_mem_147054") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147055, "ext_mem_147055") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147056, "ext_mem_147056") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147057, "ext_mem_147057") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147058, "ext_mem_147058") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147059, "ext_mem_147059") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147060, "ext_mem_147060") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147061, "ext_mem_147061") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147062, "ext_mem_147062") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147063, "ext_mem_147063") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147064, "ext_mem_147064") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147065, "ext_mem_147065") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147066, "ext_mem_147066") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147067, "ext_mem_147067") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147068, "ext_mem_147068") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147069, "ext_mem_147069") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147070, "ext_mem_147070") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147071, "ext_mem_147071") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147072, "ext_mem_147072") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147073, "ext_mem_147073") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147074, "ext_mem_147074") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147075, "ext_mem_147075") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147076, "ext_mem_147076") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147077, "ext_mem_147077") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_147078, "ext_mem_147078") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_145349, "mem_145349") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147183, "mem_out_147183") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147182, "mem_out_147182") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147181, "mem_out_147181") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147180, "mem_out_147180") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147179, "mem_out_147179") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147178, "mem_out_147178") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147177, "mem_out_147177") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147176, "mem_out_147176") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147175, "mem_out_147175") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147174, "mem_out_147174") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147173, "mem_out_147173") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147172, "mem_out_147172") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147171, "mem_out_147171") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147170, "mem_out_147170") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147169, "mem_out_147169") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147168, "mem_out_147168") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147167, "mem_out_147167") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147166, "mem_out_147166") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147165, "mem_out_147165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147164, "mem_out_147164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147163, "mem_out_147163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147162, "mem_out_147162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147161, "mem_out_147161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147160, "mem_out_147160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147159, "mem_out_147159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147158, "mem_out_147158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_zzero_params(struct futhark_context *ctx, struct memblock *mem_out_p_147847, struct memblock *mem_out_p_147848, struct memblock *mem_out_p_147849, struct memblock *mem_out_p_147850, struct memblock *mem_out_p_147851, struct memblock *mem_out_p_147852, struct memblock *mem_out_p_147853, struct memblock *mem_out_p_147854, struct memblock *mem_out_p_147855)
{
    (void) ctx;
    
    int err = 0;
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock mem_145202 = ctx->constants->mem_145202;
    struct memblock mem_145203 = ctx->constants->mem_145203;
    struct memblock mem_145204 = ctx->constants->mem_145204;
    struct memblock mem_145205 = ctx->constants->mem_145205;
    struct memblock mem_145206 = ctx->constants->mem_145206;
    struct memblock mem_145207 = ctx->constants->mem_145207;
    struct memblock mem_145208 = ctx->constants->mem_145208;
    struct memblock mem_145209 = ctx->constants->mem_145209;
    struct memblock mem_145210 = ctx->constants->mem_145210;
    
    if (memblock_set(ctx, &mem_out_147157, &mem_145209, "mem_145209") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147158, &mem_145205, "mem_145205") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147159, &mem_145207, "mem_145207") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147160, &mem_145203, "mem_145203") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147161, &mem_145204, "mem_145204") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147162, &mem_145202, "mem_145202") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147163, &mem_145208, "mem_145208") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147164, &mem_145206, "mem_145206") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_147165, &mem_145210, "mem_145210") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147847, &mem_out_147157, "mem_out_147157") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147848, &mem_out_147158, "mem_out_147158") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147849, &mem_out_147159, "mem_out_147159") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147850, &mem_out_147160, "mem_out_147160") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147851, &mem_out_147161, "mem_out_147161") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147852, &mem_out_147162, "mem_out_147162") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147853, &mem_out_147163, "mem_out_147163") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147854, &mem_out_147164, "mem_out_147164") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_147855, &mem_out_147165, "mem_out_147165") != 0)
        return 1;
    
  cleanup:
    {
        if (memblock_unref(ctx, &mem_out_147165, "mem_out_147165") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147164, "mem_out_147164") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147163, "mem_out_147163") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147162, "mem_out_147162") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147161, "mem_out_147161") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147160, "mem_out_147160") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147159, "mem_out_147159") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147158, "mem_out_147158") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_147157, "mem_out_147157") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_forward(struct futhark_context *ctx, struct futhark_f32_3d **out, const struct futhark_opaque_params *in0, const struct futhark_i64_2d *in1, const struct futhark_f32_3d *in2)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock masks_mem_145221;
    
    masks_mem_145221.references = NULL;
    
    struct memblock seqs_mem_145220;
    
    seqs_mem_145220.references = NULL;
    
    struct memblock wvoc_mem_145219;
    
    wvoc_mem_145219.references = NULL;
    
    struct memblock wval_mem_145218;
    
    wval_mem_145218.references = NULL;
    
    struct memblock wup_mem_145217;
    
    wup_mem_145217.references = NULL;
    
    struct memblock wte_mem_145216;
    
    wte_mem_145216.references = NULL;
    
    struct memblock wqry_mem_145215;
    
    wqry_mem_145215.references = NULL;
    
    struct memblock wpe_mem_145214;
    
    wpe_mem_145214.references = NULL;
    
    struct memblock wout_mem_145213;
    
    wout_mem_145213.references = NULL;
    
    struct memblock wkey_mem_145212;
    
    wkey_mem_145212.references = NULL;
    
    struct memblock wdown_mem_145211;
    
    wdown_mem_145211.references = NULL;
    wdown_mem_145211 = in0->v0->mem;
    wkey_mem_145212 = in0->v1->mem;
    wout_mem_145213 = in0->v2->mem;
    wpe_mem_145214 = in0->v3->mem;
    wqry_mem_145215 = in0->v4->mem;
    wte_mem_145216 = in0->v5->mem;
    wup_mem_145217 = in0->v6->mem;
    wval_mem_145218 = in0->v7->mem;
    wvoc_mem_145219 = in0->v8->mem;
    seqs_mem_145220 = in1->mem;
    masks_mem_145221 = in2->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 1 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && ((int64_t) 1 == in2->shape[0] && ((int64_t) 16 == in2->shape[1] && (int64_t) 16 == in2->shape[2]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_forward(ctx, &mem_out_147157, wdown_mem_145211, wkey_mem_145212, wout_mem_145213, wpe_mem_145214, wqry_mem_145215, wte_mem_145216, wup_mem_145217, wval_mem_145218, wvoc_mem_145219, seqs_mem_145220, masks_mem_145221);
        if (ret == 0) {
            struct memblock mem_145202 = ctx->constants->mem_145202;
            struct memblock mem_145203 = ctx->constants->mem_145203;
            struct memblock mem_145204 = ctx->constants->mem_145204;
            struct memblock mem_145205 = ctx->constants->mem_145205;
            struct memblock mem_145206 = ctx->constants->mem_145206;
            struct memblock mem_145207 = ctx->constants->mem_145207;
            struct memblock mem_145208 = ctx->constants->mem_145208;
            struct memblock mem_145209 = ctx->constants->mem_145209;
            struct memblock mem_145210 = ctx->constants->mem_145210;
            
            assert((*out = (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) != NULL);
            (*out)->mem = mem_out_147157;
            (*out)->shape[0] = (int64_t) 1;
            (*out)->shape[1] = (int64_t) 16;
            (*out)->shape[2] = (int64_t) 27;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_loss(struct futhark_context *ctx, float *out, const int64_t in0, const struct futhark_opaque_params *in1, const struct futhark_i64_2d *in2, const struct futhark_f32_3d *in3)
{
    int64_t dl_83939 = (int64_t) 0;
    float prim_out_147157 = 0.0F;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock masks_mem_145221;
    
    masks_mem_145221.references = NULL;
    
    struct memblock seqs_mem_145220;
    
    seqs_mem_145220.references = NULL;
    
    struct memblock wvoc_mem_145219;
    
    wvoc_mem_145219.references = NULL;
    
    struct memblock wval_mem_145218;
    
    wval_mem_145218.references = NULL;
    
    struct memblock wup_mem_145217;
    
    wup_mem_145217.references = NULL;
    
    struct memblock wte_mem_145216;
    
    wte_mem_145216.references = NULL;
    
    struct memblock wqry_mem_145215;
    
    wqry_mem_145215.references = NULL;
    
    struct memblock wpe_mem_145214;
    
    wpe_mem_145214.references = NULL;
    
    struct memblock wout_mem_145213;
    
    wout_mem_145213.references = NULL;
    
    struct memblock wkey_mem_145212;
    
    wkey_mem_145212.references = NULL;
    
    struct memblock wdown_mem_145211;
    
    wdown_mem_145211.references = NULL;
    dl_83939 = in0;
    wdown_mem_145211 = in1->v0->mem;
    wkey_mem_145212 = in1->v1->mem;
    wout_mem_145213 = in1->v2->mem;
    wpe_mem_145214 = in1->v3->mem;
    wqry_mem_145215 = in1->v4->mem;
    wte_mem_145216 = in1->v5->mem;
    wup_mem_145217 = in1->v6->mem;
    wval_mem_145218 = in1->v7->mem;
    wvoc_mem_145219 = in1->v8->mem;
    seqs_mem_145220 = in2->mem;
    masks_mem_145221 = in3->mem;
    if (!(((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 1 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && ((int64_t) 1 == in3->shape[0] && ((int64_t) 16 == in3->shape[1] && (int64_t) 16 == in3->shape[2]))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_loss(ctx, &prim_out_147157, wdown_mem_145211, wkey_mem_145212, wout_mem_145213, wpe_mem_145214, wqry_mem_145215, wte_mem_145216, wup_mem_145217, wval_mem_145218, wvoc_mem_145219, seqs_mem_145220, masks_mem_145221, dl_83939);
        if (ret == 0) {
            struct memblock mem_145202 = ctx->constants->mem_145202;
            struct memblock mem_145203 = ctx->constants->mem_145203;
            struct memblock mem_145204 = ctx->constants->mem_145204;
            struct memblock mem_145205 = ctx->constants->mem_145205;
            struct memblock mem_145206 = ctx->constants->mem_145206;
            struct memblock mem_145207 = ctx->constants->mem_145207;
            struct memblock mem_145208 = ctx->constants->mem_145208;
            struct memblock mem_145209 = ctx->constants->mem_145209;
            struct memblock mem_145210 = ctx->constants->mem_145210;
            
            *out = prim_out_147157;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_to_params(struct futhark_context *ctx, struct futhark_opaque_params **out, const struct futhark_f32_2d *in0, const struct futhark_f32_2d *in1, const struct futhark_f32_2d *in2, const struct futhark_f32_2d *in3, const struct futhark_f32_2d *in4, const struct futhark_f32_2d *in5, const struct futhark_f32_2d *in6, const struct futhark_f32_2d *in7, const struct futhark_f32_2d *in8)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock wvoc_mem_145219;
    
    wvoc_mem_145219.references = NULL;
    
    struct memblock wdown_mem_145218;
    
    wdown_mem_145218.references = NULL;
    
    struct memblock wup_mem_145217;
    
    wup_mem_145217.references = NULL;
    
    struct memblock wout_mem_145216;
    
    wout_mem_145216.references = NULL;
    
    struct memblock wval_mem_145215;
    
    wval_mem_145215.references = NULL;
    
    struct memblock wkey_mem_145214;
    
    wkey_mem_145214.references = NULL;
    
    struct memblock wqry_mem_145213;
    
    wqry_mem_145213.references = NULL;
    
    struct memblock wpe_mem_145212;
    
    wpe_mem_145212.references = NULL;
    
    struct memblock wte_mem_145211;
    
    wte_mem_145211.references = NULL;
    wte_mem_145211 = in0->mem;
    wpe_mem_145212 = in1->mem;
    wqry_mem_145213 = in2->mem;
    wkey_mem_145214 = in3->mem;
    wval_mem_145215 = in4->mem;
    wout_mem_145216 = in5->mem;
    wup_mem_145217 = in6->mem;
    wdown_mem_145218 = in7->mem;
    wvoc_mem_145219 = in8->mem;
    if (!(((int64_t) 27 == in0->shape[0] && (int64_t) 16 == in0->shape[1]) && (((int64_t) 16 == in1->shape[0] && (int64_t) 16 == in1->shape[1]) && (((int64_t) 16 == in2->shape[0] && (int64_t) 16 == in2->shape[1]) && (((int64_t) 16 == in3->shape[0] && (int64_t) 16 == in3->shape[1]) && (((int64_t) 16 == in4->shape[0] && (int64_t) 16 == in4->shape[1]) && (((int64_t) 16 == in5->shape[0] && (int64_t) 16 == in5->shape[1]) && (((int64_t) 64 == in6->shape[0] && (int64_t) 16 == in6->shape[1]) && (((int64_t) 16 == in7->shape[0] && (int64_t) 64 == in7->shape[1]) && ((int64_t) 27 == in8->shape[0] && (int64_t) 16 == in8->shape[1])))))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_to_params(ctx, &mem_out_147157, &mem_out_147158, &mem_out_147159, &mem_out_147160, &mem_out_147161, &mem_out_147162, &mem_out_147163, &mem_out_147164, &mem_out_147165, wte_mem_145211, wpe_mem_145212, wqry_mem_145213, wkey_mem_145214, wval_mem_145215, wout_mem_145216, wup_mem_145217, wdown_mem_145218, wvoc_mem_145219);
        if (ret == 0) {
            struct memblock mem_145202 = ctx->constants->mem_145202;
            struct memblock mem_145203 = ctx->constants->mem_145203;
            struct memblock mem_145204 = ctx->constants->mem_145204;
            struct memblock mem_145205 = ctx->constants->mem_145205;
            struct memblock mem_145206 = ctx->constants->mem_145206;
            struct memblock mem_145207 = ctx->constants->mem_145207;
            struct memblock mem_145208 = ctx->constants->mem_145208;
            struct memblock mem_145209 = ctx->constants->mem_145209;
            struct memblock mem_145210 = ctx->constants->mem_145210;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v0->mem = mem_out_147157;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v1->mem = mem_out_147158;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v2->mem = mem_out_147159;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v3->mem = mem_out_147160;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v4->mem = mem_out_147161;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v5->mem = mem_out_147162;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v6->mem = mem_out_147163;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v7->mem = mem_out_147164;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v8->mem = mem_out_147165;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_train(struct futhark_context *ctx, struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 **out, const struct futhark_opaque_params *in0, const struct futhark_opaque_params *in1, const struct futhark_opaque_params *in2, const struct futhark_f32_4d *in3, const struct futhark_i64_1d *in4, const struct futhark_i64_3d *in5)
{
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock mem_out_147183;
    
    mem_out_147183.references = NULL;
    
    struct memblock mem_out_147182;
    
    mem_out_147182.references = NULL;
    
    struct memblock mem_out_147181;
    
    mem_out_147181.references = NULL;
    
    struct memblock mem_out_147180;
    
    mem_out_147180.references = NULL;
    
    struct memblock mem_out_147179;
    
    mem_out_147179.references = NULL;
    
    struct memblock mem_out_147178;
    
    mem_out_147178.references = NULL;
    
    struct memblock mem_out_147177;
    
    mem_out_147177.references = NULL;
    
    struct memblock mem_out_147176;
    
    mem_out_147176.references = NULL;
    
    struct memblock mem_out_147175;
    
    mem_out_147175.references = NULL;
    
    struct memblock mem_out_147174;
    
    mem_out_147174.references = NULL;
    
    struct memblock mem_out_147173;
    
    mem_out_147173.references = NULL;
    
    struct memblock mem_out_147172;
    
    mem_out_147172.references = NULL;
    
    struct memblock mem_out_147171;
    
    mem_out_147171.references = NULL;
    
    struct memblock mem_out_147170;
    
    mem_out_147170.references = NULL;
    
    struct memblock mem_out_147169;
    
    mem_out_147169.references = NULL;
    
    struct memblock mem_out_147168;
    
    mem_out_147168.references = NULL;
    
    struct memblock mem_out_147167;
    
    mem_out_147167.references = NULL;
    
    struct memblock mem_out_147166;
    
    mem_out_147166.references = NULL;
    
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    
    struct memblock seqs_mem_145240;
    
    seqs_mem_145240.references = NULL;
    
    struct memblock dls_mem_145239;
    
    dls_mem_145239.references = NULL;
    
    struct memblock masks_mem_145238;
    
    masks_mem_145238.references = NULL;
    
    struct memblock wvoc_mem_145237;
    
    wvoc_mem_145237.references = NULL;
    
    struct memblock wval_mem_145236;
    
    wval_mem_145236.references = NULL;
    
    struct memblock wup_mem_145235;
    
    wup_mem_145235.references = NULL;
    
    struct memblock wte_mem_145234;
    
    wte_mem_145234.references = NULL;
    
    struct memblock wqry_mem_145233;
    
    wqry_mem_145233.references = NULL;
    
    struct memblock wpe_mem_145232;
    
    wpe_mem_145232.references = NULL;
    
    struct memblock wout_mem_145231;
    
    wout_mem_145231.references = NULL;
    
    struct memblock wkey_mem_145230;
    
    wkey_mem_145230.references = NULL;
    
    struct memblock wdown_mem_145229;
    
    wdown_mem_145229.references = NULL;
    
    struct memblock wvoc_mem_145228;
    
    wvoc_mem_145228.references = NULL;
    
    struct memblock wval_mem_145227;
    
    wval_mem_145227.references = NULL;
    
    struct memblock wup_mem_145226;
    
    wup_mem_145226.references = NULL;
    
    struct memblock wte_mem_145225;
    
    wte_mem_145225.references = NULL;
    
    struct memblock wqry_mem_145224;
    
    wqry_mem_145224.references = NULL;
    
    struct memblock wpe_mem_145223;
    
    wpe_mem_145223.references = NULL;
    
    struct memblock wout_mem_145222;
    
    wout_mem_145222.references = NULL;
    
    struct memblock wkey_mem_145221;
    
    wkey_mem_145221.references = NULL;
    
    struct memblock wdown_mem_145220;
    
    wdown_mem_145220.references = NULL;
    
    struct memblock wvoc_mem_145219;
    
    wvoc_mem_145219.references = NULL;
    
    struct memblock wval_mem_145218;
    
    wval_mem_145218.references = NULL;
    
    struct memblock wup_mem_145217;
    
    wup_mem_145217.references = NULL;
    
    struct memblock wte_mem_145216;
    
    wte_mem_145216.references = NULL;
    
    struct memblock wqry_mem_145215;
    
    wqry_mem_145215.references = NULL;
    
    struct memblock wpe_mem_145214;
    
    wpe_mem_145214.references = NULL;
    
    struct memblock wout_mem_145213;
    
    wout_mem_145213.references = NULL;
    
    struct memblock wkey_mem_145212;
    
    wkey_mem_145212.references = NULL;
    
    struct memblock wdown_mem_145211;
    
    wdown_mem_145211.references = NULL;
    wdown_mem_145211 = in0->v0->mem;
    wkey_mem_145212 = in0->v1->mem;
    wout_mem_145213 = in0->v2->mem;
    wpe_mem_145214 = in0->v3->mem;
    wqry_mem_145215 = in0->v4->mem;
    wte_mem_145216 = in0->v5->mem;
    wup_mem_145217 = in0->v6->mem;
    wval_mem_145218 = in0->v7->mem;
    wvoc_mem_145219 = in0->v8->mem;
    wdown_mem_145220 = in1->v0->mem;
    wkey_mem_145221 = in1->v1->mem;
    wout_mem_145222 = in1->v2->mem;
    wpe_mem_145223 = in1->v3->mem;
    wqry_mem_145224 = in1->v4->mem;
    wte_mem_145225 = in1->v5->mem;
    wup_mem_145226 = in1->v6->mem;
    wval_mem_145227 = in1->v7->mem;
    wvoc_mem_145228 = in1->v8->mem;
    wdown_mem_145229 = in2->v0->mem;
    wkey_mem_145230 = in2->v1->mem;
    wout_mem_145231 = in2->v2->mem;
    wpe_mem_145232 = in2->v3->mem;
    wqry_mem_145233 = in2->v4->mem;
    wte_mem_145234 = in2->v5->mem;
    wup_mem_145235 = in2->v6->mem;
    wval_mem_145236 = in2->v7->mem;
    wvoc_mem_145237 = in2->v8->mem;
    masks_mem_145238 = in3->mem;
    dls_mem_145239 = in4->mem;
    seqs_mem_145240 = in5->mem;
    if (!(((int64_t) 16 == in0->v0->shape[0] && ((int64_t) 64 == in0->v0->shape[1] && ((int64_t) 16 == in0->v1->shape[0] && ((int64_t) 16 == in0->v1->shape[1] && ((int64_t) 16 == in0->v2->shape[0] && ((int64_t) 16 == in0->v2->shape[1] && ((int64_t) 16 == in0->v3->shape[0] && ((int64_t) 16 == in0->v3->shape[1] && ((int64_t) 16 == in0->v4->shape[0] && ((int64_t) 16 == in0->v4->shape[1] && ((int64_t) 27 == in0->v5->shape[0] && ((int64_t) 16 == in0->v5->shape[1] && ((int64_t) 64 == in0->v6->shape[0] && ((int64_t) 16 == in0->v6->shape[1] && ((int64_t) 16 == in0->v7->shape[0] && ((int64_t) 16 == in0->v7->shape[1] && ((int64_t) 27 == in0->v8->shape[0] && (int64_t) 16 == in0->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in1->v0->shape[0] && ((int64_t) 64 == in1->v0->shape[1] && ((int64_t) 16 == in1->v1->shape[0] && ((int64_t) 16 == in1->v1->shape[1] && ((int64_t) 16 == in1->v2->shape[0] && ((int64_t) 16 == in1->v2->shape[1] && ((int64_t) 16 == in1->v3->shape[0] && ((int64_t) 16 == in1->v3->shape[1] && ((int64_t) 16 == in1->v4->shape[0] && ((int64_t) 16 == in1->v4->shape[1] && ((int64_t) 27 == in1->v5->shape[0] && ((int64_t) 16 == in1->v5->shape[1] && ((int64_t) 64 == in1->v6->shape[0] && ((int64_t) 16 == in1->v6->shape[1] && ((int64_t) 16 == in1->v7->shape[0] && ((int64_t) 16 == in1->v7->shape[1] && ((int64_t) 27 == in1->v8->shape[0] && (int64_t) 16 == in1->v8->shape[1]))))))))))))))))) && (((int64_t) 16 == in2->v0->shape[0] && ((int64_t) 64 == in2->v0->shape[1] && ((int64_t) 16 == in2->v1->shape[0] && ((int64_t) 16 == in2->v1->shape[1] && ((int64_t) 16 == in2->v2->shape[0] && ((int64_t) 16 == in2->v2->shape[1] && ((int64_t) 16 == in2->v3->shape[0] && ((int64_t) 16 == in2->v3->shape[1] && ((int64_t) 16 == in2->v4->shape[0] && ((int64_t) 16 == in2->v4->shape[1] && ((int64_t) 27 == in2->v5->shape[0] && ((int64_t) 16 == in2->v5->shape[1] && ((int64_t) 64 == in2->v6->shape[0] && ((int64_t) 16 == in2->v6->shape[1] && ((int64_t) 16 == in2->v7->shape[0] && ((int64_t) 16 == in2->v7->shape[1] && ((int64_t) 27 == in2->v8->shape[0] && (int64_t) 16 == in2->v8->shape[1]))))))))))))))))) && (((int64_t) 2 == in3->shape[0] && ((int64_t) 1 == in3->shape[1] && ((int64_t) 16 == in3->shape[2] && (int64_t) 16 == in3->shape[3]))) && ((int64_t) 2 == in4->shape[0] && ((int64_t) 2 == in5->shape[0] && ((int64_t) 1 == in5->shape[1] && (int64_t) 16 == in5->shape[2])))))))) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_train(ctx, &mem_out_147157, &mem_out_147158, &mem_out_147159, &mem_out_147160, &mem_out_147161, &mem_out_147162, &mem_out_147163, &mem_out_147164, &mem_out_147165, &mem_out_147166, &mem_out_147167, &mem_out_147168, &mem_out_147169, &mem_out_147170, &mem_out_147171, &mem_out_147172, &mem_out_147173, &mem_out_147174, &mem_out_147175, &mem_out_147176, &mem_out_147177, &mem_out_147178, &mem_out_147179, &mem_out_147180, &mem_out_147181, &mem_out_147182, &mem_out_147183, wdown_mem_145211, wkey_mem_145212, wout_mem_145213, wpe_mem_145214, wqry_mem_145215, wte_mem_145216, wup_mem_145217, wval_mem_145218, wvoc_mem_145219, wdown_mem_145220, wkey_mem_145221, wout_mem_145222, wpe_mem_145223, wqry_mem_145224, wte_mem_145225, wup_mem_145226, wval_mem_145227, wvoc_mem_145228, wdown_mem_145229, wkey_mem_145230, wout_mem_145231, wpe_mem_145232, wqry_mem_145233, wte_mem_145234, wup_mem_145235, wval_mem_145236, wvoc_mem_145237, masks_mem_145238, dls_mem_145239, seqs_mem_145240);
        if (ret == 0) {
            struct memblock mem_145202 = ctx->constants->mem_145202;
            struct memblock mem_145203 = ctx->constants->mem_145203;
            struct memblock mem_145204 = ctx->constants->mem_145204;
            struct memblock mem_145205 = ctx->constants->mem_145205;
            struct memblock mem_145206 = ctx->constants->mem_145206;
            struct memblock mem_145207 = ctx->constants->mem_145207;
            struct memblock mem_145208 = ctx->constants->mem_145208;
            struct memblock mem_145209 = ctx->constants->mem_145209;
            struct memblock mem_145210 = ctx->constants->mem_145210;
            
            assert((*out = (struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32 *) malloc(sizeof(struct futhark_opaque_tup3_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_tup9_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32_arr2d_f32))) != NULL);
            assert(((*out)->v0 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v0->mem = mem_out_147157;
            (*out)->v0->shape[0] = (int64_t) 27;
            (*out)->v0->shape[1] = (int64_t) 16;
            assert(((*out)->v1 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v1->mem = mem_out_147158;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v2->mem = mem_out_147159;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v3->mem = mem_out_147160;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v4->mem = mem_out_147161;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v5->mem = mem_out_147162;
            (*out)->v5->shape[0] = (int64_t) 16;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v6->mem = mem_out_147163;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v7->mem = mem_out_147164;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 64;
            assert(((*out)->v8 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v8->mem = mem_out_147165;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
            assert(((*out)->v9 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v9->mem = mem_out_147166;
            (*out)->v9->shape[0] = (int64_t) 27;
            (*out)->v9->shape[1] = (int64_t) 16;
            assert(((*out)->v10 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v10->mem = mem_out_147167;
            (*out)->v10->shape[0] = (int64_t) 16;
            (*out)->v10->shape[1] = (int64_t) 16;
            assert(((*out)->v11 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v11->mem = mem_out_147168;
            (*out)->v11->shape[0] = (int64_t) 16;
            (*out)->v11->shape[1] = (int64_t) 16;
            assert(((*out)->v12 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v12->mem = mem_out_147169;
            (*out)->v12->shape[0] = (int64_t) 16;
            (*out)->v12->shape[1] = (int64_t) 16;
            assert(((*out)->v13 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v13->mem = mem_out_147170;
            (*out)->v13->shape[0] = (int64_t) 16;
            (*out)->v13->shape[1] = (int64_t) 16;
            assert(((*out)->v14 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v14->mem = mem_out_147171;
            (*out)->v14->shape[0] = (int64_t) 16;
            (*out)->v14->shape[1] = (int64_t) 16;
            assert(((*out)->v15 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v15->mem = mem_out_147172;
            (*out)->v15->shape[0] = (int64_t) 64;
            (*out)->v15->shape[1] = (int64_t) 16;
            assert(((*out)->v16 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v16->mem = mem_out_147173;
            (*out)->v16->shape[0] = (int64_t) 16;
            (*out)->v16->shape[1] = (int64_t) 64;
            assert(((*out)->v17 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v17->mem = mem_out_147174;
            (*out)->v17->shape[0] = (int64_t) 27;
            (*out)->v17->shape[1] = (int64_t) 16;
            assert(((*out)->v18 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v18->mem = mem_out_147175;
            (*out)->v18->shape[0] = (int64_t) 27;
            (*out)->v18->shape[1] = (int64_t) 16;
            assert(((*out)->v19 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v19->mem = mem_out_147176;
            (*out)->v19->shape[0] = (int64_t) 16;
            (*out)->v19->shape[1] = (int64_t) 16;
            assert(((*out)->v20 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v20->mem = mem_out_147177;
            (*out)->v20->shape[0] = (int64_t) 16;
            (*out)->v20->shape[1] = (int64_t) 16;
            assert(((*out)->v21 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v21->mem = mem_out_147178;
            (*out)->v21->shape[0] = (int64_t) 16;
            (*out)->v21->shape[1] = (int64_t) 16;
            assert(((*out)->v22 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v22->mem = mem_out_147179;
            (*out)->v22->shape[0] = (int64_t) 16;
            (*out)->v22->shape[1] = (int64_t) 16;
            assert(((*out)->v23 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v23->mem = mem_out_147180;
            (*out)->v23->shape[0] = (int64_t) 16;
            (*out)->v23->shape[1] = (int64_t) 16;
            assert(((*out)->v24 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v24->mem = mem_out_147181;
            (*out)->v24->shape[0] = (int64_t) 64;
            (*out)->v24->shape[1] = (int64_t) 16;
            assert(((*out)->v25 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v25->mem = mem_out_147182;
            (*out)->v25->shape[0] = (int64_t) 16;
            (*out)->v25->shape[1] = (int64_t) 64;
            assert(((*out)->v26 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v26->mem = mem_out_147183;
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
    
    struct memblock mem_out_147165;
    
    mem_out_147165.references = NULL;
    
    struct memblock mem_out_147164;
    
    mem_out_147164.references = NULL;
    
    struct memblock mem_out_147163;
    
    mem_out_147163.references = NULL;
    
    struct memblock mem_out_147162;
    
    mem_out_147162.references = NULL;
    
    struct memblock mem_out_147161;
    
    mem_out_147161.references = NULL;
    
    struct memblock mem_out_147160;
    
    mem_out_147160.references = NULL;
    
    struct memblock mem_out_147159;
    
    mem_out_147159.references = NULL;
    
    struct memblock mem_out_147158;
    
    mem_out_147158.references = NULL;
    
    struct memblock mem_out_147157;
    
    mem_out_147157.references = NULL;
    if (ret == 0) {
        ret = futrts_entry_zzero_params(ctx, &mem_out_147157, &mem_out_147158, &mem_out_147159, &mem_out_147160, &mem_out_147161, &mem_out_147162, &mem_out_147163, &mem_out_147164, &mem_out_147165);
        if (ret == 0) {
            struct memblock mem_145202 = ctx->constants->mem_145202;
            struct memblock mem_145203 = ctx->constants->mem_145203;
            struct memblock mem_145204 = ctx->constants->mem_145204;
            struct memblock mem_145205 = ctx->constants->mem_145205;
            struct memblock mem_145206 = ctx->constants->mem_145206;
            struct memblock mem_145207 = ctx->constants->mem_145207;
            struct memblock mem_145208 = ctx->constants->mem_145208;
            struct memblock mem_145209 = ctx->constants->mem_145209;
            struct memblock mem_145210 = ctx->constants->mem_145210;
            
            assert((*out = (struct futhark_opaque_params *) malloc(sizeof(struct futhark_opaque_params))) != NULL);
            assert(((*out)->v0 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v0->mem = mem_out_147157;
            (*out)->v0->shape[0] = (int64_t) 16;
            (*out)->v0->shape[1] = (int64_t) 64;
            assert(((*out)->v1 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v1->mem = mem_out_147158;
            (*out)->v1->shape[0] = (int64_t) 16;
            (*out)->v1->shape[1] = (int64_t) 16;
            assert(((*out)->v2 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v2->mem = mem_out_147159;
            (*out)->v2->shape[0] = (int64_t) 16;
            (*out)->v2->shape[1] = (int64_t) 16;
            assert(((*out)->v3 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v3->mem = mem_out_147160;
            (*out)->v3->shape[0] = (int64_t) 16;
            (*out)->v3->shape[1] = (int64_t) 16;
            assert(((*out)->v4 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v4->mem = mem_out_147161;
            (*out)->v4->shape[0] = (int64_t) 16;
            (*out)->v4->shape[1] = (int64_t) 16;
            assert(((*out)->v5 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v5->mem = mem_out_147162;
            (*out)->v5->shape[0] = (int64_t) 27;
            (*out)->v5->shape[1] = (int64_t) 16;
            assert(((*out)->v6 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v6->mem = mem_out_147163;
            (*out)->v6->shape[0] = (int64_t) 64;
            (*out)->v6->shape[1] = (int64_t) 16;
            assert(((*out)->v7 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v7->mem = mem_out_147164;
            (*out)->v7->shape[0] = (int64_t) 16;
            (*out)->v7->shape[1] = (int64_t) 16;
            assert(((*out)->v8 = (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) != NULL);
            (*out)->v8->mem = mem_out_147165;
            (*out)->v8->shape[0] = (int64_t) 27;
            (*out)->v8->shape[1] = (int64_t) 16;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
