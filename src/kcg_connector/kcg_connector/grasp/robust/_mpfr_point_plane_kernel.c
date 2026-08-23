#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Ubuntu 22.04 ships the MPFR runtime but not the development header in the
 * target image.  These are the stable public MPFR 4.x ABI declarations used
 * by libmpfr.so.6 on x86-64.  The Python loader checks the ABI version and a
 * directed-rounding self-test before exposing this optional accelerator. */
typedef long mpfr_prec_t;
typedef long mpfr_exp_t;
typedef int mpfr_sign_t;
typedef unsigned long mp_limb_t;
typedef struct {
    mpfr_prec_t _mpfr_prec;
    mpfr_sign_t _mpfr_sign;
    mpfr_exp_t _mpfr_exp;
    mp_limb_t *_mpfr_d;
} __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct *mpfr_ptr;
typedef const __mpfr_struct *mpfr_srcptr;
typedef enum {
    MPFR_RNDN = 0,
    MPFR_RNDZ = 1,
    MPFR_RNDU = 2,
    MPFR_RNDD = 3,
    MPFR_RNDA = 4,
    MPFR_RNDF = 5
} mpfr_rnd_t;

extern void mpfr_init2(mpfr_ptr, mpfr_prec_t);
extern void mpfr_clear(mpfr_ptr);
extern int mpfr_set(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_set_d(mpfr_ptr, double, mpfr_rnd_t);
extern double mpfr_get_d(mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_add(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_sub(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_mul(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_neg(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_sin(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_cos(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_cmp(mpfr_srcptr, mpfr_srcptr);

typedef struct {
    mpfr_t lower;
    mpfr_t upper;
} iv_t;

typedef struct {
    mpfr_prec_t precision;
    int joint_count;
    int independent_count;
    int *joint_types;
    double *axes;
    iv_t *joint_starts;
    iv_t *joint_directions;
    iv_t *origins;
    iv_t base[12];
    iv_t witness[3];
    iv_t plane_origin[3];
    iv_t plane_area[3];
    iv_t plane_offset;
    iv_t work_a[12];
    iv_t work_b[12];
    iv_t work_c[12];
    iv_t rotation[9];
    iv_t local[32];
    mpfr_t scalar[8];
} evaluator_t;

static void iv_init(iv_t *value, mpfr_prec_t precision) {
    mpfr_init2(value->lower, precision);
    mpfr_init2(value->upper, precision);
}

static void iv_clear(iv_t *value) {
    mpfr_clear(value->lower);
    mpfr_clear(value->upper);
}

static void iv_set_double(iv_t *value, double source) {
    mpfr_set_d(value->lower, source, MPFR_RNDN);
    mpfr_set_d(value->upper, source, MPFR_RNDN);
}

static void iv_set_bounds(iv_t *value, double lower, double upper) {
    mpfr_set_d(value->lower, lower, MPFR_RNDD);
    mpfr_set_d(value->upper, upper, MPFR_RNDU);
}

static void iv_copy(iv_t *target, const iv_t *source) {
    mpfr_set(target->lower, source->lower, MPFR_RNDD);
    mpfr_set(target->upper, source->upper, MPFR_RNDU);
}

static void iv_add(iv_t *target, const iv_t *first, const iv_t *second) {
    mpfr_add(target->lower, first->lower, second->lower, MPFR_RNDD);
    mpfr_add(target->upper, first->upper, second->upper, MPFR_RNDU);
}

static void iv_sub(iv_t *target, const iv_t *first, const iv_t *second) {
    mpfr_sub(target->lower, first->lower, second->upper, MPFR_RNDD);
    mpfr_sub(target->upper, first->upper, second->lower, MPFR_RNDU);
}

static void iv_neg(evaluator_t *evaluator, iv_t *target, const iv_t *source) {
    mpfr_neg(evaluator->scalar[0], source->upper, MPFR_RNDD);
    mpfr_neg(evaluator->scalar[1], source->lower, MPFR_RNDU);
    mpfr_set(target->lower, evaluator->scalar[0], MPFR_RNDD);
    mpfr_set(target->upper, evaluator->scalar[1], MPFR_RNDU);
}

static void iv_mul(
    evaluator_t *evaluator,
    iv_t *target,
    const iv_t *first,
    const iv_t *second
) {
    mpfr_mul(evaluator->scalar[0], first->lower, second->lower, MPFR_RNDD);
    mpfr_mul(evaluator->scalar[1], first->lower, second->upper, MPFR_RNDD);
    mpfr_mul(evaluator->scalar[2], first->upper, second->lower, MPFR_RNDD);
    mpfr_mul(evaluator->scalar[3], first->upper, second->upper, MPFR_RNDD);
    mpfr_srcptr minimum = evaluator->scalar[0];
    for (int index = 1; index < 4; ++index) {
        if (mpfr_cmp(evaluator->scalar[index], minimum) < 0) {
            minimum = evaluator->scalar[index];
        }
    }
    mpfr_mul(evaluator->scalar[4], first->lower, second->lower, MPFR_RNDU);
    mpfr_mul(evaluator->scalar[5], first->lower, second->upper, MPFR_RNDU);
    mpfr_mul(evaluator->scalar[6], first->upper, second->lower, MPFR_RNDU);
    mpfr_mul(evaluator->scalar[7], first->upper, second->upper, MPFR_RNDU);
    mpfr_srcptr maximum = evaluator->scalar[4];
    for (int index = 5; index < 8; ++index) {
        if (mpfr_cmp(evaluator->scalar[index], maximum) > 0) {
            maximum = evaluator->scalar[index];
        }
    }
    mpfr_set(target->lower, minimum, MPFR_RNDD);
    mpfr_set(target->upper, maximum, MPFR_RNDU);
}

static int periodic_contains(
    double lower,
    double upper,
    long double base,
    long double period
) {
    if (!isfinite(lower) || !isfinite(upper) || lower > upper) {
        return 1;
    }
    long double scale = fmaxl(1.0L, fmaxl(fabsl(lower), fabsl(upper)));
    long double margin = 4096.0L * LDBL_EPSILON * scale + 1.0e-18L;
    long double lo = (long double)lower - margin;
    long double hi = (long double)upper + margin;
    if (hi - lo >= period) {
        return 1;
    }
    long double index = ceill((lo - base) / period);
    return base + index * period <= hi;
}

static void iv_sin(evaluator_t *evaluator, iv_t *target, const iv_t *source) {
    mpfr_sin(evaluator->scalar[0], source->lower, MPFR_RNDD);
    mpfr_sin(evaluator->scalar[1], source->upper, MPFR_RNDD);
    mpfr_sin(evaluator->scalar[2], source->lower, MPFR_RNDU);
    mpfr_sin(evaluator->scalar[3], source->upper, MPFR_RNDU);
    mpfr_srcptr minimum = mpfr_cmp(evaluator->scalar[0], evaluator->scalar[1]) <= 0
        ? evaluator->scalar[0] : evaluator->scalar[1];
    mpfr_srcptr maximum = mpfr_cmp(evaluator->scalar[2], evaluator->scalar[3]) >= 0
        ? evaluator->scalar[2] : evaluator->scalar[3];
    double lower = mpfr_get_d(source->lower, MPFR_RNDD);
    double upper = mpfr_get_d(source->upper, MPFR_RNDU);
    if (periodic_contains(lower, upper, -0.5L * acosl(-1.0L), 2.0L * acosl(-1.0L))) {
        mpfr_set_d(target->lower, -1.0, MPFR_RNDN);
    } else {
        mpfr_set(target->lower, minimum, MPFR_RNDD);
    }
    if (periodic_contains(lower, upper, 0.5L * acosl(-1.0L), 2.0L * acosl(-1.0L))) {
        mpfr_set_d(target->upper, 1.0, MPFR_RNDN);
    } else {
        mpfr_set(target->upper, maximum, MPFR_RNDU);
    }
}

static void iv_cos(evaluator_t *evaluator, iv_t *target, const iv_t *source) {
    mpfr_cos(evaluator->scalar[0], source->lower, MPFR_RNDD);
    mpfr_cos(evaluator->scalar[1], source->upper, MPFR_RNDD);
    mpfr_cos(evaluator->scalar[2], source->lower, MPFR_RNDU);
    mpfr_cos(evaluator->scalar[3], source->upper, MPFR_RNDU);
    mpfr_srcptr minimum = mpfr_cmp(evaluator->scalar[0], evaluator->scalar[1]) <= 0
        ? evaluator->scalar[0] : evaluator->scalar[1];
    mpfr_srcptr maximum = mpfr_cmp(evaluator->scalar[2], evaluator->scalar[3]) >= 0
        ? evaluator->scalar[2] : evaluator->scalar[3];
    double lower = mpfr_get_d(source->lower, MPFR_RNDD);
    double upper = mpfr_get_d(source->upper, MPFR_RNDU);
    if (periodic_contains(lower, upper, acosl(-1.0L), 2.0L * acosl(-1.0L))) {
        mpfr_set_d(target->lower, -1.0, MPFR_RNDN);
    } else {
        mpfr_set(target->lower, minimum, MPFR_RNDD);
    }
    if (periodic_contains(lower, upper, 0.0L, 2.0L * acosl(-1.0L))) {
        mpfr_set_d(target->upper, 1.0, MPFR_RNDN);
    } else {
        mpfr_set(target->upper, maximum, MPFR_RNDU);
    }
}

static void matrix_identity(iv_t *matrix) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            iv_set_double(&matrix[row * 4 + column], row == column ? 1.0 : 0.0);
        }
    }
}

static void rigid_compose(
    evaluator_t *evaluator,
    iv_t *output,
    const iv_t *first,
    const iv_t *second
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            iv_set_double(&output[row * 4 + column], 0.0);
            for (int inner = 0; inner < 3; ++inner) {
                iv_mul(evaluator, &evaluator->local[0],
                       &first[row * 4 + inner], &second[inner * 4 + column]);
                iv_add(&output[row * 4 + column],
                       &output[row * 4 + column], &evaluator->local[0]);
            }
        }
        iv_set_double(&output[row * 4 + 3], 0.0);
        for (int inner = 0; inner < 3; ++inner) {
            iv_mul(evaluator, &evaluator->local[0],
                   &first[row * 4 + inner], &second[inner * 4 + 3]);
            iv_add(&output[row * 4 + 3],
                   &output[row * 4 + 3], &evaluator->local[0]);
        }
        iv_add(&output[row * 4 + 3], &output[row * 4 + 3], &first[row * 4 + 3]);
    }
}

static void rpy_origin(
    evaluator_t *evaluator,
    iv_t *output,
    const double *xyz,
    const double *rpy
) {
    iv_t *roll = &evaluator->local[0];
    iv_t *pitch = &evaluator->local[1];
    iv_t *yaw = &evaluator->local[2];
    iv_t *cr = &evaluator->local[3];
    iv_t *sr = &evaluator->local[4];
    iv_t *cp = &evaluator->local[5];
    iv_t *sp = &evaluator->local[6];
    iv_t *cy = &evaluator->local[7];
    iv_t *sy = &evaluator->local[8];
    iv_set_double(roll, rpy[0]); iv_set_double(pitch, rpy[1]); iv_set_double(yaw, rpy[2]);
    iv_cos(evaluator, cr, roll); iv_sin(evaluator, sr, roll);
    iv_cos(evaluator, cp, pitch); iv_sin(evaluator, sp, pitch);
    iv_cos(evaluator, cy, yaw); iv_sin(evaluator, sy, yaw);
    matrix_identity(output);
    iv_mul(evaluator, &output[0], cy, cp);
    iv_mul(evaluator, &evaluator->local[9], cy, sp);
    iv_mul(evaluator, &evaluator->local[10], &evaluator->local[9], sr);
    iv_mul(evaluator, &evaluator->local[11], sy, cr);
    iv_sub(&output[1], &evaluator->local[10], &evaluator->local[11]);
    iv_mul(evaluator, &evaluator->local[10], &evaluator->local[9], cr);
    iv_mul(evaluator, &evaluator->local[11], sy, sr);
    iv_add(&output[2], &evaluator->local[10], &evaluator->local[11]);
    iv_mul(evaluator, &output[4], sy, cp);
    iv_mul(evaluator, &evaluator->local[9], sy, sp);
    iv_mul(evaluator, &evaluator->local[10], &evaluator->local[9], sr);
    iv_mul(evaluator, &evaluator->local[11], cy, cr);
    iv_add(&output[5], &evaluator->local[10], &evaluator->local[11]);
    iv_mul(evaluator, &evaluator->local[10], &evaluator->local[9], cr);
    iv_mul(evaluator, &evaluator->local[11], cy, sr);
    iv_sub(&output[6], &evaluator->local[10], &evaluator->local[11]);
    iv_neg(evaluator, &output[8], sp);
    iv_mul(evaluator, &output[9], cp, sr);
    iv_mul(evaluator, &output[10], cp, cr);
    for (int row = 0; row < 3; ++row) {
        iv_set_double(&output[row * 4 + 3], xyz[row]);
    }
}

static void axis_rotation(
    evaluator_t *evaluator,
    iv_t *rotation,
    const double *axis,
    const iv_t *angle
) {
    iv_t *sine = &evaluator->local[0];
    iv_t *cosine = &evaluator->local[1];
    iv_sin(evaluator, sine, angle);
    iv_cos(evaluator, cosine, angle);

    /* The real hand uses cardinal URDF axes.  Keeping this branch in the
     * rigorous MPFR kernel removes 18 general Rodrigues products per joint
     * evaluation without changing the represented rotation. */
    int cardinal = -1;
    double sign = 0.0;
    for (int index = 0; index < 3; ++index) {
        int first = (index + 1) % 3;
        int second = (index + 2) % 3;
        if ((axis[index] == 1.0 || axis[index] == -1.0)
            && axis[first] == 0.0 && axis[second] == 0.0) {
            cardinal = index;
            sign = axis[index];
            break;
        }
    }
    if (cardinal >= 0) {
        iv_t *signed_sine = &evaluator->local[2];
        iv_t *negative_sine = &evaluator->local[3];
        if (sign > 0.0) {
            iv_copy(signed_sine, sine);
        } else {
            iv_neg(evaluator, signed_sine, sine);
        }
        iv_neg(evaluator, negative_sine, signed_sine);
        for (int index = 0; index < 9; ++index) {
            iv_set_double(&rotation[index], 0.0);
        }
        if (cardinal == 0) {
            iv_set_double(&rotation[0], 1.0);
            iv_copy(&rotation[4], cosine);
            iv_copy(&rotation[5], negative_sine);
            iv_copy(&rotation[7], signed_sine);
            iv_copy(&rotation[8], cosine);
        } else if (cardinal == 1) {
            iv_copy(&rotation[0], cosine);
            iv_copy(&rotation[2], signed_sine);
            iv_set_double(&rotation[4], 1.0);
            iv_copy(&rotation[6], negative_sine);
            iv_copy(&rotation[8], cosine);
        } else {
            iv_copy(&rotation[0], cosine);
            iv_copy(&rotation[1], negative_sine);
            iv_copy(&rotation[3], signed_sine);
            iv_copy(&rotation[4], cosine);
            iv_set_double(&rotation[8], 1.0);
        }
        return;
    }

    iv_t *one_minus_cosine = &evaluator->local[2];
    iv_set_double(one_minus_cosine, 1.0);
    iv_sub(one_minus_cosine, one_minus_cosine, cosine);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            double skew = 0.0;
            if (row == 0 && column == 1) skew = -axis[2];
            if (row == 0 && column == 2) skew = axis[1];
            if (row == 1 && column == 0) skew = axis[2];
            if (row == 1 && column == 2) skew = -axis[0];
            if (row == 2 && column == 0) skew = -axis[1];
            if (row == 2 && column == 1) skew = axis[0];
            iv_set_double(&rotation[row * 3 + column], row == column ? 1.0 : 0.0);
            iv_set_double(&evaluator->local[3], skew);
            iv_mul(evaluator, &evaluator->local[4], sine, &evaluator->local[3]);
            iv_add(&rotation[row * 3 + column], &rotation[row * 3 + column], &evaluator->local[4]);
            double skew_squared = axis[row] * axis[column] - (row == column ? 1.0 : 0.0);
            iv_set_double(&evaluator->local[3], skew_squared);
            iv_mul(evaluator, &evaluator->local[4], one_minus_cosine, &evaluator->local[3]);
            iv_add(&rotation[row * 3 + column], &rotation[row * 3 + column], &evaluator->local[4]);
        }
    }
}

static void apply_joint(
    evaluator_t *evaluator,
    iv_t *output,
    const iv_t *input,
    int joint_index,
    const iv_t *position
) {
    const iv_t *origin = &evaluator->origins[joint_index * 12];
    rigid_compose(evaluator, evaluator->work_c, input, origin);
    int joint_type = evaluator->joint_types[joint_index];
    if (joint_type == 0) {
        for (int index = 0; index < 12; ++index) iv_copy(&output[index], &evaluator->work_c[index]);
        return;
    }
    if (joint_type == 1) {
        axis_rotation(evaluator, evaluator->rotation,
                      &evaluator->axes[joint_index * 3], position);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                iv_set_double(&output[row * 4 + column], 0.0);
                for (int inner = 0; inner < 3; ++inner) {
                    iv_mul(evaluator, &evaluator->local[10],
                           &evaluator->work_c[row * 4 + inner],
                           &evaluator->rotation[inner * 3 + column]);
                    iv_add(&output[row * 4 + column],
                           &output[row * 4 + column], &evaluator->local[10]);
                }
            }
            iv_copy(&output[row * 4 + 3], &evaluator->work_c[row * 4 + 3]);
        }
        return;
    }
    if (joint_type == 2) {
        for (int index = 0; index < 12; ++index) iv_copy(&output[index], &evaluator->work_c[index]);
        for (int row = 0; row < 3; ++row) {
            for (int inner = 0; inner < 3; ++inner) {
                double component = evaluator->axes[joint_index * 3 + inner];
                if (component == 0.0) continue;
                iv_set_double(&evaluator->local[10], component);
                iv_mul(evaluator, &evaluator->local[11], &evaluator->local[10], position);
                iv_mul(evaluator, &evaluator->local[12],
                       &evaluator->work_c[row * 4 + inner], &evaluator->local[11]);
                iv_add(&output[row * 4 + 3], &output[row * 4 + 3], &evaluator->local[12]);
            }
        }
    }
}

static void cross_product(
    evaluator_t *evaluator,
    iv_t *output,
    const iv_t *first,
    const iv_t *second
) {
    iv_mul(evaluator, &evaluator->local[20], &first[1], &second[2]);
    iv_mul(evaluator, &evaluator->local[21], &first[2], &second[1]);
    iv_sub(&output[0], &evaluator->local[20], &evaluator->local[21]);
    iv_mul(evaluator, &evaluator->local[20], &first[2], &second[0]);
    iv_mul(evaluator, &evaluator->local[21], &first[0], &second[2]);
    iv_sub(&output[1], &evaluator->local[20], &evaluator->local[21]);
    iv_mul(evaluator, &evaluator->local[20], &first[0], &second[1]);
    iv_mul(evaluator, &evaluator->local[21], &first[1], &second[0]);
    iv_sub(&output[2], &evaluator->local[20], &evaluator->local[21]);
}

static int set_triangle(evaluator_t *e, const double *triangle) {
    if (!e || !triangle) return 1;
    for (int index = 0; index < 9; ++index) {
        if (!isfinite(triangle[index])) return 1;
    }
    for (int index = 0; index < 3; ++index) {
        iv_set_double(&e->plane_origin[index], triangle[index]);
    }
    iv_t *edge_one = &e->local[22];
    iv_t *edge_two = &e->local[25];
    for (int index = 0; index < 3; ++index) {
        iv_set_double(&e->local[28], triangle[3 + index]);
        iv_set_double(&e->local[29], triangle[6 + index]);
        iv_sub(&edge_one[index], &e->local[28], &e->plane_origin[index]);
        iv_sub(&edge_two[index], &e->local[29], &e->plane_origin[index]);
    }
    cross_product(e, e->plane_area, edge_one, edge_two);
    iv_set_double(&e->plane_offset, 0.0);
    for (int index = 0; index < 3; ++index) {
        iv_mul(e, &e->local[30], &e->plane_area[index], &e->plane_origin[index]);
        iv_sub(&e->plane_offset, &e->plane_offset, &e->local[30]);
    }
    return 0;
}

int carts_mpfr_point_plane_abi_version(void) { return 4; }

void *carts_mpfr_point_plane_create(
    long precision,
    int joint_count,
    int independent_count,
    const int *joint_types,
    const int *source_indices,
    const double *origins_xyz,
    const double *origins_rpy,
    const double *axes,
    const double *multipliers,
    const double *offsets,
    const double *q_start,
    const double *direction,
    const double *base,
    const double *witness,
    const double *triangle
) {
    if (precision < 64 || joint_count <= 0 || independent_count <= 0
        || !joint_types || !source_indices || !origins_xyz || !origins_rpy
        || !axes || !multipliers || !offsets || !q_start || !direction
        || !base || !witness || !triangle) return NULL;
    for (int index = 0; index < joint_count; ++index) {
        if (joint_types[index] < 0 || joint_types[index] > 2) return NULL;
        if (joint_types[index] != 0
            && (source_indices[index] < 0
                || source_indices[index] >= independent_count)) return NULL;
        if (!isfinite(multipliers[index]) || !isfinite(offsets[index])) return NULL;
        for (int axis_index = 0; axis_index < 3; ++axis_index) {
            if (!isfinite(axes[index * 3 + axis_index])
                || !isfinite(origins_xyz[index * 3 + axis_index])
                || !isfinite(origins_rpy[index * 3 + axis_index])) return NULL;
        }
    }
    for (int index = 0; index < independent_count; ++index) {
        if (!isfinite(q_start[index]) || !isfinite(direction[index])) return NULL;
    }
    for (int index = 0; index < 12; ++index) if (!isfinite(base[index])) return NULL;
    for (int index = 0; index < 3; ++index) if (!isfinite(witness[index])) return NULL;
    for (int index = 0; index < 9; ++index) if (!isfinite(triangle[index])) return NULL;

    evaluator_t *e = (evaluator_t *)calloc(1, sizeof(evaluator_t));
    if (e == NULL) return NULL;
    e->precision = precision; e->joint_count = joint_count; e->independent_count = independent_count;
    e->joint_types = (int *)calloc((size_t)joint_count, sizeof(int));
    e->axes = (double *)calloc((size_t)joint_count * 3U, sizeof(double));
    e->joint_starts = (iv_t *)calloc((size_t)joint_count, sizeof(iv_t));
    e->joint_directions = (iv_t *)calloc((size_t)joint_count, sizeof(iv_t));
    e->origins = (iv_t *)calloc((size_t)joint_count * 12U, sizeof(iv_t));
    if (!e->joint_types || !e->axes || !e->joint_starts
        || !e->joint_directions || !e->origins) {
        free(e->joint_types); free(e->axes); free(e->joint_starts);
        free(e->joint_directions); free(e->origins); free(e);
        return NULL;
    }
    for (int index = 0; index < 8; ++index) mpfr_init2(e->scalar[index], precision);
    for (int index = 0; index < 32; ++index) iv_init(&e->local[index], precision);
    for (int index = 0; index < 12; ++index) {
        iv_init(&e->base[index], precision); iv_init(&e->work_a[index], precision);
        iv_init(&e->work_b[index], precision); iv_init(&e->work_c[index], precision);
    }
    for (int index = 0; index < 9; ++index) iv_init(&e->rotation[index], precision);
    for (int index = 0; index < 3; ++index) {
        iv_init(&e->witness[index], precision); iv_init(&e->plane_origin[index], precision);
        iv_init(&e->plane_area[index], precision);
    }
    iv_init(&e->plane_offset, precision);
    for (int index = 0; index < joint_count; ++index) {
        e->joint_types[index] = joint_types[index];
        for (int axis_index = 0; axis_index < 3; ++axis_index)
            e->axes[index * 3 + axis_index] = axes[index * 3 + axis_index];
        iv_init(&e->joint_starts[index], precision);
        iv_init(&e->joint_directions[index], precision);
        if (joint_types[index] == 0) {
            iv_set_double(&e->joint_starts[index], 0.0);
            iv_set_double(&e->joint_directions[index], 0.0);
        } else {
            int source = source_indices[index];
            iv_set_double(&e->local[14], q_start[source]);
            iv_set_double(&e->local[15], direction[source]);
            iv_set_double(&e->local[16], multipliers[index]);
            iv_set_double(&e->local[17], offsets[index]);
            iv_mul(e, &e->joint_starts[index], &e->local[16], &e->local[14]);
            iv_add(&e->joint_starts[index], &e->joint_starts[index], &e->local[17]);
            iv_mul(e, &e->joint_directions[index], &e->local[16], &e->local[15]);
        }
        for (int cell = 0; cell < 12; ++cell) iv_init(&e->origins[index * 12 + cell], precision);
        rpy_origin(e, &e->origins[index * 12], &origins_xyz[index * 3], &origins_rpy[index * 3]);
    }
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 4; ++column)
        iv_set_double(&e->base[row * 4 + column], base[row * 4 + column]);
    for (int index = 0; index < 3; ++index) {
        iv_set_double(&e->witness[index], witness[index]);
    }
    if (set_triangle(e, triangle) != 0) return NULL;
    return e;
}

int carts_mpfr_point_plane_set_triangle(
    void *handle,
    const double *triangle
) {
    return set_triangle((evaluator_t *)handle, triangle);
}

static int evaluate_interval(
    evaluator_t *e,
    double phase_lower,
    double phase_upper,
    double *plane_lower,
    double *plane_upper,
    double *position_lower,
    double *position_upper
) {
    if (!e || !plane_lower || !plane_upper
        || !isfinite(phase_lower) || !isfinite(phase_upper)
        || phase_lower > phase_upper) return 1;
    /* Slots 14-18 survive helpers that use slots 0-12 as scratch. */
    iv_t *phase = &e->local[14];
    iv_set_bounds(phase, phase_lower, phase_upper);
    iv_t *current = e->work_a; iv_t *next = e->work_b;
    for (int index = 0; index < 12; ++index) iv_copy(&current[index], &e->base[index]);
    for (int joint = 0; joint < e->joint_count; ++joint) {
        iv_t *position = &e->local[15];
        if (e->joint_types[joint] == 0) {
            iv_set_double(position, 0.0);
        } else {
            iv_mul(e, &e->local[16], &e->joint_directions[joint], phase);
            iv_add(position, &e->joint_starts[joint], &e->local[16]);
        }
        apply_joint(e, next, current, joint, position);
        iv_t *swap = current; current = next; next = swap;
    }
    iv_t *point = &e->local[5];
    for (int row = 0; row < 3; ++row) {
        iv_copy(&point[row], &current[row * 4 + 3]);
        for (int column = 0; column < 3; ++column) {
            iv_mul(e, &e->local[8], &current[row * 4 + column], &e->witness[column]);
            iv_add(&point[row], &point[row], &e->local[8]);
        }
        if (position_lower && position_upper) {
            position_lower[row] = mpfr_get_d(point[row].lower, MPFR_RNDD);
            position_upper[row] = mpfr_get_d(point[row].upper, MPFR_RNDU);
            if (!isfinite(position_lower[row]) || !isfinite(position_upper[row])
                || position_lower[row] > position_upper[row]) return 3;
        }
    }
    iv_copy(&e->local[12], &e->plane_offset);
    for (int index = 0; index < 3; ++index) {
        iv_mul(e, &e->local[13], &e->plane_area[index], &point[index]);
        iv_add(&e->local[12], &e->local[12], &e->local[13]);
    }
    *plane_lower = mpfr_get_d(e->local[12].lower, MPFR_RNDD);
    *plane_upper = mpfr_get_d(e->local[12].upper, MPFR_RNDU);
    return (!isfinite(*plane_lower) || !isfinite(*plane_upper)
            || *plane_lower > *plane_upper) ? 3 : 0;
}

int carts_mpfr_point_plane_evaluate(
    void *handle,
    double phase_value,
    double *lower,
    double *upper
) {
    return evaluate_interval(
        (evaluator_t *)handle,
        phase_value,
        phase_value,
        lower,
        upper,
        NULL,
        NULL
    );
}

int carts_mpfr_point_plane_evaluate_interval(
    void *handle,
    double phase_lower,
    double phase_upper,
    double *plane_lower,
    double *plane_upper,
    double *position_lower,
    double *position_upper
) {
    if (!position_lower || !position_upper) return 1;
    return evaluate_interval(
        (evaluator_t *)handle,
        phase_lower,
        phase_upper,
        plane_lower,
        plane_upper,
        position_lower,
        position_upper
    );
}

static int strict_sign(double lower, double upper) {
    if (lower > 0.0) return 1;
    if (upper < 0.0) return -1;
    return 0;
}

int carts_mpfr_point_plane_isolate_monotone_root(
    void *handle,
    double phase_lower,
    double phase_upper,
    double derivative_lower,
    double derivative_upper,
    int lower_sign,
    int upper_sign,
    int maximum_iterations,
    double *root_lower,
    double *root_upper,
    double *lower_value_lower,
    double *lower_value_upper,
    double *upper_value_lower,
    double *upper_value_upper,
    int *interpolation_iterations,
    int *newton_iterations,
    int *bisection_iterations
) {
    evaluator_t *e = (evaluator_t *)handle;
    if (!e || !root_lower || !root_upper || !lower_value_lower
        || !lower_value_upper || !upper_value_lower || !upper_value_upper
        || !interpolation_iterations || !newton_iterations
        || !bisection_iterations
        || !isfinite(phase_lower) || !isfinite(phase_upper)
        || !isfinite(derivative_lower) || !isfinite(derivative_upper)
        || phase_lower >= phase_upper || maximum_iterations <= 0
        || (lower_sign != -1 && lower_sign != 1)
        || upper_sign != -lower_sign
        || derivative_lower > derivative_upper
        || strict_sign(derivative_lower, derivative_upper) != upper_sign) return 1;

    double lo = phase_lower;
    double hi = phase_upper;
    double lo_value_lower = 0.0;
    double lo_value_upper = 0.0;
    double hi_value_lower = 0.0;
    double hi_value_upper = 0.0;
    int status = evaluate_interval(
        e, lo, lo, &lo_value_lower, &lo_value_upper, NULL, NULL
    );
    if (status != 0) return 2;
    status = evaluate_interval(
        e, hi, hi, &hi_value_lower, &hi_value_upper, NULL, NULL
    );
    if (status != 0) return 2;
    if (strict_sign(lo_value_lower, lo_value_upper) != lower_sign
        || strict_sign(hi_value_lower, hi_value_upper) != upper_sign) return 3;

    int interpolation_count = 0;
    int newton_count = 0;
    int bisection_count = 0;
    int exact_midpoint_bracket = 0;

    /* First use a safeguarded Illinois/regula-falsi proposal.  The binary64
     * interpolation is only a location hint: every accepted replacement
     * endpoint is re-evaluated by the 269-bit directed MPFR path above.  The
     * strict derivative sign already proves uniqueness, so retaining strict
     * opposite endpoint signs retains the complete root certificate. */
    double lo_actual = lo_value_lower
        + 0.5 * (lo_value_upper - lo_value_lower);
    double hi_actual = hi_value_lower
        + 0.5 * (hi_value_upper - hi_value_lower);
    double lo_weighted = lo_actual;
    double hi_weighted = hi_actual;
    int last_replaced_sign = 0;
    for (int attempt = 0; attempt < 2 && nextafter(lo, hi) < hi; ++attempt) {
        double width = hi - lo;
        double denominator = hi_weighted - lo_weighted;
        if (!isfinite(width) || width <= 0.0 || !isfinite(denominator)
            || denominator == 0.0) break;
        double proposed = lo - lo_weighted * width / denominator;
        double guard = width / 1024.0;
        if (!isfinite(proposed) || !(proposed > lo + guard)
            || !(proposed < hi - guard)) break;

        double proposed_lower = 0.0;
        double proposed_upper = 0.0;
        status = evaluate_interval(
            e, proposed, proposed,
            &proposed_lower, &proposed_upper, NULL, NULL
        );
        if (status != 0) return 2;
        int proposed_sign = strict_sign(proposed_lower, proposed_upper);
        ++interpolation_count;
        if (proposed_sign == 0) {
            double predecessor = nextafter(proposed, lo);
            double successor = nextafter(proposed, hi);
            double predecessor_lower = 0.0;
            double predecessor_upper = 0.0;
            double successor_lower = 0.0;
            double successor_upper = 0.0;
            if (!(predecessor < proposed && proposed < successor)) break;
            status = evaluate_interval(
                e, predecessor, predecessor,
                &predecessor_lower, &predecessor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            status = evaluate_interval(
                e, successor, successor,
                &successor_lower, &successor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            if (strict_sign(predecessor_lower, predecessor_upper) != lower_sign
                || strict_sign(successor_lower, successor_upper) != upper_sign) {
                break;
            }
            lo = predecessor;
            hi = successor;
            lo_value_lower = predecessor_lower;
            lo_value_upper = predecessor_upper;
            hi_value_lower = successor_lower;
            hi_value_upper = successor_upper;
            exact_midpoint_bracket = 1;
            break;
        }

        double proposed_actual = proposed_lower
            + 0.5 * (proposed_upper - proposed_lower);
        if (!isfinite(proposed_actual)) break;
        if (proposed_sign == lower_sign) {
            lo = proposed;
            lo_value_lower = proposed_lower;
            lo_value_upper = proposed_upper;
            lo_actual = proposed_actual;
            lo_weighted = lo_actual;
            hi_weighted = (
                last_replaced_sign == lower_sign
                ? 0.5 * hi_weighted
                : hi_actual
            );
            last_replaced_sign = lower_sign;
        } else if (proposed_sign == upper_sign) {
            hi = proposed;
            hi_value_lower = proposed_lower;
            hi_value_upper = proposed_upper;
            hi_actual = proposed_actual;
            hi_weighted = hi_actual;
            lo_weighted = (
                last_replaced_sign == upper_sign
                ? 0.5 * lo_weighted
                : lo_actual
            );
            last_replaced_sign = upper_sign;
        } else {
            break;
        }
    }

    /* A whole-interval derivative with a strict sign has already proved that
     * there is only one root.  Contract its bracket with outward-rounded
     * interval Newton, then retain the original endpoint-sign checks as the
     * acceptance gate.  Any unusable proposal simply falls through to the
     * proven bisection path below. */
    for (int attempt = 0; attempt < 4 && nextafter(lo, hi) < hi; ++attempt) {
        double midpoint = lo + 0.5 * (hi - lo);
        if (!(midpoint > lo && midpoint < hi) || !isfinite(midpoint)) break;
        double middle_lower = 0.0;
        double middle_upper = 0.0;
        status = evaluate_interval(
            e, midpoint, midpoint, &middle_lower, &middle_upper, NULL, NULL
        );
        if (status != 0) return 2;
        int middle_sign = strict_sign(middle_lower, middle_upper);
        if (middle_sign == 0) {
            double predecessor = nextafter(midpoint, lo);
            double successor = nextafter(midpoint, hi);
            double predecessor_lower = 0.0;
            double predecessor_upper = 0.0;
            double successor_lower = 0.0;
            double successor_upper = 0.0;
            if (!(predecessor < midpoint && midpoint < successor)) break;
            status = evaluate_interval(
                e, predecessor, predecessor,
                &predecessor_lower, &predecessor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            status = evaluate_interval(
                e, successor, successor,
                &successor_lower, &successor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            if (strict_sign(predecessor_lower, predecessor_upper) != lower_sign
                || strict_sign(successor_lower, successor_upper) != upper_sign) {
                break;
            }
            lo = predecessor;
            hi = successor;
            lo_value_lower = predecessor_lower;
            lo_value_upper = predecessor_upper;
            hi_value_lower = successor_lower;
            hi_value_upper = successor_upper;
            ++newton_count;
            exact_midpoint_bracket = 1;
            break;
        }

        double quotients[4] = {
            middle_lower / derivative_lower,
            middle_lower / derivative_upper,
            middle_upper / derivative_lower,
            middle_upper / derivative_upper
        };
        double quotient_lower = quotients[0];
        double quotient_upper = quotients[0];
        int quotient_finite = isfinite(quotients[0]);
        for (int index = 1; index < 4; ++index) {
            quotient_finite = quotient_finite && isfinite(quotients[index]);
            quotient_lower = fmin(quotient_lower, quotients[index]);
            quotient_upper = fmax(quotient_upper, quotients[index]);
        }
        if (!quotient_finite) break;
        quotient_lower = nextafter(quotient_lower, -INFINITY);
        quotient_upper = nextafter(quotient_upper, INFINITY);
        double candidate_lower = nextafter(midpoint - quotient_upper, -INFINITY);
        double candidate_upper = nextafter(midpoint - quotient_lower, INFINITY);
        if (!isfinite(candidate_lower) || !isfinite(candidate_upper)
            || candidate_lower > candidate_upper) break;

        double midpoint_ulp = fmax(
            fabs(nextafter(midpoint, INFINITY) - midpoint),
            fabs(midpoint - nextafter(midpoint, -INFINITY))
        );
        double candidate_lower_ulp = fmax(
            fabs(nextafter(candidate_lower, INFINITY) - candidate_lower),
            fabs(candidate_lower - nextafter(candidate_lower, -INFINITY))
        );
        double candidate_upper_ulp = fmax(
            fabs(nextafter(candidate_upper, INFINITY) - candidate_upper),
            fabs(candidate_upper - nextafter(candidate_upper, -INFINITY))
        );
        double padding = 32.0 * fmax(
            midpoint_ulp, fmax(candidate_lower_ulp, candidate_upper_ulp)
        );
        double proposed_lower = fmax(
            lo, nextafter(candidate_lower - padding, -INFINITY)
        );
        double proposed_upper = fmin(
            hi, nextafter(candidate_upper + padding, INFINITY)
        );
        double old_width = hi - lo;
        double new_width = proposed_upper - proposed_lower;
        if (!isfinite(proposed_lower) || !isfinite(proposed_upper)
            || proposed_lower > proposed_upper || !isfinite(new_width)
            || new_width >= old_width) break;

        double proposed_lower_value_lower = 0.0;
        double proposed_lower_value_upper = 0.0;
        double proposed_upper_value_lower = 0.0;
        double proposed_upper_value_upper = 0.0;
        status = evaluate_interval(
            e, proposed_lower, proposed_lower,
            &proposed_lower_value_lower, &proposed_lower_value_upper, NULL, NULL
        );
        if (status != 0) return 2;
        status = evaluate_interval(
            e, proposed_upper, proposed_upper,
            &proposed_upper_value_lower, &proposed_upper_value_upper, NULL, NULL
        );
        if (status != 0) return 2;
        if (strict_sign(proposed_lower_value_lower, proposed_lower_value_upper)
                != lower_sign
            || strict_sign(proposed_upper_value_lower, proposed_upper_value_upper)
                != upper_sign) {
            break;
        }
        lo = proposed_lower;
        hi = proposed_upper;
        lo_value_lower = proposed_lower_value_lower;
        lo_value_upper = proposed_lower_value_upper;
        hi_value_lower = proposed_upper_value_lower;
        hi_value_upper = proposed_upper_value_upper;
        ++newton_count;
    }

    while (nextafter(lo, hi) < hi) {
        if (bisection_count >= maximum_iterations) return 4;
        double midpoint = lo + 0.5 * (hi - lo);
        if (!(midpoint > lo && midpoint < hi) || !isfinite(midpoint)) return 4;
        double middle_lower = 0.0;
        double middle_upper = 0.0;
        status = evaluate_interval(
            e, midpoint, midpoint, &middle_lower, &middle_upper, NULL, NULL
        );
        if (status != 0) return 2;
        int middle_sign = strict_sign(middle_lower, middle_upper);
        ++bisection_count;
        if (middle_sign == 0) {
            double predecessor = nextafter(midpoint, lo);
            double successor = nextafter(midpoint, hi);
            double predecessor_lower = 0.0;
            double predecessor_upper = 0.0;
            double successor_lower = 0.0;
            double successor_upper = 0.0;
            if (!(predecessor < midpoint && midpoint < successor)) return 5;
            status = evaluate_interval(
                e, predecessor, predecessor,
                &predecessor_lower, &predecessor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            status = evaluate_interval(
                e, successor, successor,
                &successor_lower, &successor_upper, NULL, NULL
            );
            if (status != 0) return 2;
            if (strict_sign(predecessor_lower, predecessor_upper) != lower_sign
                || strict_sign(successor_lower, successor_upper) != upper_sign) return 5;
            lo = predecessor;
            hi = successor;
            lo_value_lower = predecessor_lower;
            lo_value_upper = predecessor_upper;
            hi_value_lower = successor_lower;
            hi_value_upper = successor_upper;
            exact_midpoint_bracket = 1;
            break;
        }
        if (middle_sign == lower_sign) {
            lo = midpoint;
            lo_value_lower = middle_lower;
            lo_value_upper = middle_upper;
        } else if (middle_sign == upper_sign) {
            hi = midpoint;
            hi_value_lower = middle_lower;
            hi_value_upper = middle_upper;
        } else {
            return 5;
        }
    }
    if (!exact_midpoint_bracket && nextafter(lo, hi) < hi) return 4;
    *root_lower = lo;
    *root_upper = hi;
    *lower_value_lower = lo_value_lower;
    *lower_value_upper = lo_value_upper;
    *upper_value_lower = hi_value_lower;
    *upper_value_upper = hi_value_upper;
    *interpolation_iterations = interpolation_count;
    *newton_iterations = newton_count;
    *bisection_iterations = bisection_count;
    return 0;
}

void carts_mpfr_point_plane_destroy(void *handle) {
    evaluator_t *e = (evaluator_t *)handle;
    if (!e) return;
    for (int index = 0; index < e->joint_count; ++index) {
        iv_clear(&e->joint_starts[index]); iv_clear(&e->joint_directions[index]);
        for (int cell = 0; cell < 12; ++cell) iv_clear(&e->origins[index * 12 + cell]);
    }
    for (int index = 0; index < 8; ++index) mpfr_clear(e->scalar[index]);
    for (int index = 0; index < 32; ++index) iv_clear(&e->local[index]);
    for (int index = 0; index < 12; ++index) {
        iv_clear(&e->base[index]); iv_clear(&e->work_a[index]);
        iv_clear(&e->work_b[index]); iv_clear(&e->work_c[index]);
    }
    for (int index = 0; index < 9; ++index) iv_clear(&e->rotation[index]);
    for (int index = 0; index < 3; ++index) {
        iv_clear(&e->witness[index]); iv_clear(&e->plane_origin[index]); iv_clear(&e->plane_area[index]);
    }
    iv_clear(&e->plane_offset);
    free(e->joint_types); free(e->axes); free(e->joint_starts);
    free(e->joint_directions); free(e->origins);
    free(e);
}
