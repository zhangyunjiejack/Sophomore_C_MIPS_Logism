#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include "timestamp.c"

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

// Vol ------------------------------------------------------------------------

// Volumes are used to represent the activations (i.e., state) between the
// different layers of the CNN. They all have three dimensions. The inter-
// pretation of their content depends on the layer that produced them. Before
// the first iteration, the Volume holds the data of the image we want to
// classify (the depth are the three color dimensions). After the last stage
// of the CNN, the Volume holds the probabilities that an image is part of
// a specific category.

/*
 * Represents a three-dimensional array of numbers, and its size. The numbers
 * at (x,y,d) are stored in array w at location ((v->sx * y)+x)*v->depth+d.
 */

typedef struct vol {
  uint64_t sx,sy,depth;
  double* w;
} vol_t;

/*
 * Set the value at a specific entry of the array.
 */

static inline double get_vol(vol_t* v, int x, int y, int d) {
  return v->w[((v->sx * y)+x)*v->depth+d];
}

/*
 * Get the value at a specific entry of the array.
 */

static inline void set_vol(vol_t* v, int x, int y, int d, double val) {
  v->w[((v->sx * y)+x)*v->depth+d] = val;
}

/*
 * Allocate a new array with specific dimensions and default value v.
 */

static vol_t* make_vol(int sx, int sy, int d, double v) {
  vol_t* out = (vol_t*)malloc(sizeof(struct vol));
  out->w = (double*)malloc(sizeof(double)*(sx*sy*d));
  out->sx = sx;
  out->sy = sy;
  out->depth = d;

  for (int x = 0; x < sx; x++)
    for (int y = 0; y < sy; y++)
      for (int z = 0; z < d; z++)
        set_vol(out, x, y, z, v);

  return out;
}

/*
 * Copy the contents of one Volume to another (assuming same dimensions).
 */

static vol_t* copy_vol(vol_t* dest, vol_t* src) {
  for (int x = 0; x < dest->sx; x++)
    for (int y = 0; y < dest->sy; y++)
      for (int z = 0; z < dest->depth; z++)
        set_vol(dest, x, y, z, get_vol(src, x, y, z));
}

/*
 * Deallocate the array.
 */
void free_vol(vol_t* v) {
  free(v->w);
  free(v);
}

// A note about layers --------------------------------------------------------

/*
 * What follows are the different layers of the CNN. You will not have to
 * understand what these layers are actually doing. In general terms, each
 * layer performs a "forward" operation on a batch of inputs. During this
 * forward operation, the layer takes a set of input Volumes and transforms
 * them into a set of output Volumes (one output for each input). What differs
 * is the operation performed by each layer.
 *
 * In addition to the _forward function, each layer also provides a data
 * structure, holding (fixed) parameters for that layer, a make_ function to
 * allocate an instance of the layer with a particular set of parameters and
 * a load function to load training data for that layer from a file. Note that
 * you will not have to make any changes to any of these functions. The only
 * function you need to consider is the _forward function.
 */

// Convolutional Layer --------------------------------------------------------

typedef struct conv_layer {
  // required
  int out_depth;
  int sx;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  int sy;
  int stride;
  int pad;
  double l1_decay_mul;
  double l2_decay_mul;

  // computed
  int out_sx;
  int out_sy;
  double bias;
  vol_t* biases;
  vol_t** filters;
} conv_layer_t;

conv_layer_t* make_conv_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int filters, int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  // required
  l->out_depth = filters;
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = pad;
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*filters);
  for (int i = 0; i < filters; i++) {
    l->filters[i] = make_vol(l->sx, l->sy, l->in_depth, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  return l;
}


static void depth_3(int x, int y, int default_x, int sy_out, int xy_stride,
                    int sx_out, int sy_f, int sx_f, int V_sx, int depth_V, int V_sy, double* w_of_f,
                    double* V_w, int d, double* l_biases_w, vol_t* A) {
    int ay;
    for(ay = 0; ay < sy_out; y += xy_stride, ay++) {
        x = default_x;

        int ax;
        for(ax=0; ax < sx_out; x += xy_stride, ax++) {
            double a = 0.0;

            int fy;

            for(fy = 0; fy < sy_f; fy++) {
                int oy = y + fy;

                int sx_fy = sx_f * fy;
                int sx_oy = V_sx * oy;

                int fx;

                for (fx = 0; fx < sx_f; fx++) {
                    int ox = x + fx;

                    int tmp_val1 = (sx_fy + fx) * 3;
                    int tmp_val2 = (sx_oy + ox) * depth_V;

                    if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

                        a += w_of_f[tmp_val1] * V_w[tmp_val2];
                        a += w_of_f[tmp_val1 + 1] * V_w[tmp_val2 + 1];
                        a += w_of_f[tmp_val1 + 2] * V_w[tmp_val2 + 2];


                    }
                }
            }

            a += l_biases_w[d];
            set_vol(A, ax, ay, d, a);

        }
    }
}

static void depth_16(int x, int y, int default_x, int sy_out, int xy_stride,
                    int sx_out, int sy_f, int sx_f, int V_sx, int depth_V, int V_sy, double* w_of_f,
                    double* V_w, int d, double* l_biases_w, vol_t* A) {
    int ay;
    for(ay = 0; ay < sy_out; y += xy_stride, ay++) {
        x = default_x;

        int ax;
        for(ax=0; ax < sx_out; x += xy_stride, ax++) {
            double a = 0.0;

            int fy;

            for(fy = 0; fy < sy_f; fy++) {
                int oy = y + fy;

                int sx_fy = sx_f * fy;
                int sx_oy = V_sx * oy;

                int fx;

                for (fx = 0; fx < sx_f; fx++) {
                    int ox = x + fx;

                    int tmp_val1 = (sx_fy + fx) * 16;
                    int tmp_val2 = (sx_oy + ox) * depth_V;

                    if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

                        __m256d temp_prod = _mm256_setzero_pd();
                        __m256d f_temp, v_temp;
                        double S[4] = {0, 0, 0, 0};

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 4));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 4));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 8));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 8));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 12));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 12));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        _mm256_storeu_pd((__m256 *) S, temp_prod);
                        a += S[0] + S[1] + S[2] + S[3];

                    }
                }
            }

            a += l_biases_w[d];
            set_vol(A, ax, ay, d, a);

        }
    }
}

static void depth_20(int x, int y, int default_x, int sy_out, int xy_stride,
                     int sx_out, int sy_f, int sx_f, int V_sx, int depth_V, int V_sy, double* w_of_f,
                     double* V_w, int d, double* l_biases_w, vol_t* A) {
    int ay;
    for(ay = 0; ay < sy_out; y += xy_stride, ay++) {
        x = default_x;

        int ax;
        for(ax=0; ax < sx_out; x += xy_stride, ax++) {
            double a = 0.0;

            int fy;

            for(fy = 0; fy < sy_f; fy++) {
                int oy = y + fy;

                int sx_fy = sx_f * fy;
                int sx_oy = V_sx * oy;

                int fx;

                for (fx = 0; fx < sx_f; fx++) {
                    int ox = x + fx;

                    int tmp_val1 = (sx_fy + fx) * 20;
                    int tmp_val2 = (sx_oy + ox) * depth_V;

                    if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

                        __m256d temp_prod = _mm256_setzero_pd();
                        __m256d f_temp, v_temp;
                        double S[4] = {0, 0, 0, 0};

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 4));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 4));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 8));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 8));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 12));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 12));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 16));
                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 16));
                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        _mm256_storeu_pd((__m256 *) S, temp_prod);
                        a += S[0] + S[1] + S[2] + S[3];

                    }
                }
            }

            a += l_biases_w[d];
            set_vol(A, ax, ay, d, a);

        }
    }
}

static void general_depth(int x, int y, int default_x, int sy_out, int xy_stride,
                          int sx_out, int sy_f, int sx_f, int V_sx, int depth_f, int depth_V, int V_sy, double* w_of_f,
                          double* V_w, int d, double* l_biases_w, vol_t* A) {
    int ay;
    for(ay = 0; ay < sy_out; y += xy_stride, ay++) {
        x = default_x;

        int ax;
        for(ax=0; ax < sx_out; x += xy_stride, ax++) {
            double a = 0.0;

            int fy;

            for(fy = 0; fy < sy_f; fy++) {
                int oy = y + fy;

                int sx_fy = sx_f * fy;
                int sx_oy = V_sx * oy;

                int fx;

                for (fx = 0; fx < sx_f; fx++) {
                    int ox = x + fx;

                    int tmp_val1 = (sx_fy + fx) * depth_f;
                    int tmp_val2 = (sx_oy + ox) * depth_V;

                    if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {

                        __m256d temp_prod = _mm256_setzero_pd();
                        __m256d f_temp, v_temp;
                        double S[4] = {0, 0, 0, 0};

                        int fd;
                        for (fd = 0; fd < depth_f / 4 * 4; fd += 4) {
                            f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + fd));
                            v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + fd));
                            temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));

                        }
                        int num;
                        for (num = depth_f / 4 * 4; num < depth_f; num += 1) {
                            a += w_of_f[tmp_val1 + num] * V_w[tmp_val2 + num];
                        }

                        _mm256_storeu_pd((__m256 *) S, temp_prod);
                        a += S[0] + S[1] + S[2] + S[3];


                    }
                }
            }

            a += l_biases_w[d];
            set_vol(A, ax, ay, d, a);

        }
    }
}




void conv_forward(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {

    int xy_stride = l->stride;
    int depth_out = l->out_depth;
    int sy_out = l->out_sy;
    int sx_out = l->out_sx;
    double* l_biases_w = l->biases->w;
    vol_t** filter_l = l->filters;
    int default_x = -l->pad;
    int default_y = -l->pad;
    int x;
    int y;
    int sy_f;
    int sx_f;
    int depth_f;
    double* w_of_f;
    int i;

    for (i = start; i <= end; i++) {
        vol_t* V = in[i];
        vol_t* A = out[i];

        int depth_V = V->depth;
        double* V_w = V->w;
        int V_sx = V->sx;
        int V_sy = V->sy;

        int d;
        for(d = 0; d < depth_out; d++) {
            vol_t* f = filter_l[d];

            sy_f = f->sy;
            sx_f = f->sx;
            depth_f = f->depth;
            w_of_f = f->w;

            y = default_y;

            if (depth_f == 3) {
                depth_3(x, y, default_x, sy_out, xy_stride, sx_out, sy_f, sx_f, V_sx,
                        depth_V, V_sy, w_of_f, V_w, d, l_biases_w, A);
            } else if (depth_f == 16) {
                depth_16(x, y, default_x, sy_out, xy_stride, sx_out, sy_f, sx_f, V_sx,
                        depth_V, V_sy, w_of_f, V_w, d, l_biases_w, A);
            } else if (depth_f == 20){
                depth_20(x, y, default_x, sy_out, xy_stride, sx_out, sy_f, sx_f, V_sx,
                         depth_V, V_sy, w_of_f, V_w, d, l_biases_w, A);
            } else {
                general_depth(x, y, default_x, sy_out, xy_stride, sx_out, sy_f, sx_f, V_sx,
                         depth_f, depth_V, V_sy, w_of_f, V_w, d, l_biases_w, A);
            }



//            int ay;
//            for(ay = 0; ay < sy_out; y += xy_stride, ay++) {
//                x = default_x;
//
//                int ax;
//                for(ax=0; ax < sx_out; x += xy_stride, ax++) {
//                    a = 0.0;
//
//                    int fy;
//
//                    for(fy = 0; fy < sy_f; fy++) {
//                        oy = y + fy;
//
//                        int sx_fy = sx_f * fy;
//                        int sx_oy = V_sx * oy;
//
//                        int fx;
//
//                        for (fx = 0; fx < sx_f; fx++) {
//                            ox = x + fx;
//
//                                int tmp_val1 = (sx_fy + fx) * depth_f;
//                                int tmp_val2 = (sx_oy + ox) * depth_V;
//
//                            if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {
//
//                                __m256d temp_prod = _mm256_setzero_pd();
//                                __m256d f_temp, v_temp;
//                                double S[4] = {0, 0, 0, 0};

//                                if (depth_f == 3) {
//                                    a += w_of_f[tmp_val1] * V_w[tmp_val2];
//                                    a += w_of_f[tmp_val1 + 1] * V_w[tmp_val2 + 1];
//                                    a += w_of_f[tmp_val1 + 2] * V_w[tmp_val2 + 2];
//                                } else if (depth_f == 16) {
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 4));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 4));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 8));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 8));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 12));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 12));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    _mm256_storeu_pd((__m256 *) S, temp_prod);
//                                    a += S[0] + S[1] + S[2] + S[3];
//
//
//                                } else if (depth_f == 20) {
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 4));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 4));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 8));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 8));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 12));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 12));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + 16));
//                                    v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + 16));
//                                    temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//
//                                    _mm256_storeu_pd((__m256 *) S, temp_prod);
//                                    a += S[0] + S[1] + S[2] + S[3];
//
//                                } else {
//                                    int fd;
//                                    for (fd = 0; fd < depth_f / 4 * 4; fd += 4) {
//                                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + fd));
//                                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + fd));
//                                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//
//                                    }
//                                    for (fd = depth_f / 4 * 4; fd < depth_f; fd += 1) {
//                                        f_temp = _mm256_loadu_pd((__m256d *) (w_of_f + tmp_val1 + fd));
//                                        v_temp = _mm256_loadu_pd((__m256d *) (V_w + tmp_val2 + fd));
//                                        temp_prod = _mm256_add_pd(temp_prod, _mm256_mul_pd(f_temp, v_temp));
//                                    }
//                                    _mm256_storeu_pd((__m256 *) S, temp_prod);
//                                    a += S[0] + S[1] + S[2] + S[3];
//
//                                }
//                            }
//                        }
//                    }
//
//                    a += l_biases_w[d];
//                    set_vol(A, ax, ay, d, a);
//                }
//            }
        }
    }
}





void conv_load(conv_layer_t* l, const char* fn) {

    #pragma omp parallel
    {
        int sx, sy, depth, filters;

        vol_t **filters_l = l->filters;
        int out_depth_l = l->out_depth;
        int biases_l = l->biases;


        FILE *fin = fopen(fn, "r");

        fscanf(fin, "%d %d %d %d", &sx, &sy, &depth, &filters);
        assert(sx == l->sx);
        assert(sy == l->sy);
        assert(depth == l->in_depth);
        assert(filters == out_depth_l);

        int d;
        #pragma omp parallel for

        for (d = 0; d < out_depth_l; d++) {
            int x;
            for (x = 0; x < sx; x++) {
                int y;
                for (y = 0; y < sy; y++) {
                    int z;
                    for (z = 0; z < depth; z++) {
                        double val;
                        fscanf(fin, "%lf", &val);
                        set_vol(filters_l[d], x, y, z, val);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (d = 0; d < out_depth_l; d++) {
            double val;
            fscanf(fin, "%lf", &val);
            set_vol(biases_l, 0, 0, d, val);
        }

        fclose(fin);
    }
}

// Relu Layer -----------------------------------------------------------------

typedef struct relu_layer {
  // required
  int in_depth;
  int in_sx;
  int in_sy;

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} relu_layer_t;

relu_layer_t* make_relu_layer(int in_sx, int in_sy, int in_depth) {
  relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = l->in_sx;
  l->out_sy = l->in_sy;
  l->out_depth = l->in_depth;

  return l;
}

void relu_forward(relu_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int j = start; j <= end; j++) {
    for (int i = 0; i < l->in_sx*l->in_sy*l->in_depth; i++) {
      out[j]->w[i] = (in[j]->w[i] < 0.0) ? 0.0 : in[j]->w[i];
    }
  }
}

// Pool Layer -----------------------------------------------------------------

typedef struct pool_layer {
  // required
  int sx;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  int sy;
  int stride;
  int pad;

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} pool_layer_t;

pool_layer_t* make_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

  // required
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = 0;

  // computed
  l->out_depth = in_depth;
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  return l;
}

void pool_forward(pool_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];

    int n=0;
    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax<l->out_sx; x+=l->stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay<l->out_sy; y+=l->stride,ay++) {

          double a = -99999;
          for(int fx=0;fx<l->sx;fx++) {
            for(int fy=0;fy<l->sy;fy++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy>=0 && oy<V->sy && ox>=0 && ox<V->sx) {
                double v = get_vol(V, ox, oy, d);
                if(v > a) { a = v; }
              }
            }
          }
          n++;
          set_vol(A, ax, ay, d, a);
        }
      }
    }
  }
}

// FC Layer -------------------------------------------------------------------

typedef struct fc_layer {
  // required
  int out_depth;
  int in_depth;
  int in_sx;
  int in_sy;

  // optional
  double l1_decay_mul;
  double l2_decay_mul;

  // computed
  int out_sx;
  int out_sy;
  int num_inputs;
  double bias;
  vol_t* biases;
  vol_t** filters;
} fc_layer_t;

fc_layer_t* make_fc_layer(int in_sx, int in_sy, int in_depth,
                          int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  // required
  l->out_depth = num_neurons;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // optional
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->num_inputs = l->in_sx * l->in_sy * l->in_depth;
  l->out_sx = 1;
  l->out_sy = 1;

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*num_neurons);
  for (int i = 0; i < l->out_depth; i++) {
    l->filters[i] = make_vol(1, 1, l->num_inputs, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  return l;
}

void fc_forward(fc_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];

    for(int i=0;i<l->out_depth;i++) {
      double a = 0.0;
      for(int d=0;d<l->num_inputs;d++) {
        a += V->w[d] * l->filters[i]->w[d];
      }
      a += l->biases->w[i];
      A->w[i] = a;
    }
  }
}

void fc_load(fc_layer_t* l, const char* fn) {
  FILE* fin = fopen(fn, "r");

  int num_inputs;
  int out_depth;
  fscanf(fin, "%d %d", &num_inputs, &out_depth);
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val;
      fscanf(fin, "%lf", &val);
      l->filters[i]->w[d] = val;
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val;
    fscanf(fin, "%lf", &val);
    l->biases->w[i] = val;
  }

  fclose(fin);
}

// Softmax Layer --------------------------------------------------------------

// Maximum supported out_depth
#define MAX_ES 16

typedef struct softmax_layer {
  // required
  int in_depth;
  int in_sx;
  int in_sy;
  double* es;

  // computed
  int out_depth;
  int out_sx;
  int out_sy;
} softmax_layer_t;

softmax_layer_t* make_softmax_layer(int in_sx, int in_sy, int in_depth) {
  softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = 1;
  l->out_sy = 1;
  l->out_depth = l->in_sx * l->in_sy * l->in_depth;

  l->es = (double*)malloc(sizeof(double)*l->out_depth);

  return l;
}

void softmax_forward(softmax_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  double es[MAX_ES];

  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];

    // compute max activation
    double amax = V->w[0];
    for(int i=1;i<l->out_depth;i++) {
      if(V->w[i] > amax) amax = V->w[i];
    }

    // compute exponentials (carefully to not blow up)
    double esum = 0.0;
    for(int i=0;i<l->out_depth;i++) {
      double e = exp(V->w[i] - amax);
      esum += e;
      es[i] = e;
    }

    // normalize and output to sum to one
    for(int i=0;i<l->out_depth;i++) {
      es[i] /= esum;
      A->w[i] = es[i];
    }
  }
}

// Neural Network -------------------------------------------------------------

/*
 * This represents the CNN we will use in this project. It consists of 11
 * layers, which means that there are 12 volumes of data (where the first one
 * is the input data and the last one the classification result).
 */

#define LAYERS 11

typedef struct network {
//#pragma omp parallel
  vol_t* v[LAYERS+1];
  conv_layer_t* l0;
  relu_layer_t* l1;
  pool_layer_t* l2;
  conv_layer_t* l3;
  relu_layer_t* l4;
  pool_layer_t* l5;
  conv_layer_t* l6;
  relu_layer_t* l7;
  pool_layer_t* l8;
  fc_layer_t* l9;
  softmax_layer_t* l10;
} network_t;

/*
 * Instantiate our specific CNN.
 */

network_t* make_network() {
//#pragma omp parallel

  network_t* net = (network_t*)malloc(sizeof(network_t));
    vol_t** net_v = net->v;
  net_v[0] = make_vol(32, 32, 3, 0.0);
  net->l0 = make_conv_layer(32, 32, 3, 5, 16, 1, 2);

  net_v[1] = make_vol(net->l0->out_sx, net->l0->out_sy, net->l0->out_depth, 0.0);
  net->l1 = make_relu_layer(net->v[1]->sx, net->v[1]->sy, net->v[1]->depth);
  net_v[2] = make_vol(net->l1->out_sx, net->l1->out_sy, net->l1->out_depth, 0.0);
  net->l2 = make_pool_layer(net->v[2]->sx, net->v[2]->sy, net->v[2]->depth, 2, 2);
  net_v[3] = make_vol(net->l2->out_sx, net->l2->out_sy, net->l2->out_depth, 0.0);
  net->l3 = make_conv_layer(net->v[3]->sx, net->v[3]->sy, net->v[3]->depth, 5, 20, 1, 2);
  net_v[4] = make_vol(net->l3->out_sx, net->l3->out_sy, net->l3->out_depth, 0.0);
  net->l4 = make_relu_layer(net->v[4]->sx, net->v[4]->sy, net->v[4]->depth);
  net_v[5] = make_vol(net->l4->out_sx, net->l4->out_sy, net->l4->out_depth, 0.0);
  net->l5 = make_pool_layer(net->v[5]->sx, net->v[5]->sy, net->v[5]->depth, 2, 2);
  net_v[6] = make_vol(net->l5->out_sx, net->l5->out_sy, net->l5->out_depth, 0.0);
  net->l6 = make_conv_layer(net->v[6]->sx, net->v[6]->sy, net->v[6]->depth, 5, 20, 1, 2);
  net_v[7] = make_vol(net->l6->out_sx, net->l6->out_sy, net->l6->out_depth, 0.0);
  net->l7 = make_relu_layer(net->v[7]->sx, net->v[7]->sy, net->v[7]->depth);
  net_v[8] = make_vol(net->l7->out_sx, net->l7->out_sy, net->l7->out_depth, 0.0);
  net->l8 = make_pool_layer(net->v[8]->sx, net->v[8]->sy, net->v[8]->depth, 2, 2);
  net_v[9] = make_vol(net->l8->out_sx, net->l8->out_sy, net->l8->out_depth, 0.0);
  net->l9 = make_fc_layer(net->v[9]->sx, net->v[9]->sy, net->v[9]->depth, 10);
  net_v[10] = make_vol(net->l9->out_sx, net->l9->out_sy, net->l9->out_depth, 0.0);
  net->l10 = make_softmax_layer(net->v[10]->sx, net->v[10]->sy, net->v[10]->depth);
  net_v[11] = make_vol(net->l10->out_sx, net->l10->out_sy, net->l10->out_depth, 0.0);
  return net;
}

/*
 * Free our specific CNN.
 */

void free_network(network_t* net) {
    int i;

#pragma omp parallel for
  for (i = 0; i < LAYERS+1; i++)
      free_vol(net->v[i]);

      free(net->l0);
      free(net->l1);
      free(net->l2);
      free(net->l3);
      free(net->l4);
      free(net->l5);
      free(net->l6);
      free(net->l7);
      free(net->l8);
      free(net->l9);
      free(net->l10);

  free(net);
}

/*
 * We organize data as "batches" of volumes. Each batch consists of a number of samples,
 * each of which contains a volume for every intermediate layer. Say we have L layers
 * and a set of N input images. Then batch[l][n] contains the volume at layer l for
 * input image n.
 *
 * By using batches, we can process multiple images at once in each run of the forward
 * functions of the different layers.
 */

typedef vol_t** batch_t;

/*
 * This function allocates a new batch for the network old_net with size images.
 */

batch_t* make_batch(network_t* old_net, int size) {
  batch_t* out = (batch_t*)malloc(sizeof(vol_t**)*(LAYERS+1));
    int i, j;
    vol_t** old_net_v = old_net->v;
  for (i = 0; i < LAYERS+1; i++) {
      out[i] = (vol_t**)malloc(sizeof(vol_t*)*size);
      vol_t* old_net_v_i = old_net_v[i];
      vol_t** old_net_v_i_sx = old_net_v_i->sx;
      vol_t** old_net_v_i_sy = old_net_v_i->sy;
      vol_t** old_net_v_i_depth = old_net_v_i->depth;

      for (j = 0; j < size; j++) {
      out[i][j] = make_vol(old_net_v_i_sx, old_net_v_i_sy, old_net_v_i_depth, 0.0);
    }
  }

  return out;
}

/*
 * Free a previously allocated batch.
 */

void free_batch(batch_t* v, int size) {
    int i;
    int j;
#pragma omp parallel for
  for (i = 0; i < LAYERS+1; i++) {
    for (j = 0; j < size; j++) {
      free_vol(v[i][j]);
    }
    free(v[i]);
  }
  free(v);
}

/*
 * Apply our network to a specific batch of inputs. The batch has to be given
 * as input to v and start/end are the first and the last image in that batch
 * to process (start and end are inclusive).
 */

void net_forward(network_t* net, batch_t* v, int start, int end) {
    #pragma omp parallel
    {
//        printf("\n%d\n", timestamp_us());
        conv_forward(net->l0, v[0], v[1], start, end);
//        printf("\n%d\n", timestamp_us());
        relu_forward(net->l1, v[1], v[2], start, end);
//        printf("\n%d\n", timestamp_us());
        pool_forward(net->l2, v[2], v[3], start, end);
//        printf("\n%d\n", timestamp_us());
        conv_forward(net->l3, v[3], v[4], start, end);
//        printf("\n%d\n", timestamp_us());
        relu_forward(net->l4, v[4], v[5], start, end);
//        printf("\n%d\n", timestamp_us());
        pool_forward(net->l5, v[5], v[6], start, end);
//        printf("\n%d\n", timestamp_us());
        conv_forward(net->l6, v[6], v[7], start, end);
//        printf("\n%d\n", timestamp_us());
        relu_forward(net->l7, v[7], v[8], start, end);
//        printf("\n%d\n", timestamp_us());
        pool_forward(net->l8, v[8], v[9], start, end);
//        printf("\n%d\n", timestamp_us());
        fc_forward(net->l9, v[9], v[10], start, end);
//        printf("\n%d\n", timestamp_us());
        softmax_forward(net->l10, v[10], v[11], start, end);
//        printf("\n%d\n%s\n", timestamp_us(), "----------------------------------------------");
    }

}

/*
 * Putting everything together: Take a set of n input images as 3-dimensional
 * Volumes and process them using the CNN in batches of 1. Then look at the
 * output (which is a set of 10 labels, each of which tells us the likelihood
 * of a specific category) and classify the image as a cat iff the likelihood
 * of "cat" is larger than 50%. Writes the cat likelihood for all images into
 * an output array (0 = definitely no cat, 1 = definitely cat).
 */

#define CAT_LABEL 3
void net_classify_cats(network_t* net, vol_t** input, double* output, int n) {
        int i;
        #pragma omp parallel
            {
                batch_t* batch = make_batch(net, 1);
                #pragma omp for
                for (i = 0; i < n; i++) {
                    copy_vol(batch[0][0], input[i]);
                    net_forward(net, batch, 0, 0);
                    output[i] = batch[11][0]->w[CAT_LABEL];
                }
                free_batch(batch, 1);
            }


}

// IGNORE EVERYTHING BELOW THIS POINT -----------------------------------------

// Including C files in other C files is very bad style and should be avoided
// in any real application. We do it here since we want everything that you
// may edit to be in one file, without having to fix the interfaces between
// the different components of the system.

#include "util.c"
#include "main.c"
