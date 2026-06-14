#ifndef FLAPPY_PUFFERLIB_COMPAT_TENSOR_H
#define FLAPPY_PUFFERLIB_COMPAT_TENSOR_H

#include <stdint.h>

#define PUF_MAX_DIMS 8

typedef struct FloatTensor {
    float *data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct ByteTensor {
    unsigned char *data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

#endif
