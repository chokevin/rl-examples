#ifndef FLAPPY_PUFFERLIB_COMPAT_VECENV_H
#define FLAPPY_PUFFERLIB_COMPAT_VECENV_H

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

typedef struct DictItem {
    const char *key;
    double value;
} DictItem;

typedef struct Dict {
    DictItem *items;
    int size;
    int capacity;
} Dict;

static inline Dict *create_dict(int capacity) {
    Dict *dict = (Dict *)calloc(1, sizeof(Dict));
    dict->capacity = capacity;
    dict->items = (DictItem *)calloc((size_t)capacity, sizeof(DictItem));
    return dict;
}

static inline DictItem *dict_get_unsafe(Dict *dict, const char *key) {
    for (int i = 0; i < dict->size; ++i) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem *dict_get(Dict *dict, const char *key) {
    DictItem *item = dict_get_unsafe(dict, key);
    if (item == NULL) {
        fprintf(stderr, "dict_get failed to find key: %s\n", key);
    }
    assert(item != NULL);
    return item;
}

static inline void dict_set(Dict *dict, const char *key, double value) {
    DictItem *item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->value = value;
        return;
    }
    assert(dict->size < dict->capacity);
    dict->items[dict->size].key = key;
    dict->items[dict->size].value = value;
    dict->size += 1;
}

typedef struct StaticVec {
    void *envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int *buffer_env_starts;
    int *buffer_env_counts;
    void *observations;
    float *actions;
    float *rewards;
    float *terminals;
    void *gpu_observations;
    float *gpu_actions;
    float *gpu_rewards;
    float *gpu_terminals;
    int obs_size;
    int num_atns;
    int gpu;
} StaticVec;

StaticVec *create_static_vec(
    int total_agents,
    int num_buffers,
    int gpu,
    Dict *vec_kwargs,
    Dict *env_kwargs
);
void static_vec_reset(StaticVec *vec);
void static_vec_close(StaticVec *vec);
void static_vec_log(StaticVec *vec, Dict *out);
void static_vec_eval_log(StaticVec *vec, Dict *out);
void static_vec_render(StaticVec *vec, int env_id);
int get_obs_size(void);
int get_num_atns(void);
int *get_act_sizes(void);
int get_num_act_sizes(void);
const char *get_obs_dtype(void);
size_t get_obs_elem_size(void);
void cpu_vec_step(StaticVec *vec);

#ifdef OBS_SIZE

#define FLAPPY_STRINGIFY_INNER(x) #x
#define FLAPPY_STRINGIFY(x) FLAPPY_STRINGIFY_INNER(x)

static const char dtype_symbol[] = FLAPPY_STRINGIFY(OBS_TENSOR_T);

static inline size_t obs_element_size(void) {
    OBS_TENSOR_T tensor;
    return sizeof(*tensor.data);
}

void my_init(Env *env, Dict *kwargs);
void my_log(Log *log, Dict *out);

static Env *my_vec_init(
    int *num_envs_out,
    int *buffer_env_starts,
    int *buffer_env_counts,
    Dict *vec_kwargs,
    Dict *env_kwargs
) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    assert(total_agents > 0);
    assert(num_buffers > 0);
    assert(total_agents % num_buffers == 0);

    Env *envs = (Env *)calloc((size_t)total_agents, sizeof(Env));
    int agents_per_buffer = total_agents / num_buffers;
    for (int i = 0; i < total_agents; ++i) {
        envs[i].rng = (unsigned int)i;
        my_init(&envs[i], env_kwargs);
    }

    for (int buf = 0; buf < num_buffers; ++buf) {
        buffer_env_starts[buf] = buf * agents_per_buffer;
        buffer_env_counts[buf] = agents_per_buffer;
    }
    *num_envs_out = total_agents;
    return envs;
}

StaticVec *create_static_vec(
    int total_agents,
    int num_buffers,
    int gpu,
    Dict *vec_kwargs,
    Dict *env_kwargs
) {
    (void)gpu;
    StaticVec *vec = (StaticVec *)calloc(1, sizeof(StaticVec));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->obs_size = OBS_SIZE;
    vec->num_atns = NUM_ATNS;
    vec->gpu = 0;
    vec->buffer_env_starts = (int *)calloc((size_t)num_buffers, sizeof(int));
    vec->buffer_env_counts = (int *)calloc((size_t)num_buffers, sizeof(int));

    int num_envs = 0;
    vec->envs = my_vec_init(
        &num_envs,
        vec->buffer_env_starts,
        vec->buffer_env_counts,
        vec_kwargs,
        env_kwargs
    );
    vec->size = num_envs;

    vec->observations = calloc((size_t)total_agents * OBS_SIZE, obs_element_size());
    vec->actions = (float *)calloc((size_t)total_agents * NUM_ATNS, sizeof(float));
    vec->rewards = (float *)calloc((size_t)total_agents, sizeof(float));
    vec->terminals = (float *)calloc((size_t)total_agents, sizeof(float));
    vec->gpu_observations = vec->observations;
    vec->gpu_actions = vec->actions;
    vec->gpu_rewards = vec->rewards;
    vec->gpu_terminals = vec->terminals;

    Env *envs = (Env *)vec->envs;
    for (int i = 0; i < vec->size; ++i) {
        envs[i].observations = (float *)vec->observations + (size_t)i * OBS_SIZE;
        envs[i].actions = vec->actions + (size_t)i * NUM_ATNS;
        envs[i].rewards = vec->rewards + i;
        envs[i].terminals = vec->terminals + i;
    }
    return vec;
}

void static_vec_reset(StaticVec *vec) {
    Env *envs = (Env *)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->total_agents * sizeof(float));
    for (int i = 0; i < vec->size; ++i) {
        c_reset(&envs[i]);
    }
}

void cpu_vec_step(StaticVec *vec) {
    Env *envs = (Env *)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->total_agents * sizeof(float));
    for (int i = 0; i < vec->size; ++i) {
        c_step(&envs[i]);
    }
}

static float aggregate_logs(StaticVec *vec, Log *out) {
    Env *envs = (Env *)vec->envs;
    memset(out, 0, sizeof(Log));
    int num_fields = (int)(sizeof(Log) / sizeof(float));
    for (int i = 0; i < vec->size; ++i) {
        if (envs[i].log.n == 0.0f) {
            continue;
        }
        for (int field = 0; field < num_fields; ++field) {
            ((float *)out)[field] += ((float *)&envs[i].log)[field];
        }
    }

    float n = out->n;
    if (n == 0.0f) {
        return 0.0f;
    }
    for (int field = 0; field < num_fields; ++field) {
        ((float *)out)[field] /= n;
    }
    return n;
}

void static_vec_log(StaticVec *vec, Dict *out) {
    Env *envs = (Env *)vec->envs;
    Log aggregate;
    float n = aggregate_logs(vec, &aggregate);
    if (n == 0.0f) {
        return;
    }
    for (int i = 0; i < vec->size; ++i) {
        memset(&envs[i].log, 0, sizeof(Log));
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_eval_log(StaticVec *vec, Dict *out) {
    Log aggregate;
    float n = aggregate_logs(vec, &aggregate);
    if (n == 0.0f) {
        return;
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_render(StaticVec *vec, int env_id) {
    Env *envs = (Env *)vec->envs;
    c_render(&envs[env_id]);
}

void static_vec_close(StaticVec *vec) {
    Env *envs = (Env *)vec->envs;
    for (int i = 0; i < vec->size; ++i) {
        c_close(&envs[i]);
    }
    free(vec->envs);
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);
    free(vec->observations);
    free(vec->actions);
    free(vec->rewards);
    free(vec->terminals);
    free(vec);
}

int get_obs_size(void) {
    return OBS_SIZE;
}

int get_num_atns(void) {
    return NUM_ATNS;
}

static int act_sizes[] = ACT_SIZES;

int *get_act_sizes(void) {
    return act_sizes;
}

int get_num_act_sizes(void) {
    return (int)(sizeof(act_sizes) / sizeof(act_sizes[0]));
}

const char *get_obs_dtype(void) {
    return dtype_symbol;
}

size_t get_obs_elem_size(void) {
    return obs_element_size();
}

#endif

#ifdef __cplusplus
}
#endif

#endif
