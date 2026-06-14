#ifndef FLAPPY_POLICY_H
#define FLAPPY_POLICY_H

#include <stdbool.h>

#include "flappy_env.h"

typedef struct FlappyPolicy {
    int hidden_size;
    int network_layers;
    float pipe_gap;
    bool has_pipe_gap;
    bool legacy_encoder_gelu;
    float *encoder_weight;
    float *encoder_bias;
    float *network_weight;
    float *network_bias;
    float *decoder_weight;
    float *decoder_bias;
} FlappyPolicy;

void flappy_policy_init(FlappyPolicy *policy);
bool flappy_policy_load(FlappyPolicy *policy, const char *path);
void flappy_policy_unload(FlappyPolicy *policy);
FlappyAction flappy_policy_action(
    const FlappyPolicy *policy,
    const float observation[FLAPPY_OBSERVATION_SIZE],
    float logits[2]
);

#endif
