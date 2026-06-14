#include "flappy_policy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void flappy_policy_init(FlappyPolicy *policy) {
    policy->hidden_size = 0;
    policy->network_layers = 0;
    policy->pipe_gap = 0.0f;
    policy->has_pipe_gap = false;
    policy->legacy_encoder_gelu = false;
    policy->encoder_weight = NULL;
    policy->encoder_bias = NULL;
    policy->network_weight = NULL;
    policy->network_bias = NULL;
    policy->decoder_weight = NULL;
    policy->decoder_bias = NULL;
}

void flappy_policy_unload(FlappyPolicy *policy) {
    free(policy->encoder_weight);
    free(policy->encoder_bias);
    free(policy->network_weight);
    free(policy->network_bias);
    free(policy->decoder_weight);
    free(policy->decoder_bias);
    flappy_policy_init(policy);
}

static bool read_label(FILE *file, const char *expected) {
    char label[64];
    return fscanf(file, "%63s", label) == 1 && strcmp(label, expected) == 0;
}

static bool read_float_array(FILE *file, const char *label, float *values, int count) {
    if (!read_label(file, label)) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        if (fscanf(file, "%f", &values[i]) != 1) {
            return false;
        }
    }
    return true;
}

bool flappy_policy_load(FlappyPolicy *policy, const char *path) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return false;
    }

    FlappyPolicy loaded;
    flappy_policy_init(&loaded);
    bool ok = false;

    char version[64];
    if (fscanf(file, "%63s", version) != 1) {
        goto done;
    }
    bool is_v1 = strcmp(version, "flappy_policy_v1") == 0;
    bool is_v2 = strcmp(version, "flappy_policy_v2") == 0;
    if (!is_v1 && !is_v2) {
        goto done;
    }

    if (!read_label(file, "hidden_size") || fscanf(file, "%d", &loaded.hidden_size) != 1) {
        goto done;
    }
    if (loaded.hidden_size < 1 || loaded.hidden_size > 4096) {
        goto done;
    }
    if (is_v2 &&
        (!read_label(file, "network_layers") ||
         fscanf(file, "%d", &loaded.network_layers) != 1)) {
        goto done;
    }
    if (loaded.network_layers < 0 || loaded.network_layers > 16) {
        goto done;
    }
    if (!read_label(file, "pipe_gap") || fscanf(file, "%f", &loaded.pipe_gap) != 1) {
        goto done;
    }
    loaded.has_pipe_gap = true;
    loaded.legacy_encoder_gelu = is_v1;

    int encoder_weight_count = loaded.hidden_size * FLAPPY_OBSERVATION_SIZE;
    int encoder_bias_count = loaded.hidden_size;
    int network_weight_count = loaded.network_layers * loaded.hidden_size * loaded.hidden_size;
    int network_bias_count = loaded.network_layers * loaded.hidden_size;
    int decoder_weight_count = 2 * loaded.hidden_size;
    int decoder_bias_count = 2;

    loaded.encoder_weight = (float *)calloc((size_t)encoder_weight_count, sizeof(float));
    loaded.encoder_bias = (float *)calloc((size_t)encoder_bias_count, sizeof(float));
    if (loaded.network_layers > 0) {
        loaded.network_weight = (float *)calloc((size_t)network_weight_count, sizeof(float));
        loaded.network_bias = (float *)calloc((size_t)network_bias_count, sizeof(float));
    }
    loaded.decoder_weight = (float *)calloc((size_t)decoder_weight_count, sizeof(float));
    loaded.decoder_bias = (float *)calloc((size_t)decoder_bias_count, sizeof(float));
    if (loaded.encoder_weight == NULL || loaded.encoder_bias == NULL ||
        (loaded.network_layers > 0 &&
         (loaded.network_weight == NULL || loaded.network_bias == NULL)) ||
        loaded.decoder_weight == NULL || loaded.decoder_bias == NULL) {
        goto done;
    }

    ok = read_float_array(file, "encoder_weight", loaded.encoder_weight, encoder_weight_count) &&
         read_float_array(file, "encoder_bias", loaded.encoder_bias, encoder_bias_count);
    for (int layer = 0; ok && layer < loaded.network_layers; ++layer) {
        char label[64];
        snprintf(label, sizeof(label), "network_%d_weight", layer);
        ok = read_float_array(
            file,
            label,
            loaded.network_weight + layer * loaded.hidden_size * loaded.hidden_size,
            loaded.hidden_size * loaded.hidden_size
        );
        snprintf(label, sizeof(label), "network_%d_bias", layer);
        ok = ok && read_float_array(
            file,
            label,
            loaded.network_bias + layer * loaded.hidden_size,
            loaded.hidden_size
        );
    }
    ok = ok &&
         read_float_array(file, "decoder_weight", loaded.decoder_weight, decoder_weight_count) &&
         read_float_array(file, "decoder_bias", loaded.decoder_bias, decoder_bias_count);

done:
    fclose(file);
    if (!ok) {
        flappy_policy_unload(&loaded);
        return false;
    }

    flappy_policy_unload(policy);
    *policy = loaded;
    return true;
}

static float gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.70710678118f));
}

static void apply_network_layer(
    const FlappyPolicy *policy,
    int layer,
    const float *input,
    float *output
) {
    const float *weight = policy->network_weight +
                          layer * policy->hidden_size * policy->hidden_size;
    const float *bias = policy->network_bias + layer * policy->hidden_size;
    for (int out_idx = 0; out_idx < policy->hidden_size; ++out_idx) {
        float value = bias[out_idx];
        for (int in_idx = 0; in_idx < policy->hidden_size; ++in_idx) {
            value += weight[out_idx * policy->hidden_size + in_idx] * input[in_idx];
        }
        output[out_idx] = gelu(value);
    }
}

FlappyAction flappy_policy_action(
    const FlappyPolicy *policy,
    const float observation[FLAPPY_OBSERVATION_SIZE],
    float logits[2]
) {
    float *hidden = (float *)calloc((size_t)policy->hidden_size, sizeof(float));
    float *next_hidden = (float *)calloc((size_t)policy->hidden_size, sizeof(float));
    if (hidden == NULL || next_hidden == NULL) {
        free(hidden);
        free(next_hidden);
        logits[0] = 0.0f;
        logits[1] = 0.0f;
        return FLAPPY_ACTION_NONE;
    }

    for (int hidden_idx = 0; hidden_idx < policy->hidden_size; ++hidden_idx) {
        float value = policy->encoder_bias[hidden_idx];
        for (int obs_idx = 0; obs_idx < FLAPPY_OBSERVATION_SIZE; ++obs_idx) {
            value += policy->encoder_weight[hidden_idx * FLAPPY_OBSERVATION_SIZE + obs_idx] *
                     observation[obs_idx];
        }
        hidden[hidden_idx] = policy->legacy_encoder_gelu ? gelu(value) : value;
    }

    for (int layer = 0; layer < policy->network_layers; ++layer) {
        apply_network_layer(policy, layer, hidden, next_hidden);
        float *swap = hidden;
        hidden = next_hidden;
        next_hidden = swap;
    }

    logits[0] = policy->decoder_bias[0];
    logits[1] = policy->decoder_bias[1];
    for (int hidden_idx = 0; hidden_idx < policy->hidden_size; ++hidden_idx) {
        logits[0] += policy->decoder_weight[hidden_idx] * hidden[hidden_idx];
        logits[1] += policy->decoder_weight[policy->hidden_size + hidden_idx] *
                     hidden[hidden_idx];
    }

    FlappyAction action = logits[1] > logits[0] ? FLAPPY_ACTION_FLAP : FLAPPY_ACTION_NONE;
    free(hidden);
    free(next_hidden);
    return action;
}
