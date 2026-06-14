#include <assert.h>
#include <stdio.h>

#include "../examples/flappy_bird/flappy_policy.h"

static void write_test_policy(const char *path) {
    FILE *file = fopen(path, "w");
    assert(file != NULL);
    fprintf(file, "flappy_policy_v1\n");
    fprintf(file, "hidden_size 2\n");
    fprintf(file, "pipe_gap 123\n");
    fprintf(file, "encoder_weight 0 0 0 0 0 0 0 0\n");
    fprintf(file, "encoder_bias 1 0\n");
    fprintf(file, "decoder_weight 0 0 2 0\n");
    fprintf(file, "decoder_bias 0 0\n");
    fclose(file);
}

static void write_test_policy_v2(const char *path) {
    FILE *file = fopen(path, "w");
    assert(file != NULL);
    fprintf(file, "flappy_policy_v2\n");
    fprintf(file, "hidden_size 2\n");
    fprintf(file, "network_layers 1\n");
    fprintf(file, "pipe_gap 220\n");
    fprintf(file, "encoder_weight 1 0 0 0 0 1 0 0\n");
    fprintf(file, "encoder_bias 0 0\n");
    fprintf(file, "network_0_weight 0 0 0 0\n");
    fprintf(file, "network_0_bias 0 1\n");
    fprintf(file, "decoder_weight 0 0 0 2\n");
    fprintf(file, "decoder_bias 0 0\n");
    fclose(file);
}

int main(void) {
    const char *path = "build/test_policy.txt";
    write_test_policy(path);

    FlappyPolicy policy;
    flappy_policy_init(&policy);
    assert(flappy_policy_load(&policy, path));
    assert(policy.hidden_size == 2);
    assert(policy.has_pipe_gap);
    assert(policy.pipe_gap == 123.0f);

    float observation[FLAPPY_OBSERVATION_SIZE] = {0};
    float logits[2] = {0};
    FlappyAction action = flappy_policy_action(&policy, observation, logits);

    assert(action == FLAPPY_ACTION_FLAP);
    assert(logits[1] > logits[0]);
    flappy_policy_unload(&policy);

    const char *v2_path = "build/test_policy_v2.txt";
    write_test_policy_v2(v2_path);
    flappy_policy_init(&policy);
    assert(flappy_policy_load(&policy, v2_path));
    assert(policy.hidden_size == 2);
    assert(policy.network_layers == 1);

    action = flappy_policy_action(&policy, observation, logits);

    assert(action == FLAPPY_ACTION_FLAP);
    assert(logits[1] > logits[0]);
    flappy_policy_unload(&policy);
    return 0;
}
