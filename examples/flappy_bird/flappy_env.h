#ifndef FLAPPY_ENV_H
#define FLAPPY_ENV_H

#include <stdbool.h>
#include <stdint.h>

#define FLAPPY_MAX_PIPES 4
#define FLAPPY_OBSERVATION_SIZE 4

typedef enum FlappyAction {
    FLAPPY_ACTION_NONE = 0,
    FLAPPY_ACTION_FLAP = 1,
} FlappyAction;

typedef struct FlappyEnvConfig {
    int screen_width;
    int screen_height;
    float gravity;
    float flap_velocity;
    float bird_x;
    float bird_radius;
    float pipe_width;
    float pipe_gap;
    float pipe_speed;
    float pipe_spacing;
    int pipe_count;
    float alive_reward;
    float pass_reward;
    float crash_reward;
} FlappyEnvConfig;

typedef struct FlappyPipe {
    float x;
    float gap_y;
    bool scored;
} FlappyPipe;

typedef struct FlappyEnv {
    FlappyEnvConfig config;
    float bird_y;
    float bird_velocity;
    FlappyPipe pipes[FLAPPY_MAX_PIPES];
    uint32_t rng_state;
    int score;
    int frame;
    bool done;
} FlappyEnv;

typedef struct FlappyStepResult {
    float observation[FLAPPY_OBSERVATION_SIZE];
    float reward;
    bool terminated;
    int score;
    int frame;
} FlappyStepResult;

FlappyEnvConfig flappy_env_default_config(void);
void flappy_env_reset(FlappyEnv *env, FlappyEnvConfig config, uint32_t seed);
FlappyStepResult flappy_env_step(FlappyEnv *env, FlappyAction action);
void flappy_env_get_observation(const FlappyEnv *env, float out[FLAPPY_OBSERVATION_SIZE]);

#endif
