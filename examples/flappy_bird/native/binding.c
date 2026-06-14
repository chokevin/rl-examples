#include "../flappy_env.h"

#include <math.h>
#include <stdint.h>

#define OBS_SIZE FLAPPY_OBSERVATION_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    FlappyEnv game;
    FlappyEnvConfig config;
    uint32_t seed;
    int max_steps;
    int episode_length;
    float episode_return;
    int reward_shaping;
    float centering_reward;
} Env;

void c_reset(Env *env);
void c_step(Env *env);
void c_render(Env *env);
void c_close(Env *env);

#include "vecenv.h"

static float required(Dict *kwargs, const char *key) {
    return (float)dict_get(kwargs, key)->value;
}

void my_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->config.screen_width = (int)required(kwargs, "screen_width");
    env->config.screen_height = (int)required(kwargs, "screen_height");
    env->config.gravity = required(kwargs, "gravity");
    env->config.flap_velocity = required(kwargs, "flap_velocity");
    env->config.bird_x = required(kwargs, "bird_x");
    env->config.bird_radius = required(kwargs, "bird_radius");
    env->config.pipe_width = required(kwargs, "pipe_width");
    env->config.pipe_gap = required(kwargs, "pipe_gap");
    env->config.pipe_speed = required(kwargs, "pipe_speed");
    env->config.pipe_spacing = required(kwargs, "pipe_spacing");
    env->config.pipe_count = (int)required(kwargs, "pipe_count");
    env->config.alive_reward = required(kwargs, "alive_reward");
    env->config.pass_reward = required(kwargs, "pass_reward");
    env->config.crash_reward = required(kwargs, "crash_reward");
    env->max_steps = (int)required(kwargs, "max_steps");
    env->reward_shaping = required(kwargs, "reward_shaping") != 0.0f;
    env->centering_reward = required(kwargs, "centering_reward");
    env->seed = (uint32_t)required(kwargs, "seed") + env->rng * 9973u;
}

void my_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

void c_reset(Env *env) {
    env->episode_length = 0;
    env->episode_return = 0.0f;
    env->seed = env->seed == UINT32_MAX ? 1u : env->seed + 1u;
    flappy_env_reset(&env->game, env->config, env->seed);
    flappy_env_get_observation(&env->game, env->observations);
}

void c_step(Env *env) {
    FlappyAction action = env->actions[0] > 0.5f
        ? FLAPPY_ACTION_FLAP
        : FLAPPY_ACTION_NONE;
    FlappyStepResult result = flappy_env_step(&env->game, action);
    float reward = result.reward;
    if (env->reward_shaping && !result.terminated) {
        float centered = fmaxf(0.0f, 1.0f - fabsf(result.observation[3]) / 0.35f);
        reward += env->centering_reward * centered;
    }

    env->episode_length += 1;
    env->episode_return += reward;
    int truncated = env->episode_length >= env->max_steps && !result.terminated;
    int done = result.terminated || truncated;

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;

    if (done) {
        env->log.score += (float)result.score;
        env->log.perf += (float)result.score;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)env->episode_length;
        env->log.n += 1.0f;
        c_reset(env);
        env->rewards[0] = reward;
        env->terminals[0] = 1.0f;
        return;
    }

    for (int i = 0; i < FLAPPY_OBSERVATION_SIZE; ++i) {
        env->observations[i] = result.observation[i];
    }
}

void c_render(Env *env) {
    (void)env;
}

void c_close(Env *env) {
    (void)env;
}
