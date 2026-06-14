#include "flappy_env.h"

#include <float.h>
#include <stdint.h>

FlappyEnvConfig flappy_env_default_config(void) {
    FlappyEnvConfig config = {
        .screen_width = 800,
        .screen_height = 450,
        .gravity = 0.35f,
        .flap_velocity = -6.5f,
        .bird_x = 160.0f,
        .bird_radius = 14.0f,
        .pipe_width = 70.0f,
        .pipe_gap = 130.0f,
        .pipe_speed = 2.6f,
        .pipe_spacing = 260.0f,
        .pipe_count = 3,
        .alive_reward = 0.01f,
        .pass_reward = 1.0f,
        .crash_reward = -1.0f,
    };
    return config;
}

static uint32_t next_random(FlappyEnv *env) {
    uint32_t x = env->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng_state = x == 0 ? 1u : x;
    return env->rng_state;
}

static float random_gap_y(FlappyEnv *env) {
    const FlappyEnvConfig *config = &env->config;
    float margin = config->pipe_gap * 0.5f + config->bird_radius + 12.0f;
    float min_y = margin;
    float max_y = (float)config->screen_height - margin;
    float unit = (float)next_random(env) / (float)UINT32_MAX;
    return min_y + (max_y - min_y) * unit;
}

void flappy_env_reset(FlappyEnv *env, FlappyEnvConfig config, uint32_t seed) {
    env->config = config;
    env->bird_y = (float)config.screen_height * 0.5f;
    env->bird_velocity = 0.0f;
    env->rng_state = seed == 0 ? 1u : seed;
    env->score = 0;
    env->frame = 0;
    env->done = false;

    for (int i = 0; i < config.pipe_count; ++i) {
        env->pipes[i].x = (float)config.screen_width + 180.0f +
                          (float)i * config.pipe_spacing;
        env->pipes[i].gap_y = random_gap_y(env);
        env->pipes[i].scored = false;
    }
}

static float rightmost_pipe_x(const FlappyEnv *env) {
    float rightmost = -FLT_MAX;
    for (int i = 0; i < env->config.pipe_count; ++i) {
        if (env->pipes[i].x > rightmost) {
            rightmost = env->pipes[i].x;
        }
    }
    return rightmost;
}

static const FlappyPipe *next_pipe(const FlappyEnv *env) {
    const FlappyPipe *next = &env->pipes[0];
    float best_x = FLT_MAX;
    float bird_left = env->config.bird_x - env->config.bird_radius;

    for (int i = 0; i < env->config.pipe_count; ++i) {
        const FlappyPipe *pipe = &env->pipes[i];
        float pipe_right = pipe->x + env->config.pipe_width;
        if (pipe_right >= bird_left && pipe->x < best_x) {
            next = pipe;
            best_x = pipe->x;
        }
    }

    return next;
}

void flappy_env_get_observation(const FlappyEnv *env, float out[FLAPPY_OBSERVATION_SIZE]) {
    const FlappyPipe *pipe = next_pipe(env);
    out[0] = env->bird_y / (float)env->config.screen_height;
    out[1] = env->bird_velocity / (float)env->config.screen_height;
    out[2] = (pipe->x + env->config.pipe_width - env->config.bird_x) /
             (float)env->config.screen_width;
    out[3] = (pipe->gap_y - env->bird_y) / (float)env->config.screen_height;
}

static bool hit_bounds(const FlappyEnv *env) {
    return env->bird_y - env->config.bird_radius <= 0.0f ||
           env->bird_y + env->config.bird_radius >= (float)env->config.screen_height;
}

static bool hit_pipe(const FlappyEnv *env, const FlappyPipe *pipe) {
    float bird_left = env->config.bird_x - env->config.bird_radius;
    float bird_right = env->config.bird_x + env->config.bird_radius;
    float bird_top = env->bird_y - env->config.bird_radius;
    float bird_bottom = env->bird_y + env->config.bird_radius;
    float pipe_right = pipe->x + env->config.pipe_width;
    float gap_top = pipe->gap_y - env->config.pipe_gap * 0.5f;
    float gap_bottom = pipe->gap_y + env->config.pipe_gap * 0.5f;

    bool overlaps_x = bird_right > pipe->x && bird_left < pipe_right;
    bool inside_gap = bird_top > gap_top && bird_bottom < gap_bottom;
    return overlaps_x && !inside_gap;
}

FlappyStepResult flappy_env_step(FlappyEnv *env, FlappyAction action) {
    FlappyStepResult result = {
        .reward = 0.0f,
        .terminated = env->done,
        .score = env->score,
        .frame = env->frame,
    };

    if (env->done) {
        flappy_env_get_observation(env, result.observation);
        return result;
    }

    if (action == FLAPPY_ACTION_FLAP) {
        env->bird_velocity = env->config.flap_velocity;
    }

    env->bird_velocity += env->config.gravity;
    env->bird_y += env->bird_velocity;
    env->frame += 1;
    result.reward = env->config.alive_reward;

    for (int i = 0; i < env->config.pipe_count; ++i) {
        FlappyPipe *pipe = &env->pipes[i];
        pipe->x -= env->config.pipe_speed;

        if (!pipe->scored &&
            pipe->x + env->config.pipe_width < env->config.bird_x - env->config.bird_radius) {
            pipe->scored = true;
            env->score += 1;
            result.reward += env->config.pass_reward;
        }
    }

    for (int i = 0; i < env->config.pipe_count; ++i) {
        FlappyPipe *pipe = &env->pipes[i];
        if (pipe->x + env->config.pipe_width < 0.0f) {
            float next_x = rightmost_pipe_x(env) + env->config.pipe_spacing;
            float min_spawn_x = (float)env->config.screen_width + 180.0f;
            pipe->x = next_x > min_spawn_x ? next_x : min_spawn_x;
            pipe->gap_y = random_gap_y(env);
            pipe->scored = false;
        }
    }

    env->done = hit_bounds(env);
    for (int i = 0; !env->done && i < env->config.pipe_count; ++i) {
        env->done = hit_pipe(env, &env->pipes[i]);
    }

    if (env->done) {
        result.reward = env->config.crash_reward;
    }

    result.terminated = env->done;
    result.score = env->score;
    result.frame = env->frame;
    flappy_env_get_observation(env, result.observation);
    return result;
}
