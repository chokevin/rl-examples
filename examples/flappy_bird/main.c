#include "flappy_env.h"
#include "flappy_policy.h"

#include <stdio.h>
#include <string.h>

#include "raylib.h"

static void draw_pipe_pair(const FlappyEnv *env, const FlappyPipe *pipe) {
    const FlappyEnvConfig *config = &env->config;
    int x = (int)pipe->x;
    int width = (int)config->pipe_width;
    int gap_top = (int)(pipe->gap_y - config->pipe_gap * 0.5f);
    int gap_bottom = (int)(pipe->gap_y + config->pipe_gap * 0.5f);

    DrawRectangle(x, 0, width, gap_top, GREEN);
    DrawRectangle(x, gap_bottom, width, config->screen_height - gap_bottom, GREEN);
}

static void draw_env(const FlappyEnv *env, bool policy_loaded, FlappyAction action, float logits[2]) {
    ClearBackground((Color){135, 206, 235, 255});

    for (int i = 0; i < env->config.pipe_count; ++i) {
        draw_pipe_pair(env, &env->pipes[i]);
    }

    DrawCircle((int)env->config.bird_x, (int)env->bird_y, env->config.bird_radius, YELLOW);
    DrawCircleLines((int)env->config.bird_x, (int)env->bird_y, env->config.bird_radius, ORANGE);

    DrawText(TextFormat("score: %d", env->score), 20, 18, 24, DARKBLUE);
    if (policy_loaded) {
        DrawText(
            TextFormat(
                "policy: %s  logits=[%.2f %.2f]  r: reset",
                action == FLAPPY_ACTION_FLAP ? "flap" : "no-op",
                logits[0],
                logits[1]
            ),
            20,
            46,
            18,
            DARKBLUE
        );
    } else {
        DrawText("space/click: flap  r: reset", 20, 46, 18, DARKBLUE);
    }

    if (env->done) {
        DrawRectangle(0, 0, env->config.screen_width, env->config.screen_height,
                      (Color){0, 0, 0, 110});
        DrawText("crashed", env->config.screen_width / 2 - 58,
                 env->config.screen_height / 2 - 35, 32, RAYWHITE);
        DrawText("press r to reset", env->config.screen_width / 2 - 78,
                 env->config.screen_height / 2 + 6, 20, RAYWHITE);
    }
}

static bool parse_args(int argc, char **argv, const char **policy_path) {
    *policy_path = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            *policy_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--policy path/to/policy.txt]\n", argv[0]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    FlappyEnv env;
    FlappyEnvConfig config = flappy_env_default_config();
    const char *policy_path = NULL;
    FlappyPolicy policy;
    flappy_policy_init(&policy);

    if (!parse_args(argc, argv, &policy_path)) {
        return 1;
    }
    bool policy_loaded = policy_path != NULL;
    if (policy_loaded && !flappy_policy_load(&policy, policy_path)) {
        fprintf(stderr, "failed to load policy: %s\n", policy_path);
        return 1;
    }
    if (policy_loaded && policy.has_pipe_gap) {
        config.pipe_gap = policy.pipe_gap;
    }

    flappy_env_reset(&env, config, 1u);

    InitWindow(config.screen_width, config.screen_height, "Basic Flappy Bird Env");
    // Human-playable raylib loop only. Training runs headless and is not capped.
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            flappy_env_reset(&env, config, 1u);
        }

        FlappyAction action = FLAPPY_ACTION_NONE;
        float logits[2] = {0};
        if (policy_loaded) {
            float observation[FLAPPY_OBSERVATION_SIZE];
            flappy_env_get_observation(&env, observation);
            action = flappy_policy_action(&policy, observation, logits);
        } else if (IsKeyPressed(KEY_SPACE) || IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            action = FLAPPY_ACTION_FLAP;
        }

        if (!env.done) {
            flappy_env_step(&env, action);
        }

        BeginDrawing();
        draw_env(&env, policy_loaded, action, logits);
        EndDrawing();
    }

    flappy_policy_unload(&policy);
    CloseWindow();
    return 0;
}
