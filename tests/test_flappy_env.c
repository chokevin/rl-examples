#include <assert.h>
#include <math.h>

#include "../examples/flappy_bird/flappy_env.h"

static void assert_close(float actual, float expected) {
    assert(fabsf(actual - expected) < 0.0001f);
}

static void test_reset_observation(void) {
    FlappyEnv env;
    FlappyEnvConfig config = flappy_env_default_config();
    float observation[FLAPPY_OBSERVATION_SIZE];

    flappy_env_reset(&env, config, 123u);
    flappy_env_get_observation(&env, observation);

    assert_close(observation[0], 0.5f);
    assert_close(observation[1], 0.0f);
    assert(observation[2] > 1.0f);
    assert(observation[3] > -0.5f);
    assert(observation[3] < 0.5f);
    assert(env.score == 0);
    assert(!env.done);
}

static void test_flap_moves_bird_up(void) {
    FlappyEnv env;
    FlappyEnvConfig config = flappy_env_default_config();

    flappy_env_reset(&env, config, 123u);
    float start_y = env.bird_y;
    FlappyStepResult result = flappy_env_step(&env, FLAPPY_ACTION_FLAP);

    assert(env.bird_y < start_y);
    assert(result.reward > 0.0f);
    assert(!result.terminated);
    assert(result.frame == 1);
}

static void test_scores_when_pipe_is_passed(void) {
    FlappyEnv env;
    FlappyEnvConfig config = flappy_env_default_config();

    flappy_env_reset(&env, config, 123u);
    env.pipes[0].x = env.config.bird_x - env.config.bird_radius -
                     env.config.pipe_width - 0.1f;
    env.pipes[0].gap_y = env.bird_y;
    env.pipes[0].scored = false;

    FlappyStepResult result = flappy_env_step(&env, FLAPPY_ACTION_NONE);

    assert(env.score == 1);
    assert(result.score == 1);
    assert(result.reward > env.config.pass_reward);
    assert(!result.terminated);
}

static void test_crashes_on_bounds(void) {
    FlappyEnv env;
    FlappyEnvConfig config = flappy_env_default_config();

    flappy_env_reset(&env, config, 123u);
    env.bird_y = (float)env.config.screen_height - env.config.bird_radius + 1.0f;

    FlappyStepResult result = flappy_env_step(&env, FLAPPY_ACTION_NONE);

    assert(result.terminated);
    assert(result.reward == env.config.crash_reward);
    assert(env.done);
}

int main(void) {
    test_reset_observation();
    test_flap_moves_bird_up();
    test_scores_when_pipe_is_passed();
    test_crashes_on_bounds();
    return 0;
}
