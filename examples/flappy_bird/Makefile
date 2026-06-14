CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2
CFLAGS += -I.
MATH_LIBS ?= -lm
BUILD_DIR ?= build
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_EXT ?= .dylib
SHARED_FLAGS ?= -dynamiclib
else
SHARED_EXT ?= .so
SHARED_FLAGS ?= -shared
endif
LIB_TARGET := $(BUILD_DIR)/libflappy_env$(SHARED_EXT)

RAYLIB_CFLAGS ?= $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS ?= $(shell pkg-config --libs raylib 2>/dev/null)

ifeq ($(strip $(RAYLIB_LIBS)),)
ifeq ($(UNAME_S),Darwin)
RAYLIB_LIBS = -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
else
RAYLIB_LIBS = -lraylib -lm -lpthread -ldl -lrt -lX11
endif
endif

.PHONY: all shared native train-native run run-policy test clean

all: flappy_bird

flappy_bird: main.c flappy_env.c flappy_env.h flappy_policy.c flappy_policy.h
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) main.c flappy_env.c flappy_policy.c -o $@ $(RAYLIB_LIBS) $(MATH_LIBS)

shared: $(LIB_TARGET)

native:
	uv run python ../build_flappy_native.py

train-native:
	uv run python ../train_flappy_native.py

$(LIB_TARGET): flappy_env.c flappy_env.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC $(SHARED_FLAGS) flappy_env.c -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_flappy_env: flappy_env.c flappy_env.h ../../tests/test_flappy_env.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) flappy_env.c ../../tests/test_flappy_env.c -o $@

$(BUILD_DIR)/test_flappy_policy: flappy_policy.c flappy_policy.h flappy_env.h ../../tests/test_flappy_policy.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) flappy_policy.c ../../tests/test_flappy_policy.c -o $@ $(MATH_LIBS)

test: $(BUILD_DIR)/test_flappy_env $(BUILD_DIR)/test_flappy_policy
	./$(BUILD_DIR)/test_flappy_env
	./$(BUILD_DIR)/test_flappy_policy

run: flappy_bird
	./flappy_bird

run-policy: flappy_bird
	./flappy_bird --policy $(or $(POLICY),../../experiments/flappy_bird/policy.txt)

clean:
	rm -rf $(BUILD_DIR) flappy_bird
