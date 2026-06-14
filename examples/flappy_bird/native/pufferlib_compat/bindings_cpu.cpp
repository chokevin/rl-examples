#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define PUFFER_STRINGIFY_INNER(x) #x
#define PUFFER_STRINGIFY(x) PUFFER_STRINGIFY_INNER(x)

#include "vecenv.h"

namespace py = pybind11;

struct VecEnv {
    StaticVec *vec = nullptr;
    int total_agents = 0;
    int obs_size = 0;
    int num_atns = 0;
    std::vector<int> act_sizes;
    std::string obs_dtype;
    size_t obs_elem_size = 0;
};

static double get_config(py::dict kwargs, const char *key) {
    if (!kwargs.contains(key)) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    return kwargs[key].cast<double>();
}

static Dict *py_dict_to_c_dict(py::dict py_dict) {
    Dict *dict = create_dict(static_cast<int>(py_dict.size()));
    for (auto item : py_dict) {
        const char *key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(dict, key, item.second.cast<double>());
        } catch (const py::cast_error &) {
        }
    }
    return dict;
}

static void free_dict(Dict *dict) {
    free(dict->items);
    free(dict);
}

static std::unique_ptr<VecEnv> create_vec(py::dict args, int gpu = 0) {
    if (gpu != 0) {
        throw std::runtime_error("Flappy native CPU backend does not support gpu != 0");
    }

    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();
    int total_agents = static_cast<int>(get_config(vec_kwargs, "total_agents"));
    int num_buffers = static_cast<int>(get_config(vec_kwargs, "num_buffers"));
    if (total_agents < 1) {
        throw std::runtime_error("vec.total_agents must be positive");
    }
    if (num_buffers < 1 || total_agents % num_buffers != 0) {
        throw std::runtime_error("vec.total_agents must be divisible by vec.num_buffers");
    }

    Dict *vec_dict = py_dict_to_c_dict(vec_kwargs);
    Dict *env_dict = py_dict_to_c_dict(env_kwargs);

    auto result = std::make_unique<VecEnv>();
    {
        py::gil_scoped_release no_gil;
        result->vec = create_static_vec(total_agents, num_buffers, 0, vec_dict, env_dict);
    }
    free_dict(vec_dict);
    free_dict(env_dict);

    result->total_agents = total_agents;
    result->obs_size = get_obs_size();
    result->num_atns = get_num_atns();
    int *raw_act_sizes = get_act_sizes();
    result->act_sizes.assign(raw_act_sizes, raw_act_sizes + get_num_act_sizes());
    result->obs_dtype = get_obs_dtype();
    result->obs_elem_size = get_obs_elem_size();
    return result;
}

static void vec_reset(VecEnv &vec) {
    py::gil_scoped_release no_gil;
    static_vec_reset(vec.vec);
}

static void vec_cpu_step(VecEnv &vec, long long actions_ptr) {
    std::memcpy(
        vec.vec->actions,
        reinterpret_cast<void *>(actions_ptr),
        static_cast<size_t>(vec.total_agents * vec.num_atns) * sizeof(float)
    );
    py::gil_scoped_release no_gil;
    cpu_vec_step(vec.vec);
}

static py::dict vec_log(VecEnv &vec) {
    Dict *out = create_dict(32);
    static_vec_log(vec.vec, out);
    py::dict result;
    for (int i = 0; i < out->size; ++i) {
        result[out->items[i].key] = out->items[i].value;
    }
    free_dict(out);
    return result;
}

static void vec_close(VecEnv &vec) {
    if (vec.vec != nullptr) {
        static_vec_close(vec.vec);
        vec.vec = nullptr;
    }
}

static void puff_advantage_cpu(
    long long values_ptr,
    long long rewards_ptr,
    long long dones_ptr,
    long long importance_ptr,
    long long advantages_ptr,
    int num_steps,
    int horizon,
    float gamma,
    float lambda,
    float rho_clip,
    float c_clip
) {
    const float *values = reinterpret_cast<const float *>(values_ptr);
    const float *rewards = reinterpret_cast<const float *>(rewards_ptr);
    const float *dones = reinterpret_cast<const float *>(dones_ptr);
    const float *importance = reinterpret_cast<const float *>(importance_ptr);
    float *advantages = reinterpret_cast<float *>(advantages_ptr);
    for (int row = 0; row < num_steps; ++row) {
        int offset = row * horizon;
        float last_puffer_lam = 0.0f;
        for (int t = horizon - 2; t >= 0; --t) {
            int next_t = t + 1;
            float next_nonterminal = 1.0f - dones[offset + next_t];
            float imp = importance[offset + t];
            float rho_t = imp < rho_clip ? imp : rho_clip;
            float c_t = imp < c_clip ? imp : c_clip;
            float delta = rho_t * rewards[offset + next_t] +
                gamma * values[offset + next_t] * next_nonterminal -
                values[offset + t];
            last_puffer_lam = delta +
                gamma * lambda * c_t * last_puffer_lam * next_nonterminal;
            advantages[offset + t] = last_puffer_lam;
        }
    }
}

PYBIND11_MODULE(_C, m) {
    m.attr("precision_bytes") = 4;
    m.attr("env_name") = PUFFER_STRINGIFY(ENV_NAME);
    m.attr("gpu") = 0;

    m.def("puff_advantage_cpu", &puff_advantage_cpu);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 0);

    py::class_<VecEnv, std::unique_ptr<VecEnv>>(m, "VecEnv")
        .def_readonly("total_agents", &VecEnv::total_agents)
        .def_readonly("obs_size", &VecEnv::obs_size)
        .def_readonly("num_atns", &VecEnv::num_atns)
        .def_readonly("act_sizes", &VecEnv::act_sizes)
        .def_readonly("obs_dtype", &VecEnv::obs_dtype)
        .def_readonly("obs_elem_size", &VecEnv::obs_elem_size)
        .def_property_readonly("gpu", [](VecEnv &) { return 0; })
        .def_property_readonly("obs_ptr", [](VecEnv &vec) {
            return reinterpret_cast<long long>(vec.vec->observations);
        })
        .def_property_readonly("rewards_ptr", [](VecEnv &vec) {
            return reinterpret_cast<long long>(vec.vec->rewards);
        })
        .def_property_readonly("terminals_ptr", [](VecEnv &vec) {
            return reinterpret_cast<long long>(vec.vec->terminals);
        })
        .def("reset", &vec_reset)
        .def("cpu_step", &vec_cpu_step)
        .def("render", [](VecEnv &vec, int env_id) {
            static_vec_render(vec.vec, env_id);
        })
        .def("log", &vec_log)
        .def("close", &vec_close);
}
