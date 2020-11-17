package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "plugins",
    hdrs = [
        "interpolate_plugin.h",
        "nonzero_plugin.h"
    ],
    srcs = [
        "interpolate_plugin.cpp",
        "nonzero_plugin.cpp",
        "nonzero_kernel_cu"
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/conversion/conversionctx",
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/converters/impl/plugins",
    srcs = ["interpolate_plugin.h"],
)

genrule(
    name = "nonzero_kernel_cu",
    srcs = ["nonzero_kernel.cu"],
    outs = ["nonzero_kernel.o"],
    cmd = "$(location @local_config_cuda//:nvcc) --compile $(SRCS) --output-file $(OUTS) --extended-lambda",
    tools = ["@local_config_cuda//:nvcc"],
)