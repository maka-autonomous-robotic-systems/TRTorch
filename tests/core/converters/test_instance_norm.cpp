#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenBatchNormConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5:1),
            %2: Float(5:1),
            %3: Float(5:1),
            %4: Float(5:1)):
        %5 : bool = prim::Constant[value=0]()
        %6 : bool = prim::Constant[value=1]()
        %7 : float = prim::Constant[value=1.0000000000000001e-05]()
        %8 : float = prim::Constant[value=0.10000000000000001]()
        %9 : Tensor = aten::instance_norm(%0, %1, %2, %3, %4, %6, %7, %8, %5)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
  auto weight = at::randint(1, 10, {5}, {at::kCUDA});
  auto bias = at::randint(1, 10, {5}, {at::kCUDA});
  auto running_mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto running_var = at::randint(1, 10, {5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, running_mean, running_var});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, running_mean, running_var});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
