#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto instance_norm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias,
                            Tensor? running_mean, Tensor? running_var,
                            bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);
      auto use_input_stats = args[5].unwrapToBool();
      auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

      torch::Tensor gamma, beta;

      auto eps = args[7].unwrapToDouble(1e-5f);

      LOG_DEBUG("momentum disregarded");
      LOG_DEBUG("cudnn disregarded");

      auto should_unpack = util::toVec(orig_shape).size() < 4;
      if (should_unpack) {
        // expand spatial dims from 1D to 2D
        auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
        LOG_DEBUG(
            "Input shape is less than 4D got: "
            << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto in_shuffle = ctx->net->addShuffle(*input);
        in_shuffle->setReshapeDimensions(new_shape);
        in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
        input = in_shuffle->getOutput(0);
      }

      long n_channels = util::toVec(input->getDimensions())[1];
      if (ctx->input_is_dynamic) {
        gamma = args[1].unwrapToTensor();
        beta = args[2].unwrapToTensor();
      } else {
        gamma = args[1].unwrapToTensor(at::full({n_channels}, 1, {options}));
        beta = args[2].unwrapToTensor(at::full({n_channels}, 0, {options}));
      }

      nvinfer1::ITensor* out_tensor;
      if (use_input_stats) {

        auto zeros_weights = Weights(ctx, at::full({1}, 0, {options}));
        auto half_weights = Weights(ctx, at::full({1}, 0.5, {options}));
        auto ones_weights = Weights(ctx, at::full({1}, 1, {options}));
        auto ones_weights_channels = Weights(ctx, at::full({n_channels}, 1, {options}));
        auto twos_weights = Weights(ctx, at::full({1}, 2, {options}));
        auto eps_weights = Weights(ctx, at::full({1}, eps, {options}));

        const bool keep_dims = true;
        // Compute mean over last two axes (height and width)
        uint32_t reduces_axes = (1 << 2) | (1 << 3);
        auto mean_trt = ctx->net->addReduce(*input, nvinfer1::ReduceOperation::kAVG, reduces_axes, keep_dims)->getOutput(0);

        // Compute delta with mean
        auto delta_trt = ctx->net->addElementWise(*input, *mean_trt, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);

        // Compute delta ^ 2
        auto var_trt = ctx->net->addScale(*delta_trt, nvinfer1::ScaleMode::kUNIFORM, zeros_weights.data, ones_weights.data, twos_weights.data)->getOutput(0);
        // Compute average over delta ^ 2
        var_trt = ctx->net->addReduce(*var_trt, nvinfer1::ReduceOperation::kAVG, reduces_axes, keep_dims)->getOutput(0);
        // Compute variance - sqrt(avg(delta ^ 2))
        var_trt = ctx->net->addScale(*var_trt, nvinfer1::ScaleMode::kUNIFORM, eps_weights.data, ones_weights.data, half_weights.data)->getOutput(0);
        LOG_DEBUG("Delta tensor shape: " << delta_trt->getDimensions());
        LOG_DEBUG("Variance tensor shape: " << var_trt->getDimensions());
        out_tensor = ctx->net->addElementWise(*delta_trt, *var_trt, nvinfer1::ElementWiseOperation::kDIV)->getOutput(0);

        auto gamma_weights = Weights(ctx, gamma);
        auto beta_weights = Weights(ctx, beta);

        out_tensor = ctx->net->addScale(*var_trt, nvinfer1::ScaleMode::kCHANNEL, beta_weights.data, gamma_weights.data, ones_weights_channels.data)->getOutput(0);
      } else {
        torch::Tensor mean, var;
        if (ctx->input_is_dynamic) {
          mean = args[3].unwrapToTensor();
          var = args[4].unwrapToTensor();
        } else {
          mean = args[3].unwrapToTensor(at::full({n_channels}, 0, {options}));
          var = args[4].unwrapToTensor(at::full({n_channels}, 1, {options}));
        }

        auto scale = gamma / torch::sqrt(var + eps);
        auto bias = beta - mean * scale;

        auto scale_weights = Weights(ctx, scale);
        auto bias_weights = Weights(ctx, bias);

        out_tensor =
            ctx->net->addScaleNd(*input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, {}, 1)->getOutput(0);
      }
      
      if (should_unpack) {
        LOG_DEBUG("Inserting shuffle layer to reshape to back to original shape: " << orig_shape);
        auto out_shuffle = ctx->net->addShuffle(*out_tensor);
        out_shuffle->setReshapeDimensions(orig_shape);
        out_shuffle->setName(std::string("[Reshape output to " + util::toStr(orig_shape) + ']').c_str());
        out_tensor = out_shuffle->getOutput(0);
      }

      ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
