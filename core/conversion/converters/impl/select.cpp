#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"
#include "plugins/nonzero_plugin.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto select_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::squeeze.dim(Tensor(a) self, int dim) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    // Wrap negative indices
                    auto dim = (args[1].unwrapToInt() + in->getDimensions().nbDims) % in->getDimensions().nbDims;
                    LOG_DEBUG("dim " << dim);

                    auto shuffle_layer = ctx->net->addShuffle(*in);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::squeezeDims(in->getDimensions(), dim));

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_layer->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::Int.Tensor(Tensor a) -> (int)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::nonzero(Tensor self) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();

                    auto creator = new plugins::NonZeroPluginCreator();
                    auto plugin = creator->createPlugin("nonzero");

                    auto nonzero_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
                    TRTORCH_CHECK(nonzero_layer, "Unable to create nonzero plugin from node" << *n);

                    nonzero_layer->setName(util::node_info(n).c_str());

                    auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], nonzero_layer->getOutput(0));
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto dim = args[1].unwrapToInt();
                    auto start = args[2].unwrapToInt();
                    auto end = args[3].unwrapToInt();
                    
                    auto dims = in->getDimensions();

                    end = std::min((long)dims.d[dim], end);
                    dims.d[dim] = end - start;

                    nvinfer1::Dims start_idx;
                    start_idx.nbDims = dims.nbDims;
                    nvinfer1::Dims sizes;
                    sizes.nbDims = dims.nbDims;
                    nvinfer1::Dims strides;
                    strides.nbDims = dims.nbDims;
                    for (int i = 0; i < dims.nbDims; i++) {
                      if (i == dim) {
                        start_idx.d[i] = start;
                        sizes.d[i] = end - start;
                      } else {
                        start_idx.d[i] = 0;
                        sizes.d[i] = dims.d[i];
                      }
                      strides.d[i] = 0;
                    }

                    auto slice_layer = ctx->net->addSlice(*in,start_idx, sizes, strides);
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_layer->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto pad_dims = util::toDims(args[1].unwrapToIntList());
                    auto left = pad_dims.d[0];
                    auto right = pad_dims.d[1];
                    auto top = pad_dims.d[2];
                    auto bottom = pad_dims.d[3];

                    auto dims = in->getDimensions();

                    nvinfer1::Dims2 pre_pad = {top, left};
                    nvinfer1::Dims2 post_pad = {bottom, right};

                    auto padding_layer = ctx->net->addPaddingNd(*in, pre_pad, post_pad);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], padding_layer->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    LOG_DEBUG("begin select");
                    auto in = args[0].ITensor();
                    auto axis = args[1].unwrapToInt();
                    auto ind = (int32_t)args[2].unwrapToInt();

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::tensor({ind}).to(torch::kI32);
                    auto weights = Weights(ctx, indices);

                    // IConstantLayer to convert indices from Weights to ITensor
                    auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                    TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
                    shuffle_layer->setName(util::node_info(n).c_str());
                    auto shuffle_out = shuffle_layer->getOutput(0);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto axis = args[1].unwrapToInt();
                    auto start = (int32_t)args[2].unwrapToInt();
                    auto length = (int32_t)args[3].unwrapToInt();

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::arange(start, start + length, 1).to(torch::kI32);
                    auto weights = Weights(ctx, indices);

                    // IConstantLayer to convert indices from Weights to ITensor
                    auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                    TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
                    shuffle_layer->setName(util::node_info(n).c_str());
                    auto shuffle_out = shuffle_layer->getOutput(0);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto axis = args[1].unwrapToInt();
                    torch::Tensor start = args[2].IValue()->toTensor().to(torch::kI32);
                    int32_t startIdx = start.item().to<int32_t>();
                    auto length = (int32_t)args[3].unwrapToInt();

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::arange(startIdx, startIdx + length, 1).to(torch::kI32);
                    auto weights = Weights(ctx, indices);

                    // IConstantLayer to convert indices from Weights to ITensor
                    auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                    TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
                    shuffle_layer->setName(util::node_info(n).c_str());
                    auto shuffle_out = shuffle_layer->getOutput(0);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern(
            {"aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto embeddingTensor = args[0].ITensorOrFreeze(ctx);
               auto indicesTensor = args[1].ITensor();
               // Set datatype for indices tensor to INT32
               indicesTensor->setType(nvinfer1::DataType::kINT32);

               // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
               auto gather_layer = ctx->net->addGather(*embeddingTensor, *indicesTensor, 0);
               TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto axis = args[1].unwrapToInt();
               auto maxDim = static_cast<int64_t>(in->getDimensions().d[axis]);
               // Handle case when given tensor index is negative
               auto startIdx = args[2].unwrapToInt();
               auto start = (startIdx < 0) ? (maxDim + startIdx) : startIdx;
               // Bound the end index to input tensor dimensions at specified axis
               auto endIdx = std::min(args[3].unwrapToInt(), maxDim);
               auto end = (endIdx < 0) ? (maxDim + endIdx) : endIdx;
               auto step = args[4].unwrapToInt();

               // indices to be accessed need to be an at::Tensor
               at::Tensor indices = torch::arange(start, end, step).to(torch::kI32);
               auto weights = Weights(ctx, indices);

               // IConstantLayer to convert indices from Weights to ITensor
               auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
               TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
               auto const_out = const_layer->getOutput(0);

               // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
               auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
               TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Slice layer output shape: " << out->getDimensions());

               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch