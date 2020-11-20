#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "plugins/set_max_plugin.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto set_max_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {R"SIG(custom_ops::set_max(Tensor input, int[] output_size, Scalar min_threshold) -> (Tensor))SIG",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto input = args[0].ITensor(); // assumes non-static input Tensor
       auto out_size = util::toDims(args[1].unwrapToIntList());
       auto minimum_threshold = args[2].unwrapToDouble();

       auto creator = new plugins::SetMaxPluginCreator();
       auto plugin = creator->createPlugin("set_max");
       LOG_DEBUG("out dims: " << out_size);
       plugin->setOutDimensions(nvinfer1::Dims2(out_size.d[out_size.nbDims - 2], out_size.d[out_size.nbDims - 1]));
       plugin->setMinimumThreshold(minimum_threshold);

       auto set_max_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&input), 1, *plugin);
       TRTORCH_CHECK(set_max_layer, "Unable to create set_max plugin from node" << *n);

       set_max_layer->setName(util::node_info(n).c_str());

       auto set_max_out = ctx->AssociateValueAndTensor(n->outputs()[0], set_max_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << set_max_out->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
