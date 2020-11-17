#include "conv2d_plugin.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * Conv2DPlugin class implementations
 */

Conv2DPlugin::Conv2DPlugin() {
  cudnnCreate(&cudnn_);
}

Conv2DPlugin::Conv2DPlugin(cudnnHandle_t cudnn) : cudnn_(cudnn) {}

Conv2DPlugin::Conv2DPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
}

int Conv2DPlugin::getNbOutputs() const {
  return 1;
}

const char* Conv2DPlugin::getPluginType() const {
  return "Conv2D";
}

const char* Conv2DPlugin::getPluginVersion() const {
  return "1";
}

const char* Conv2DPlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* Conv2DPlugin::clone() const {
  return new Conv2DPlugin();
}

nvinfer1::DimsExprs Conv2DPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output;
  output.nbDims = 2;

  output.d[0] = exprBuilder.constant(1);
  output.d[1] = exprBuilder.constant(4);

  return output;
}

nvinfer1::DataType Conv2DPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const {
  return DataType::kFLOAT;
}

int Conv2DPlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);

  return 0;
}

void Conv2DPlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();

  data.copy((char*)buffer, size);
}

std::string Conv2DPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t Conv2DPlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool Conv2DPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) {
  TRTORCH_ASSERT(0 <= pos && pos <= 1, "There should be exactly 2 connections to the plugin - 1 input, 1 output");
  TRTORCH_ASSERT(nbInputs == 1, "Expected a single tensor as input to interpolate plugin");
  TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to interpolate plugin");

  const PluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const PluginTensorDesc& out = inOut[1];

  return (in.type == out.type) && (in.format == out.format);
}

void Conv2DPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
  dtype_ = DataType::kFLOAT;
}

size_t Conv2DPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int Conv2DPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc->dims), [](void*) {}, tensor_options_);
  auto input_ptr = (float *)input.data_ptr();
  at::Tensor output = at::from_blob(outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);
  auto output_ptr = (float *)output.data_ptr();

  at::Tensor index = at::zeros({1}, at::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  auto index_ptr = (int32_t *)index.data_ptr();

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);

  return 0;
}

/*
 * Conv2DPluginCreator class implementations
 */
const char* Conv2DPluginCreator::getPluginNamespace() const {
  return "";
}

const char* Conv2DPluginCreator::getPluginName() const {
  return "Interpolate";
}

const char* Conv2DPluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* Conv2DPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

Conv2DPlugin* Conv2DPluginCreator::createPlugin(
    const char* name,
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> size,
    std::string mode,
    bool align_corners) {
  name_ = name;
  return new Conv2DPlugin(in_shape, out_shape, size, mode, align_corners);
}

nvinfer1::IPluginV2* Conv2DPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new Conv2DPlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* Conv2DPluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(Conv2DPluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch