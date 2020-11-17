#include "nonzero_plugin.h"

using namespace nvinfer1;

void NonZeroKernel(float *input, float *output, int32_t *index, int batch_size, int channels, int height, int width);

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * NonZeroPlugin class implementations
 */

NonZeroPlugin::NonZeroPlugin() {}

NonZeroPlugin::NonZeroPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
}

int NonZeroPlugin::getNbOutputs() const {
  return 1;
}

const char* NonZeroPlugin::getPluginType() const {
  return "NonZero";
}

const char* NonZeroPlugin::getPluginVersion() const {
  return "1";
}

const char* NonZeroPlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* NonZeroPlugin::clone() const {
  return new NonZeroPlugin();
}

nvinfer1::DimsExprs NonZeroPlugin::getOutputDimensions(
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

nvinfer1::DataType NonZeroPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const {
  return DataType::kFLOAT;
}

int NonZeroPlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);

  return 0;
}

void NonZeroPlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();

  data.copy((char*)buffer, size);
}

std::string NonZeroPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t NonZeroPlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool NonZeroPlugin::supportsFormatCombination(
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

void NonZeroPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
  dtype_ = DataType::kFLOAT;
}

size_t NonZeroPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int NonZeroPlugin::enqueue(
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

  auto num_dims = input.dim();
  auto dims = input.sizes();
  const int batch_size = num_dims > 3 ? dims[num_dims - 3] : 1;
  const int channels = num_dims > 2 ? dims[num_dims - 2] : 1;
  const int height = num_dims > 1 ? dims[num_dims - 1] : 1;
  const int width = dims[num_dims - 1];

  NonZeroKernel(input_ptr, output_ptr, index_ptr, batch_size, channels, height, width);

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);

  return 0;
}

/*
 * NonZeroPluginCreator class implementations
 */
const char* NonZeroPluginCreator::getPluginNamespace() const {
  return "";
}

const char* NonZeroPluginCreator::getPluginName() const {
  return "NonZero";
}

const char* NonZeroPluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* NonZeroPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

NonZeroPlugin* NonZeroPluginCreator::createPlugin(
    const char* name) {
  name_ = name;
  return new NonZeroPlugin();
}

nvinfer1::IPluginV2* NonZeroPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new NonZeroPlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* NonZeroPluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(NonZeroPluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch