#include "set_max_plugin.h"

using namespace nvinfer1;

void SetMaxKernel(
    float* input_ptr,
    float* output_ptr,
    float min_threshold,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width);

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * SetMaxPlugin class implementations
 */

SetMaxPlugin::SetMaxPlugin() {}

SetMaxPlugin::SetMaxPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);

  {
    torch::IValue value;
    input_archive.read("out_dims", value);
    auto out_dims_vector = value.toIntVector();
    out_dims_ = {static_cast<int32_t>(out_dims_vector[0]), static_cast<int32_t>(out_dims_vector[1])};
  }
  {
    torch::IValue value;
    input_archive.read("min_threshold", value);
    min_threshold_ = value.toDouble();
  }
}

void SetMaxPlugin::setOutDimensions(const nvinfer1::Dims2& out_dimensions) {
  out_dims_ = out_dimensions;
}

void SetMaxPlugin::setMinimumThreshold(float threshold) {
  min_threshold_ = threshold;
}

nvinfer1::Dims2 SetMaxPlugin::getOutDimensions() const {
  return out_dims_;
}

float SetMaxPlugin::getMinimumThreshold() const {
  return min_threshold_;
}

int SetMaxPlugin::getNbOutputs() const {
  return 1;
}

const char* SetMaxPlugin::getPluginType() const {
  return "SetMax";
}

const char* SetMaxPlugin::getPluginVersion() const {
  return "1";
}

const char* SetMaxPlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* SetMaxPlugin::clone() const {
  auto plugin = new SetMaxPlugin();
  plugin->setOutDimensions(getOutDimensions());
  plugin->setMinimumThreshold(getMinimumThreshold());
  return plugin;
}

nvinfer1::DimsExprs SetMaxPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output = inputs[0];

  for (int i = 0; i < out_dims_.nbDims; i++) {
    output.d[i + (output.nbDims - out_dims_.nbDims)] = exprBuilder.constant(out_dims_.d[i]);
  }

  return output;
}

nvinfer1::DataType SetMaxPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const {
  return DataType::kFLOAT;
}

int SetMaxPlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);

  return 0;
}

void SetMaxPlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();

  data.copy((char*)buffer, size);
}

std::string SetMaxPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  output_archive.write("out_dims", torch::IValue(std::vector<int64_t>{out_dims_.d[0], out_dims_.d[1]}));
  output_archive.write("min_threshold", torch::IValue(min_threshold_));

  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t SetMaxPlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool SetMaxPlugin::supportsFormatCombination(
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

void SetMaxPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
  dtype_ = DataType::kFLOAT;
}

size_t SetMaxPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int SetMaxPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc->dims), [](void*) {}, tensor_options_);
  auto input_ptr = (float*)input.data_ptr();
  at::Tensor output = at::from_blob(
      outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);
  auto output_ptr = (float*)output.data_ptr();

  std::cout << "input 0 dims " << inputDesc->dims << std::endl;
  std::cout << "out dims " << outputDesc->dims << std::endl;
  std::cout << "requested out dims " << out_dims_ << std::endl;
  std::cout << "min threshold " << min_threshold_ << std::endl;

  auto num_dims = input.dim();
  auto dims = input.sizes();
  const int batch_size = num_dims > 3 ? dims[num_dims - 4] : 1;
  const int channels = num_dims > 2 ? dims[num_dims - 3] : 1;
  const int height = num_dims > 1 ? dims[num_dims - 2] : 1;
  const int width = dims[num_dims - 1];

  auto output_num_dims = input.dim();
  auto output_dims = input.sizes();
  const int output_height = output_num_dims > 1 ? output_dims[output_num_dims - 2] : 1;
  const int output_width = output_dims[output_num_dims - 1];

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  std::cout << "kernel" << std::endl;

  SetMaxKernel(input_ptr, output_ptr, min_threshold_, batch_size, channels, height, width, output_height, output_width);

  std::cout << "done" << std::endl;

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);

  return 0;
}

/*
 * SetMaxPluginCreator class implementations
 */
const char* SetMaxPluginCreator::getPluginNamespace() const {
  return "";
}

const char* SetMaxPluginCreator::getPluginName() const {
  return "SetMax";
}

const char* SetMaxPluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* SetMaxPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

SetMaxPlugin* SetMaxPluginCreator::createPlugin(const char* name) {
  name_ = name;
  return new SetMaxPlugin();
}

nvinfer1::IPluginV2* SetMaxPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new SetMaxPlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* SetMaxPluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(SetMaxPluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch