#include "conv2d_plugin.h"

#include <cudnn.h>

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

Conv2DPlugin::Conv2DPlugin() : padding_(0, 0), stride_(1, 1) {
  cudnnCreate(&cudnn_);
}

Conv2DPlugin::Conv2DPlugin(cudnnHandle_t cudnn) : cudnn_(cudnn), padding_(0, 0), stride_(1, 1) {}

Conv2DPlugin::Conv2DPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);

  {
    torch::IValue value;
    input_archive.read("padding", value);
    auto padding_vector = value.toIntVector();
    padding_ = {static_cast<int32_t>(padding_vector[0]), static_cast<int32_t>(padding_vector[1])};
  }

  {
    torch::IValue value;
    input_archive.read("stride", value);
    auto stride_vector = value.toIntVector();
    stride_ = {static_cast<int32_t>(stride_vector[0]), static_cast<int32_t>(stride_vector[1])};
  }

  cudnnCreate(&cudnn_);
}

void Conv2DPlugin::setPadding(const nvinfer1::Dims2& padding) {
  padding_ = padding;
}

void Conv2DPlugin::setStride(const nvinfer1::Dims2& stride) {
  stride_ = stride;
}

nvinfer1::Dims2 Conv2DPlugin::getPadding() const {
  return padding_;
}

nvinfer1::Dims2 Conv2DPlugin::getStride() const {
  return stride_;
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
  auto plugin = new Conv2DPlugin(cudnn_);
  plugin->setPadding(getPadding());
  plugin->setStride(getStride());
  return plugin;
}

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

nvinfer1::DimsExprs Conv2DPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  auto get_dimension = [&](const nvinfer1::IDimensionExpr &input_size, const nvinfer1::IDimensionExpr &kernel_size, int padding, int stride) {
    return exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
              *exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV,
                *exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                  *exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, input_size, kernel_size),
                  *exprBuilder.constant(padding * 2)),
                *exprBuilder.constant(stride)),
              *exprBuilder.constant(1));
  };

  nvinfer1::DimsExprs output;
  output.nbDims = 4;

  output.d[0] = inputs[0].d[0];
  output.d[1] = inputs[1].d[0];
  output.d[2] = get_dimension(*inputs[0].d[2], *inputs[1].d[2], padding_.d[0], stride_.d[0]);
  output.d[3] = get_dimension(*inputs[0].d[3], *inputs[1].d[3], padding_.d[1], stride_.d[1]);

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

  output_archive.write("padding", torch::IValue(std::vector<int64_t>{padding_.d[0], padding_.d[1]}));
  output_archive.write("stride", torch::IValue(std::vector<int64_t>{stride_.d[0], stride_.d[1]}));

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
  TRTORCH_ASSERT(0 <= pos && pos <= 2, "There should be exactly 3 connections to the plugin - 2 input, 1 output");
  TRTORCH_ASSERT(nbInputs == 2, "Expected a two tensors as input to conv2d plugin");
  TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to conv2d plugin");

  const PluginTensorDesc& in1 = inOut[0];

  const PluginTensorDesc& in2 = inOut[1];

  if (pos == 0) {
    return (in1.type == nvinfer1::DataType::kFLOAT) && (in1.format == nvinfer1::TensorFormat::kLINEAR);
  }

  if (pos == 1) {
    return (in2.type == nvinfer1::DataType::kFLOAT) && (in2.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 2, accessing information about output tensor
  const PluginTensorDesc& out = inOut[2];

  return (in1.type == out.type) && (in1.format == out.format);
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
  at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc[0].dims), [](void*) {}, tensor_options_);
  auto input_ptr = (float *)input.data_ptr();
  at::Tensor weights = at::from_blob((void*)inputs[1], util::toVec(inputDesc[1].dims), [](void*) {}, tensor_options_);
  auto weights_ptr = (float *)weights.data_ptr();
  at::Tensor output = at::from_blob(outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);
  auto output_ptr = (float *)output.data_ptr();

  std::cout << "input 0 dims " << inputDesc[0].dims << std::endl;
  std::cout << "input 1 dims " << inputDesc[1].dims << std::endl;
  std::cout << "out dims " << outputDesc->dims << std::endl;
  std::cout << "padding " << padding_ << std::endl;
  std::cout << "stride " << stride_ << std::endl;

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

    cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/inputDesc[0].dims.d[0],
                                        /*channels=*/inputDesc[0].dims.d[1],
                                        /*image_height=*/inputDesc[0].dims.d[2],
                                        /*image_width=*/inputDesc[0].dims.d[3]));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/inputDesc[1].dims.d[0],
                                        /*in_channels=*/inputDesc[1].dims.d[1],
                                        /*kernel_height=*/inputDesc[1].dims.d[2],
                                        /*kernel_width=*/inputDesc[1].dims.d[3]));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/padding_.d[0],
                                             /*pad_width=*/padding_.d[1],
                                             /*vertical_stride=*/stride_.d[0],
                                             /*horizontal_stride=*/stride_.d[1],
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

  int batch_size = 0;
  int channels = 0;
  int height = 0;
  int width = 0;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/outputDesc[0].dims.d[0],
                                        /*channels=*/outputDesc[0].dims.d[1],
                                        /*image_height=*/outputDesc[0].dims.d[2],
                                        /*image_width=*/outputDesc[0].dims.d[3]));

  int num_algos = 0;
  cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm_v7(cudnn_,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          1,
                                          &num_algos,
                                          &convolution_algorithm));

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm.algo,
                                                     &workspace_bytes));

  void* d_workspace = nullptr;
  cudaMalloc(&d_workspace, workspace_bytes);

  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(cudnn_,
                                     &alpha,
                                     input_descriptor,
                                     input_ptr,
                                     kernel_descriptor,
                                     weights_ptr,
                                     convolution_descriptor,
                                     convolution_algorithm.algo,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     output_ptr));

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
  return "Conv2D";
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
    const char* name) {
  name_ = name;
  return new Conv2DPlugin();
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