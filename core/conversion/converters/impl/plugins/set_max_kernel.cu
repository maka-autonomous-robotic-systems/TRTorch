#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/torch.h>

void SetMaxKernel(
    float* input_ptr,
    float* output_ptr,
    float min_threshold,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width) {
  std::cout << "get max" << std::endl;
  const auto num_el = batch_size * channels * height * width;

  thrust::device_ptr<float> input_dev_ptr = thrust::device_pointer_cast(input_ptr);
  auto max_iter = thrust::max_element(input_dev_ptr, input_dev_ptr + num_el);

  float max_value = *max_iter;
  std::cout << "clamp" << std::endl;
  std::cout << "max value " << max_value << std::endl;

  max_value = std::max(max_value, min_threshold);

  const float out_height_scale = out_height / height;
  const float out_width_scale = out_width / width;

  thrust::device_ptr<float> outut_dev_ptr = thrust::device_pointer_cast(input_ptr);
  std::cout << "start thrust" << std::endl;

  thrust::fill(outut_dev_ptr, outut_dev_ptr + batch_size * channels * out_height * out_width, 0.0f);

  thrust::counting_iterator<int64_t> count_iter(0);
  thrust::for_each(count_iter, count_iter + num_el, [=] __device__(int64_t index) {
    if (input_ptr[index] < max_value)
      return;

    const int batch = index / (channels * height * width);
    index = index % (channels * height * width);

    const int channel = index / (height * width);
    index = index % (height * width);

    const int y = index / width;
    const int x = index % width;

    const int y_output = min((int)round((y + 0.5f) * out_height_scale), height - 1);
    const int x_output = min((int)round((x + 0.5f) * out_width_scale), width - 1);

    const int out_index = batch * (channels * out_height * out_width) + channel * (out_height * out_width) +
        y_output * out_width + x_output;

    output_ptr[out_index] = 1.0f;
  });
}
