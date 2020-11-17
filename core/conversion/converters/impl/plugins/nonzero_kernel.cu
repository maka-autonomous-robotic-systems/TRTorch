#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>


void NonZeroKernel(float *input_ptr, float *output_ptr, int32_t *index_ptr, int batch_size, int channels, int height, int width) {
  thrust::counting_iterator<long> count_iter(0);

  thrust::for_each(count_iter, count_iter + ((long)batch_size * channels * height * width), [=] __device__ (long index) {
    if (input_ptr[index] == 0) return;
    
    const int batch = index / (channels * height * width);
    index = index % (channels * height * width);

    const int channel = index / (height * width);
    index = index % (height * width);

    const int y = index / width;
    const int x = index % width;

    const int32_t out_index = atomicAdd(index_ptr, 1);
    if (out_index == 0) {
      output_ptr[out_index + 0] = batch;
      output_ptr[out_index + 1] = channel;
      output_ptr[out_index + 2] = y;
      output_ptr[out_index + 3] = x;
    }
  });
}