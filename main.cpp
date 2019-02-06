#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#if defined(__APPLE__) || defined(__MACOSX)
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

constexpr bool callSuccessful(cudaError_t status)
{
    return status == cudaSuccess;
}

constexpr bool callSuccessful(cl_int status)
{
  return status == CL_SUCCESS;
}

#define CHECK_GPU_CALL(x) \
  do \
  { \
    if (!callSuccessful((x))) \
    { \
      std::cerr << __func__ << ':' << __LINE__ << " Error in: " << #x << '\n'; \
      std::exit(-1); \
    } \
  } \
  while (false) \


int main(int, char**)
{
  const std::size_t N = 1024;

  std::unique_ptr<int> h_ptr(new int[1024]);
  std::fill_n(h_ptr.get(), N, 0);

  // OpenCL
  cl_uint ocl_platform_count = 0;
  CHECK_GPU_CALL(
    clGetPlatformIDs(0,
                     nullptr,
                     &ocl_platform_count));
  std::vector<cl_platform_id> ocl_platform_ids(ocl_platform_count);
  CHECK_GPU_CALL(
    clGetPlatformIDs(ocl_platform_count,
                     ocl_platform_ids.data(),
                     nullptr));

  cl_uint ocl_device_count = 0;
  CHECK_GPU_CALL(
    clGetDeviceIDs(ocl_platform_ids[0],
                   CL_DEVICE_TYPE_GPU,
                   0,
                   nullptr,
                   &ocl_device_count));
  std::vector<cl_device_id> ocl_device_ids(ocl_device_count);
  CHECK_GPU_CALL(
    clGetDeviceIDs(ocl_platform_ids[0],
                   CL_DEVICE_TYPE_GPU,
                   ocl_device_count,
                   ocl_device_ids.data(),
                   nullptr));

  const cl_context_properties ocl_context_props [] =
  {
    CL_CONTEXT_PLATFORM,
    reinterpret_cast<cl_context_properties>(ocl_platform_ids[0]),
    0, 0
  };

  cl_int ocl_error = CL_SUCCESS;

  cl_context ocl_context = clCreateContext(ocl_context_props,
                                           ocl_device_count,
                                           ocl_device_ids.data(),
                                           nullptr,
                                           nullptr,
                                           &ocl_error);
  CHECK_GPU_CALL(ocl_error);

  cl_device_id ocl_device_id = ocl_device_ids[0];

  cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(ocl_context,
                                                                  ocl_device_id,
                                                                  0,
                                                                  &ocl_error);
  CHECK_GPU_CALL(ocl_error);

  cl_mem ocl_buffer = clCreateBuffer(ocl_context,
                                     CL_MEM_USE_HOST_PTR,
                                     N * sizeof(int),
                                     h_ptr.get(),
                                     &ocl_error);
  CHECK_GPU_CALL(ocl_error);


  // init CUDA
  CHECK_GPU_CALL(cudaHostRegister(h_ptr.get(), N * sizeof(*h_ptr), 0));


  // initialize using OpenCL
  int* mapped_buffer = static_cast<int*>(clEnqueueMapBuffer(ocl_queue,
                                                            ocl_buffer,
                                                            CL_TRUE,
                                                            CL_MAP_WRITE,
                                                            0,
                                                            N * sizeof(int),
                                                            0,
                                                            nullptr,
                                                            nullptr,
                                                            &ocl_error));
  CHECK_GPU_CALL(ocl_error);

  std::fill_n(h_ptr.get(), N, 42);

  CHECK_GPU_CALL(
    clEnqueueUnmapMemObject(ocl_queue,
                            ocl_buffer,
                            mapped_buffer,
                            0,
                            nullptr,
                            nullptr));

  // retrieve using CUDA
  std::vector<int> v(N);
  CHECK_GPU_CALL(
    cudaMemcpy(v.data(),
               h_ptr.get(),
               N * sizeof(*h_ptr),
               cudaMemcpyDeviceToHost));

  if (!std::all_of(v.begin(), v.end(), [](int i) { return i == 42; }))
  {
    std::cerr << "Did not retrieve correct data.\n";
    std::exit(-1);
  }

  // free OpenCL
  CHECK_GPU_CALL(clReleaseMemObject(ocl_buffer));
  CHECK_GPU_CALL(clReleaseCommandQueue(ocl_queue));
  CHECK_GPU_CALL(clReleaseContext(ocl_context));

  // free CUDA
  CHECK_GPU_CALL(cudaHostUnregister(h_ptr.get()));

  return 0;
}
