#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <torch/extension.h>

int calculate_output_size(int in, int padding, int dilation, int kernel,
                          int stride) {
  return (in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

#define CUDNN_CALL(f)                                                          \
  {                                                                            \
    cudnnStatus_t err = (f);                                                   \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      std::cout << "    Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

// This is not a real function, just an example of the mapping.
inline at::ScalarType getScalarType(cudnnDataType_t cudnn_data_type) {
  if (cudnn_data_type == CUDNN_DATA_FLOAT) {
    return at::kFloat;
  } else if (cudnn_data_type == CUDNN_DATA_HALF) {
    return at::kHalf;
  } else if (cudnn_data_type == CUDNN_DATA_DOUBLE) {
    return at::kDouble;
  } else if (cudnn_data_type == CUDNN_DATA_BFLOAT16) {
    return at::kBFloat16;
  } else if (cudnn_data_type == CUDNN_DATA_INT8) {
    return at::kChar;
  } else if (cudnn_data_type == CUDNN_DATA_INT32) {
    return at::kInt;
  }
  // If the cudnn_data_type isn't in the map, you'd handle the error.
  // A real implementation might throw an exception or return an optional.
  TORCH_CHECK(false, "Unsupported cudnnDataType_t value.", cudnn_data_type);
}

inline cudnnDataType_t getDataType(const at::Tensor &t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kChar) {
    return CUDNN_DATA_INT8;
  }  else if (scalar_type == at::kInt) {
    return CUDNN_DATA_INT32;
  }
  TORCH_CHECK(false, "TensorDescriptor does not support ", scalar_type);
}

at::Tensor my_custom_convolution2d_forward(
    const at::Tensor &input, const at::Tensor &weight,
    const std::vector<int32_t> &padding, const std::vector<int32_t> &stride,
    const std::vector<int32_t> &upscale, const int groups,
    const bool allow_tf32, const cudnnDataType_t dataType,
    const cudnnMathType_t mathType, const cudnnDataType_t outputDataType) {
  cudnnHandle_t handle = at::native::getCudnnHandle();
  // Create descriptors
  at::native::TensorDescriptor input_desc;
  at::native::FilterDescriptor weight_desc;
  at::native::TensorDescriptor output_desc;
  at::native::ConvolutionDescriptor conv_desc;

  // n, h, w, in_c
  AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      input_desc.mut_desc(), CUDNN_TENSOR_NHWC, getDataType(input),
      input.size(0), input.size(3), input.size(1), input.size(2)));
  // Set filter type
  // out_c, kernel_h, kernel_w, in_c
  AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      weight_desc.mut_desc(), getDataType(input), CUDNN_TENSOR_NHWC,
      weight.size(0), weight.size(3), weight.size(1), weight.size(2)));

  // Set cudnn convolution type
  const int dim = 2;
  // we can also use cudnnSetConvolution2dDescriptor
  AT_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      conv_desc.mut_desc(), dim, const_cast<int *>(padding.data()),
      const_cast<int *>(stride.data()), const_cast<int *>(upscale.data()),
      CUDNN_CROSS_CORRELATION, dataType));
  AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc.mut_desc(), groups));
  // See Note [behavior of cudnnFind and cudnnGet]
  AT_CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.mut_desc(), mathType));

  // Determine output size
  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
      conv_desc.mut_desc(), input_desc.mut_desc(), weight_desc.mut_desc(),
      &out_n, &out_c, &out_h, &out_w));

  // Create output tensor and set description
  at::Tensor output =
      at::empty({out_n, out_h, out_w, out_c},
                input.options().dtype(getScalarType(outputDataType)));
  AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc.mut_desc(),
                                            CUDNN_TENSOR_NHWC, outputDataType,
                                            out_n, out_c, out_h, out_w));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
  //       handle,
  //       input_desc.mut_desc(), weight_desc.mut_desc(), conv_desc.mut_desc(),
  //       output_desc.mut_desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
  //       &algo));
  algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // Determine workspace size
  size_t workspace_size = 0;
  AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      handle, input_desc.desc(), weight_desc.desc(), conv_desc.desc(),
      output_desc.desc(), algo, &workspace_size));

  // Allocate workspace
  at::Tensor workspace =
      at::empty(workspace_size, input.options().dtype(at::kByte));

  // Call cuDNN
  float alpha = 1.0f;
  float beta = 0.0f;
  AT_CUDNN_CHECK(cudnnConvolutionForward(
      handle, &alpha, input_desc.desc(), input.data_ptr(), weight_desc.desc(),
      weight.data_ptr(), conv_desc.desc(), algo, workspace.data_ptr(),
      workspace_size, &beta, output_desc.desc(), output.data_ptr()));
  return output;
}

// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_custom_convolution2d_forward", &my_custom_convolution2d_forward,
        "Custom Convolution Forward NHWC format (CUDA)");
  py::enum_<cudnnDataType_t>(m, "cudnnDataType_t")
      .value("CUDNN_DATA_FLOAT", cudnnDataType_t::CUDNN_DATA_FLOAT)
      .value("CUDNN_DATA_DOUBLE", cudnnDataType_t::CUDNN_DATA_DOUBLE)
      .value("CUDNN_DATA_HALF", cudnnDataType_t::CUDNN_DATA_HALF)
      .value("CUDNN_DATA_INT8", cudnnDataType_t::CUDNN_DATA_INT8)
      .value("CUDNN_DATA_INT32", cudnnDataType_t::CUDNN_DATA_INT32)
      .value("CUDNN_DATA_INT8x4", cudnnDataType_t::CUDNN_DATA_INT8x4)
      .value("CUDNN_DATA_UINT8", cudnnDataType_t::CUDNN_DATA_UINT8)
      .value("CUDNN_DATA_UINT8x4", cudnnDataType_t::CUDNN_DATA_UINT8x4)
      .value("CUDNN_DATA_INT8x32", cudnnDataType_t::CUDNN_DATA_INT8x32)
      .value("CUDNN_DATA_BFLOAT16", cudnnDataType_t::CUDNN_DATA_BFLOAT16)
      .value("CUDNN_DATA_INT64", cudnnDataType_t::CUDNN_DATA_INT64)
      .value("CUDNN_DATA_BOOLEAN", cudnnDataType_t::CUDNN_DATA_BOOLEAN)
      .value("CUDNN_DATA_FP8_E4M3", cudnnDataType_t::CUDNN_DATA_FP8_E4M3)
      .value("CUDNN_DATA_FP8_E5M2", cudnnDataType_t::CUDNN_DATA_FP8_E5M2)
      .value("CUDNN_DATA_FAST_FLOAT_FOR_FP8",
             cudnnDataType_t::CUDNN_DATA_FAST_FLOAT_FOR_FP8)
      .value("CUDNN_DATA_FP8_E8M0", cudnnDataType_t::CUDNN_DATA_FP8_E8M0)
      .value("CUDNN_DATA_FP4_E2M1", cudnnDataType_t::CUDNN_DATA_FP4_E2M1);

  py::enum_<cudnnMathType_t>(m, "cudnnMathType_t")
      .value("CUDNN_DEFAULT_MATH", cudnnMathType_t::CUDNN_DEFAULT_MATH)
      .value("CUDNN_TENSOR_OP_MATH", cudnnMathType_t::CUDNN_TENSOR_OP_MATH)
      .value("CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",
             cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
      .value("CUDNN_FMA_MATH", cudnnMathType_t::CUDNN_FMA_MATH);
}