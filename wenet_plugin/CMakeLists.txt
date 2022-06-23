cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(MHAPlugin LANGUAGES CXX CUDA)

if(NOT MSVC)
  # Enable all compile warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(NOT MSVC)
  set_ifndef(TRT_LIB /opt/tensorrt/lib)
  set_ifndef(TRT_INCLUDE /opt/tensorrt/include)
  set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
  set_ifndef(CUDA_LIB /usr/local/cuda/lib64)
endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})
find_library(_CUBLAS_LIB cublas HINTS ${CUDA_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

# -------- BUILDING --------

add_compile_definitions(TENSORRT_BUILD_LIB)
# Add include directories
include_directories(${COMMON_HEADER_DIRS} ${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/plugin/)
add_subdirectory(plugin/ft_utils)

# Define plugin library target
FILE(GLOB pluginSources plugin/*.cu)
message("-- pluginSources : ${pluginSources}")

add_library(mhalugin MODULE
  ${pluginSources}
)

# Use C++14
target_compile_features(mhalugin PUBLIC cxx_std_14 )

# Link TensorRT's nvinfer lib
  target_link_libraries(mhalugin PRIVATE ${NVINFER_LIB} ${CUBLAS_LIB} cublasMMWrapper)

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
# set_target_properties(mhalugin PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
# )
# set_property(TARGET mhalugin PROPERTY CUDA_ARCHITECTURES 75)
find_package(CUDA 10.2 REQUIRED)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR} is greater or equal than 11, enable -DENABLE_BF16 flag")
endif()
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")

set(SM_SETS 52 60 61 70 75 80 86)
set(USING_WMMA False)
set(FIND_SM False)

foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86)
      set(USING_WMMA True)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  #                      -rdc=true")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
  set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0;8.6")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()



# add_subdirectory(FasterTransformer)
# add_subdirectory(FasterTransformer/3rdparty)
# add_subdirectory(FasterTransformer/src)

# set(COMMON_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/FasterTransformer
#   ${CUDA_PATH}/include
# )

# message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

# include_directories(
#   ${COMMON_HEADER_DIRS}
# )

# add_library(transformer-shared SHARED
#   $<TARGET_OBJECTS:BaseBeamSearchLayer>
#   $<TARGET_OBJECTS:BaseSamplingLayer>
#   $<TARGET_OBJECTS:BeamSearchLayer>
#   $<TARGET_OBJECTS:DecoderCrossAttentionLayer>
#   $<TARGET_OBJECTS:DecoderSelfAttentionLayer>
#   $<TARGET_OBJECTS:DynamicDecodeLayer>
#   $<TARGET_OBJECTS:FfnLayer>
#   $<TARGET_OBJECTS:FusedAttentionLayer>
#   $<TARGET_OBJECTS:GptContextAttentionLayer>
#   $<TARGET_OBJECTS:GptJ>
#   $<TARGET_OBJECTS:GptJContextDecoder>
#   $<TARGET_OBJECTS:GptJDecoder>
#   $<TARGET_OBJECTS:GptJDecoderLayerWeight>
#   $<TARGET_OBJECTS:GptJTritonBackend>
#   $<TARGET_OBJECTS:GptJWeight>
#   $<TARGET_OBJECTS:OnlineBeamSearchLayer>
#   $<TARGET_OBJECTS:ParallelGpt>
#   $<TARGET_OBJECTS:ParallelGptContextDecoder>
#   $<TARGET_OBJECTS:ParallelGptDecoder>
#   $<TARGET_OBJECTS:ParallelGptDecoderLayerWeight>
#   $<TARGET_OBJECTS:ParallelGptTritonBackend>
#   $<TARGET_OBJECTS:ParallelGptWeight>
#   $<TARGET_OBJECTS:T5Decoder>
#   $<TARGET_OBJECTS:T5Decoding>
#   $<TARGET_OBJECTS:T5Encoder>
#   $<TARGET_OBJECTS:T5TritonBackend>
#   $<TARGET_OBJECTS:TensorParallelDecoderCrossAttentionLayer>
#   $<TARGET_OBJECTS:TensorParallelDecoderSelfAttentionLayer>
#   $<TARGET_OBJECTS:TensorParallelGeluFfnLayer>
#   $<TARGET_OBJECTS:TensorParallelGptContextAttentionLayer>
#   $<TARGET_OBJECTS:TensorParallelReluFfnLayer>
#   $<TARGET_OBJECTS:TensorParallelUnfusedAttentionLayer>
#   $<TARGET_OBJECTS:TopKSamplingLayer>
#   $<TARGET_OBJECTS:TopKTopPSamplingLayer>
#   $<TARGET_OBJECTS:TopPSamplingLayer>
#   $<TARGET_OBJECTS:UnfusedAttentionLayer>
#   $<TARGET_OBJECTS:activation_int8_kernels>
#   $<TARGET_OBJECTS:activation_kernels>
#   $<TARGET_OBJECTS:add_bias_transpose_kernels>
#   $<TARGET_OBJECTS:add_residual_kernels>
#   $<TARGET_OBJECTS:ban_bad_words>
#   $<TARGET_OBJECTS:stop_criteria>
#   $<TARGET_OBJECTS:beam_search_penalty_kernels>
#   $<TARGET_OBJECTS:beam_search_topk_kernels>
#   $<TARGET_OBJECTS:bert_preprocess_kernels>
#   $<TARGET_OBJECTS:calibrate_quantize_weight_kernels>
#   $<TARGET_OBJECTS:cublasAlgoMap>
#   $<TARGET_OBJECTS:cublasMMWrapper>
#   $<TARGET_OBJECTS:decoder_masked_multihead_attention>
#   $<TARGET_OBJECTS:decoding_kernels>
#   $<TARGET_OBJECTS:gpt_kernels>
#   $<TARGET_OBJECTS:layernorm_int8_kernels>
#   $<TARGET_OBJECTS:layernorm_kernels>
#   $<TARGET_OBJECTS:layout_transformer_int8_kernels>
#   $<TARGET_OBJECTS:longformer_kernels>
#   $<TARGET_OBJECTS:matrix_transpose_kernels>
#   $<TARGET_OBJECTS:matrix_vector_multiplication>
#   $<TARGET_OBJECTS:memory_utils>
#   $<TARGET_OBJECTS:nccl_utils>
#   $<TARGET_OBJECTS:custom_ar_comm>
#   $<TARGET_OBJECTS:custom_ar_kernels>
#   $<TARGET_OBJECTS:word_list>
#   $<TARGET_OBJECTS:online_softmax_beamsearch_kernels>
#   $<TARGET_OBJECTS:quantization_int8_kernels>
#   $<TARGET_OBJECTS:sampling_penalty_kernels>
#   $<TARGET_OBJECTS:sampling_topk_kernels>
#   $<TARGET_OBJECTS:sampling_topp_kernels>
#   $<TARGET_OBJECTS:softmax_int8_kernels>
#   $<TARGET_OBJECTS:transpose_int8_kernels>
#   $<TARGET_OBJECTS:trt_fused_multi_head_attention>
#   $<TARGET_OBJECTS:unfused_attention_kernels>
#   $<TARGET_OBJECTS:logprob_kernels>)
# set_target_properties(transformer-shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(transformer-shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(transformer-shared PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(transformer-shared PUBLIC -lcudart -lnccl -lmpi -lcublas -lcublasLt -lcurand)
