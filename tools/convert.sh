# Same 4.50 bpw as NVFP4 (closest quality match):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4.gguf Q4_0_ROCMFP4

# Smaller 4.25 bpw (speed-first layout, a little more loss):
build-strix-rocmfp4/bin/llama-quantize --allow-requantize \
  model-NVFP4.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST
