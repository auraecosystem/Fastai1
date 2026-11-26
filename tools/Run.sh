build-strix-rocmfp4/bin/llama-cli \
  -m /path/to/model-Q8_0_ROCMFPX_AGENT.gguf \
  -dev ROCm0 \
  -ngl 999 \
  -fa on \
  -c 8192 \
  -b 512 \
  -ub 512 \
  --jinja
