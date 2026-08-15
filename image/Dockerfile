# Anemll 0.1.1 + FlashInfer CUTLASS MXFP4 convert so --enable-expert-parallel
# can load DeepSeek-V4 on GB10. Does not rebuild vLLM/FlashInfer/b12x.
FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1

COPY overlay/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
COPY overlay/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
COPY patches/patch_flashinfer_moe_init.py /tmp/patch_flashinfer_moe_init.py
RUN python3 /tmp/patch_flashinfer_moe_init.py && rm /tmp/patch_flashinfer_moe_init.py

LABEL org.opencontainers.image.title="dspark-vllm-gx10 cutlass-ep"
LABEL org.opencontainers.image.description="Anemll 0.1.1 plus CUTLASS MXFP4 convert for expert-parallel DeepSeek-V4"
LABEL org.opencontainers.image.base.name="ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
