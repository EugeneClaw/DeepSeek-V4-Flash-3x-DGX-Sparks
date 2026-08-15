#!/usr/bin/env python3
"""Compat shim: FlashInfer 0.6.15 Python passes 8 args to MoERunner.init;
the Anemll AOT fused_moe_120 module is still the 7-arg signature
(no use_fused_finalize). Fall back so CUTLASS MXFP4 can actually apply.
"""
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py"
)

OLD = """            if instance_key not in MoERunner.runner_dict:
                MoERunner.runner_dict[instance_key] = module.init(
                    x_dtype,
                    weight_dtype,
                    output_dtype,
                    use_deepseek_fp8_block_scale,
                    use_w4_group_scaling,
                    use_mxfp8_act_scaling,
                    use_packed_weights,
                    use_fused_finalize,
                )
"""

NEW = """            if instance_key not in MoERunner.runner_dict:
                try:
                    MoERunner.runner_dict[instance_key] = module.init(
                        x_dtype,
                        weight_dtype,
                        output_dtype,
                        use_deepseek_fp8_block_scale,
                        use_w4_group_scaling,
                        use_mxfp8_act_scaling,
                        use_packed_weights,
                        use_fused_finalize,
                    )
                except TypeError as _fi_init_err:
                    # AOT fused_moe_120: init(3 dtypes + 4 bools). 0.6.15
                    # Python added use_fused_finalize as the 8th arg.
                    if "Expected 7" not in str(_fi_init_err):
                        raise
                    MoERunner.runner_dict[instance_key] = module.init(
                        x_dtype,
                        weight_dtype,
                        output_dtype,
                        use_deepseek_fp8_block_scale,
                        use_w4_group_scaling,
                        use_mxfp8_act_scaling,
                        use_packed_weights,
                    )
"""


def main() -> None:
    text = TARGET.read_text()
    if "Expected 7" in text and "_fi_init_err" in text:
        print(f"already patched: {TARGET}")
        return
    if OLD not in text:
        raise SystemExit(f"anchor not found in {TARGET}")
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
