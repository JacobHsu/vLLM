import os
import sys

# FlashInfer JIT-compiles some kernels with nvcc, but pip-installed vLLM has no
# system CUDA toolkit. Point it at the nvcc that ships inside the nvidia-cuda-nvcc
# pip package instead of requiring a full CUDA Toolkit install.
_cuda_dir = os.path.normpath(
    os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.12",
                 "site-packages", "nvidia", "cu13")
)
if os.path.isdir(_cuda_dir):
    os.environ.setdefault("CUDA_HOME", _cuda_dir)
    os.environ["PATH"] = os.path.join(_cuda_dir, "bin") + os.pathsep + os.environ.get("PATH", "")

# Blackwell (sm_120) known issue: FlashInfer's JIT sampler fails to build against
# this CUDA toolchain (https://github.com/vllm-project/vllm/issues/44305).
# Fall back to the PyTorch-native sampler.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# WSL2 doesn't expose CUDA Unified Virtual Addressing by default, which the
# vLLM model runner requires; this opts in (needs WSL kernel >= 4.19.121).
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="facebook/opt-125m", enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
