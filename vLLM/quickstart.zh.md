# 快速入門 (Quickstart)

> 原文出處:[docs.vllm.ai — Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
> 本文為繁體中文翻譯版,僅供個人學習參考,內容以官方原文為準。

本指南將協助你快速上手 vLLM,完成以下兩件事:

- [離線批次推論](#離線批次推論-offline-batched-inference)
- [線上服務](#線上服務-online-serving)

## 事前準備 (Prerequisites)

- 作業系統:Linux
- Python:3.10 – 3.13

> **備註**
> vLLM 也能在 macOS 上運作,透過 [vLLM-Metal](https://github.com/vllm-project/vllm-metal) 使用 Apple Silicon GPU 加速。詳見 [GPU 安裝指南](installation/gpu.md),並選擇「Apple Silicon」分頁。

## 安裝 (Installation)

### NVIDIA CUDA

如果你使用 NVIDIA GPU,可以直接透過 [pip](https://pypi.org/project/vllm/) 安裝 vLLM。

建議使用 [uv](https://docs.astral.sh/uv/) 這個速度非常快的 Python 環境管理工具來建立及管理 Python 環境。請依照[官方文件](https://docs.astral.sh/uv/#getting-started)安裝 `uv`。安裝好 `uv` 之後,可以用以下指令建立新的 Python 環境並安裝 vLLM:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`uv` 可以透過 `--torch-backend=auto`(或環境變數 `UV_TORCH_BACKEND=auto`)檢查已安裝的 CUDA 驅動版本,[自動選擇合適的 PyTorch 索引來源](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)。若要指定特定版本(例如 `cu126`),可設定 `--torch-backend=cu126`(或 `UV_TORCH_BACKEND=cu126`)。

另一個方便的用法是搭配 `--with [dependency]` 使用 `uv run`,這樣就能直接執行 `vllm serve` 等指令,而不需要建立任何永久性的環境:

```bash
uv run --with vllm vllm --help
```

你也可以使用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 來建立與管理 Python 環境。若想在 conda 環境中使用 `uv`,可以透過 `pip` 安裝它。

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

### AMD ROCm

如果你使用 AMD GPU,可以透過 `uv` 安裝 vLLM。

建議使用 [uv](https://docs.astral.sh/uv/),因為它會將額外的套件索引來源設定為[比預設索引來源更高的優先順序](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。`uv` 同時也是速度非常快的 Python 環境管理工具。請依照[官方文件](https://docs.astral.sh/uv/#getting-started)安裝 `uv`。安裝好之後,用以下指令建立新環境並安裝 vLLM:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

> **備註**
> 目前僅支援 Python 3.12、ROCm 7.0,以及 `glibc >= 2.35`。

> **備註**
> 過去 Docker image 是透過 AMD 自家的 docker 發布流程建置,位於 `rocm/vllm-dev`。此做法已逐漸淘汰,改為使用 vLLM 自己的 docker 發布流程。

> **提示**
> 也提供每日建置的 nightly Docker image:[vllm/vllm-openai-rocm:nightly](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags),可用來測試最新的開發版本。

### Intel GPU

vLLM 透過 XPU 後端支援 Intel GPU。預先建置好的 XPU wheel 檔案即將推出。

自 v0.26.0 版本起,vLLM 已為 Intel GPU 加入官方 Docker image。也提供 nightly Docker image:[vllm/vllm-openai-xpu:nightly](https://hub.docker.com/r/vllm/vllm-openai-xpu/tags)。

> **提示**
> 更詳細的說明(包含從原始碼建置、Docker image 設定等),請參考 [GPU 安裝指南](installation/gpu.md),並選擇「Intel XPU」分頁。

### Google TPU

若要在 Google TPU 上執行 vLLM,需要安裝 `vllm-tpu` 套件。

```bash
uv pip install vllm-tpu
```

> **備註**
> 更詳細的說明(包含 Docker、從原始碼安裝、疑難排解等),請參考 [vLLM on TPU 官方文件](https://docs.vllm.ai/projects/tpu/en/latest/)。

### Ascend NPU

如果你使用 Ascend NPU,可以透過社群維護的硬體外掛 [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) 來執行 vLLM。

請依照 [vLLM Ascend 快速入門](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)中的安裝說明操作。

> **備註**
> Ascend 的設定會依你的 NPU 硬體與 CANN 版本而異。支援的版本、Docker image 及疑難排解,請參考 [vLLM Ascend 官方文件](https://docs.vllm.ai/projects/ascend/en/latest/)。

### Apple Silicon (Mac)

如果你使用搭載 Apple Silicon 的 Mac,可以透過 Apple 的 Metal 框架,使用 vLLM-Metal 進行 GPU 加速推論。

請依照 [vLLM-Metal 官方文件](https://github.com/vllm-project/vllm-metal#installation)中的安裝說明操作。

> **備註**
> vLLM-Metal 使用 MLX 而非 PyTorch 作為運算後端,需要使用來自 Hugging Face 上 [mlx-community](https://huggingface.co/mlx-community) 的 MLX 最佳化模型。

> **提示**
> 更詳細的說明,請參考 [GPU 安裝指南](installation/gpu.md),並選擇「Apple Silicon」分頁。

> **備註**
> 更多細節與非 CUDA 平台的安裝方式,請參考[安裝指南](installation/README.md)。

## 離線批次推論 (Offline Batched Inference)

安裝好 vLLM 之後,你就可以針對一份輸入提示詞(prompt)清單產生文字(也就是離線批次推論)。範例腳本請見:[examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

這個範例的第一行匯入了 [LLM][vllm.LLM] 與 [SamplingParams][vllm.SamplingParams] 這兩個類別:

- [LLM][vllm.LLM] 是使用 vLLM 引擎執行離線推論的主要類別。
- [SamplingParams][vllm.SamplingParams] 用來指定取樣(sampling)過程的參數。

```python
from vllm import LLM, SamplingParams
```

接下來這段程式碼定義了一份用於文字生成的輸入提示詞清單,以及取樣參數。[取樣溫度(sampling temperature)](https://arxiv.org/html/2402.05201v1)設為 `0.8`,[核採樣機率(nucleus sampling probability)](https://en.wikipedia.org/wiki/Top-p_sampling)設為 `0.95`。關於取樣參數的更多資訊可參考[這裡](../api/README.md#inference-parameters)。

> **重要**
> 預設情況下,若 Hugging Face 模型倉庫中存在 `generation_config.json`,vLLM 會套用模型作者建議的取樣參數。在大多數情況下,若未特別指定 [SamplingParams][vllm.SamplingParams],這樣做能為你帶來最佳結果。
>
> 但如果你偏好使用 vLLM 預設的取樣參數,請在建立 [LLM][vllm.LLM] 實例時設定 `generation_config="vllm"`。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

[LLM][vllm.LLM] 類別會初始化 vLLM 引擎,並載入 [OPT-125M 模型](https://arxiv.org/abs/2205.01068)以進行離線推論。支援的模型清單請見[這裡](../models/supported_models.md)。

```python
llm = LLM(model="facebook/opt-125m")
```

> **備註**
> 預設情況下,vLLM 會從 [Hugging Face](https://huggingface.co/) 下載模型。如果你想改用 [ModelScope](https://www.modelscope.cn) 上的模型,請在初始化引擎前設定環境變數 `VLLM_USE_MODELSCOPE`。
>
> ```shell
> export VLLM_USE_MODELSCOPE=True
> ```

接下來就是最有趣的部分了!透過 `llm.generate` 產生輸出結果。這個方法會將輸入的提示詞加入 vLLM 引擎的等待佇列,並執行引擎以高吞吐量產生輸出。輸出結果會以一份 `RequestOutput` 物件清單的形式回傳,其中包含所有輸出的 token。

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

> **備註**
> `llm.generate` 方法不會自動將模型的聊天範本(chat template)套用到輸入的提示詞上。因此,如果你使用的是 Instruct 模型或 Chat 模型,應該手動套用對應的聊天範本,才能得到預期的行為。或者,你也可以使用 `llm.chat` 方法,並傳入與 OpenAI `client.chat.completions` 相同格式的訊息清單:
>
> <details><summary>展開程式碼</summary>
>
> ```python
> # 使用 tokenizer 套用聊天範本
> from transformers import AutoTokenizer
>
> tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
> messages_list = [
>     [{"role": "user", "content": prompt}]
>     for prompt in prompts
> ]
> texts = tokenizer.apply_chat_template(
>     messages_list,
>     tokenize=False,
>     add_generation_prompt=True,
> )
>
> # 產生輸出
> outputs = llm.generate(texts, sampling_params)
>
> # 印出輸出結果
> for output in outputs:
>     prompt = output.prompt
>     generated_text = output.outputs[0].text
>     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
>
> # 使用 chat 介面
> outputs = llm.chat(messages_list, sampling_params)
> for idx, output in enumerate(outputs):
>     prompt = prompts[idx]
>     generated_text = output.outputs[0].text
>     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
> ```
>
> </details>

## 線上服務 (Online Serving)

vLLM 可以部署成一個實作 OpenAI API 協定的伺服器。這讓 vLLM 可以直接替換掉原本使用 OpenAI API 的應用程式。
預設情況下,伺服器會啟動於 `http://localhost:8000`。你可以透過 `--host` 與 `--port` 參數指定位址。該伺服器一次只服務一個模型,並實作了諸如[列出模型](https://platform.openai.com/docs/api-reference/models/list)、[建立聊天回應](https://platform.openai.com/docs/api-reference/chat/completions/create)、[建立文字補全](https://platform.openai.com/docs/api-reference/completions/create)等端點。

執行以下指令,以 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型啟動 vLLM 伺服器:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

> **備註**
> 預設情況下,伺服器會使用儲存在 tokenizer 中預先定義好的聊天範本。
> 若要覆寫此範本,可參考[這裡](../serving/online_serving/README.md#chat-template)的說明。

> **重要**
> 預設情況下,若 Hugging Face 模型倉庫中存在 `generation_config.json`,伺服器會套用其內容。這代表某些取樣參數的預設值,可能會被模型作者建議的數值覆蓋。
>
> 若要停用此行為,啟動伺服器時請加上 `--generation-config vllm` 參數。

這個伺服器可以用與 OpenAI API 相同的格式查詢。舉例來說,若要列出模型:

```bash
curl http://localhost:8000/v1/models
```

你可以透過 `--api-key` 參數或環境變數 `VLLM_API_KEY`,讓伺服器檢查標頭(header)中的 API 金鑰。
`--api-key` 之後可以傳入多組金鑰,伺服器會接受其中任何一組金鑰,這在金鑰輪替(key rotation)時相當實用。

### 使用 vLLM 搭配 OpenAI Completions API

伺服器啟動後,你可以用輸入的提示詞查詢模型:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

由於這個伺服器相容於 OpenAI API,你可以直接把它當作任何使用 OpenAI API 之應用程式的替代品。舉例來說,另一種查詢伺服器的方式是透過 `openai` 這個 Python 套件:

<details><summary>展開程式碼</summary>

```python
from openai import OpenAI

# 修改 OpenAI 的 API 金鑰與 API base,改用 vLLM 的 API 伺服器
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="San Francisco is a",
)
print("Completion result:", completion)
```

</details>

更詳細的客戶端範例請見:[examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

### 使用 vLLM 搭配 OpenAI Chat Completions API

vLLM 的設計也支援 OpenAI Chat Completions API。聊天介面是一種更動態、互動式的模型溝通方式,允許來回多輪對話並將其保存在聊天記錄中。這對於需要脈絡(context)或更詳細說明的任務相當實用。

你可以使用[建立聊天回應](https://platform.openai.com/docs/api-reference/chat/completions/create)端點來與模型互動:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

你也可以改用 `openai` 這個 Python 套件:

<details><summary>展開程式碼</summary>

```python
from openai import OpenAI
# 設定 OpenAI 的 API 金鑰與 API base,改用 vLLM 的 API 伺服器
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
)
print("Chat response:", chat_response)
```

</details>

## 關於 Attention 後端 (On Attention Backends)

目前,vLLM 針對不同平台與加速器架構,支援多種高效率 Attention 運算後端。它會自動選擇與你的系統及模型規格相容、且效能最佳的後端。

如果需要,你也可以透過 `--attention-backend` 命令列參數手動指定要使用的後端:

```bash
# 用於線上服務
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# 用於離線推論
python script.py --attention-backend FLASHINFER
```

部分可用的後端選項包括:

- 在 NVIDIA CUDA 上:`FLASH_ATTN` 或 `FLASHINFER`。
- 在 AMD ROCm 上:`TRITON_ATTN`、`ROCM_ATTN`、`ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN`、`TRITON_MLA`、`ROCM_AITER_MLA` 或 `ROCM_AITER_TRITON_MLA`。
- 在 Intel XPU 上:`FLASH_ATTN`、`TRITON_ATTN`、`TRITON_MLA`、`XPU_MLA_SPARSE`、`TORCH_SDPA` 或 `TURBOQUANT`。

> **警告**
> vLLM 並未內建包含 Flash Infer 的預先建置 wheel 檔案,因此你必須自行在環境中安裝它。請參考 [Flash Infer 官方文件](https://docs.flashinfer.ai/),或參考 [docker/Dockerfile](../../docker/Dockerfile) 中的安裝說明。
