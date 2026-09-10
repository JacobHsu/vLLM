# vLLM 快速入門 — 本機實測記錄

> 這份文件不是官方文件的完整翻譯,而是**在這台機器上實際操作、只保留真正用到的步驟**的紀錄,包含中途遇到的錯誤與實際解法。
> 完整翻譯版請見:[vllm-quickstart-zh-tw.md](vllm-quickstart-zh-tw.md)
> 官方原文:[docs.vllm.ai — Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
> 本文使用的所有腳本都存放在 [scripts/](scripts/) 目錄下。
>
> ⚠️ **以下所有指令都是在 WSL2 的 Ubuntu shell 裡執行,不是 Windows PowerShell/CMD。** vLLM 不支援原生 Windows,裝在 WSL 裡的 Python 環境,Windows 端的 `python`/`vllm` 指令都找不到、不能用。要先用 `wsl -d Ubuntu` 進入 Ubuntu,再執行下面的指令。
>
> 下面指令裡的 `REPO_DIR` 代表這個 repo 在 WSL 裡看到的路徑,依你自己 clone / 存放的位置設定(下面每個「實際執行」區塊都已經包含這行,複製整段貼上就會自動設定好,不用另外處理;但如果是自己手動重新打,記得每次開新的 WSL shell 都要重設一次,因為這只是當次 shell 的暫時變數):
>
> ```bash
> export REPO_DIR=/mnt/x/path/to/vllm-repo   # 換成你自己 clone 這個 repo 的實際路徑
> ```

## 本機環境

| 項目 | 實際值 |
|---|---|
| 主機 OS | Windows 11 Pro (build 26200) |
| 執行平台 | WSL2 — Ubuntu 26.04 LTS(kernel 6.18.33.2-microsoft-standard-WSL2) |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU,16GB VRAM(Blackwell,**sm_120**) |
| 驅動 | 610.74(CUDA UMD 13.3) |
| CPU / RAM | Intel Core Ultra 9 290HX,63GB RAM |
| Python | 3.12.14(由 uv 建立獨立環境,不用系統內建的 3.14) |
| 安裝路徑 | `~/vllm-quickstart`(WSL 原生檔案系統,非 `/mnt/d`,避免 DrvFs 效能損耗) |
| 最終版本 | `torch==2.13.0+cu132`、`vllm==0.29.0` |

因為 GPU 是 NVIDIA,以下只走官方文件中的 **NVIDIA CUDA** 安裝路徑,其餘平台(AMD ROCm / Intel GPU / Google TPU / Ascend NPU / Apple Silicon)略過不記錄。

## 安裝步驟(實際執行,含遇到的坑)

### 1. 安裝 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> 實測備註:直接 `curl | sh` 一次失敗於 `Could not resolve host: releases.astral.sh`(WSL 內 DNS 偶發性問題),重試即可,非設定錯誤。

### 2. 建立虛擬環境

```bash
mkdir -p ~/vllm-quickstart && cd ~/vllm-quickstart
uv venv --python 3.12 --seed
source .venv/bin/activate
```

結果:uv 自動下載 `cpython-3.12.14-linux-x86_64-gnu`(32.6MiB)並建立 `.venv`。

### 3. 安裝 vLLM

```bash
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` 會偵測本機 CUDA 驅動版本並自動選對應的 PyTorch/CUDA wheel。實測中 uv 偵測到驅動支援 CUDA 13,自動選用 `nvidia-*-cu13` 系列套件(對 Blackwell sm_120 是正確選擇),共解析 196 個套件,下載約數 GB。安裝完成後確認:

```
torch: 2.13.0+cu132
vllm: 0.29.0
cuda available: True
device: NVIDIA GeForce RTX 5080 Laptop GPU
capability: (12, 0)
```

### 4. 補裝 build-essential(官方文件沒提到,但這台機器一定要裝)

```bash
sudo apt update && sudo apt install -y build-essential
```

沒裝的話,後面 vLLM 啟動時的 Triton/JIT 編譯階段會直接報錯 `Failed to find C compiler`。這是 WSL Ubuntu 預設「最小安裝」不含編譯器,不是 vLLM 的問題,任何 WSL 上要跑 vLLM 的人幾乎都得補這一步。

## Blackwell(RTX 50 系列 / sm_120)在 WSL2 上實測遇到的 3 個問題

這張卡是最新的 Blackwell 架構,在 vLLM 生態系裡仍是「新戰場」,實測從零到能跑,依序踩了以下 3 個坑,**都已在 [scripts/run_server.sh](scripts/run_server.sh) 與 [scripts/offline_inference_test.py](scripts/offline_inference_test.py) 中修好**:

| # | 錯誤訊息 | 原因 | 解法 |
|---|---|---|---|
| 1 | `RuntimeError: UVA is not available` | vLLM 新版 model runner 需要 Unified Virtual Addressing,WSL2 預設不會開啟 | 設定環境變數 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`(需要 WSL kernel ≥ 4.19.121,這台是 6.18,足夠) |
| 2 | `torch._inductor.exc.InductorError: ... Failed to find C compiler` | WSL 沒裝 `build-essential` | `sudo apt install -y build-essential`(見上) |
| 3 | `RuntimeError: CUDA compiler and CUDA toolkit headers are incompatible`(來自 FlashInfer JIT 編譯 sampling kernel) | FlashInfer 的 JIT sampler 在 Blackwell sm_120 + CUDA 13.3 這個組合下編譯會失敗,這是[已知上游 issue](https://github.com/vllm-project/vllm/issues/44305) | 設定環境變數 `VLLM_USE_FLASHINFER_SAMPLER=0`,改用 PyTorch 原生的 top-p/top-k 取樣實作(效能略低,但能正常運作) |

另外,pip 安裝的 vLLM 不會裝系統級 CUDA Toolkit,但 FlashInfer 某些 kernel 需要用 `nvcc` 現場編譯。解法是直接指向 pip 幫我們裝好的 `nvidia-cuda-nvcc` 套件路徑,不需要額外裝一整套 CUDA Toolkit:

```bash
export CUDA_HOME="$HOME/vllm-quickstart/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
```

## 離線批次推論(Offline Batched Inference)— 實測通過

腳本:[scripts/offline_inference_test.py](scripts/offline_inference_test.py)(在官方範例基礎上,加了上述環境變數設定,並補上 `if __name__ == "__main__":` — vLLM 的 engine 會另外 spawn 子行程,少了這個保護在 Linux/WSL 上會直接炸掉,官方文件的範例片段沒特別強調這點)

實際執行(先進入 WSL,啟用 venv,再用 `$REPO_DIR` 執行腳本):

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo  # x/path換成你自己 clone 這個 repo 的實際路徑
cd ~/vllm-quickstart
source .venv/bin/activate
python "$REPO_DIR/scripts/offline_inference_test.py"
```

> 上面三個 Blackwell 修正(`CUDA_HOME`、`VLLM_USE_FLASHINFER_SAMPLER`、`VLLM_WSL2_ENABLE_PIN_MEMORY`)都已經寫在腳本開頭自動設定,不用手動 export。

實際輸出:

```
Prompt: 'Hello, my name is', Generated text: " my birthday in October. I'm 25. I'm very fortunate to have a"
Prompt: 'The president of the United States is', Generated text: ' looking to change the rules for the travel industry, taking steps that will transform the'
Prompt: 'The capital of France is', Generated text: ' moving rapidly towards the French presidential election. Its capital was bombed twice by an Islamic'
Prompt: 'The future of AI is', Generated text: ' being hazed by a pandemic\nA new study suggests that the rise of'
```

(內容本身沒有意義——`facebook/opt-125m` 只是個 1.25 億參數的小模型,純粹用來驗證引擎能跑起來。)

## 線上服務(Online Serving)— 實測通過

腳本:[scripts/run_server.sh](scripts/run_server.sh)(內含 `cd`、啟用 venv、三個 Blackwell 修正環境變數,最後執行 `vllm serve Qwen/Qwen2.5-1.5B-Instruct`)

實際執行(一樣要先進入 WSL):

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo   # 換成你自己 clone 這個 repo 的實際路徑
bash "$REPO_DIR/scripts/run_server.sh"
```

實測備註:第一次啟動耗時約 6 分鐘,其中大半時間在下載模型權重(約 3GB)+ 預設的 `torch.compile` 對上百種 batch size 做 CUDA Graph 預先編譯(這次**沒有**加 `enforce_eager`,代表 build-essential 裝好之後,預設的完整編譯路徑在這張 Blackwell 卡上也能正常跑完,沒有再遇到問題)。

啟動後驗證:

```bash
curl http://localhost:8000/v1/models
```

```json
{"object":"list","data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct", ...}]}
```

### Completions API

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

實際輸出(節錄):`"text":" city in the state of California,"`

### Chat Completions API

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

實際輸出(節錄):`"content":"The New York Yankees won the World Series in 2020, defeating the Houston Astros in seven games."`

> 這個答案其實是錯的(2020 年世界大賽冠軍是道奇隊),純粹是 1.5B 小模型的知識能力限制,跟環境設定無關,只是如實記錄實測結果。

測試完成後關閉伺服器:

```bash
pkill -f "vllm serve"
```

## 結論

這台機器(RTX 5080 Laptop + WSL2 Ubuntu)**確認可以完整跑通 vLLM 的離線推論與線上服務**。唯一需要額外處理的是 Blackwell(sm_120)在目前生態系裡的三個已知相容性問題,整理成一份環境變數清單,之後每次開新的 shell 要跑 vLLM 時都需要設定(已寫入 [scripts/run_server.sh](scripts/run_server.sh)):

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$HOME/vllm-quickstart/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
```
