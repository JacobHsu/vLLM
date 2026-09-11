# 量化實測記錄(RTX 5080 Blackwell)

> 驗證 [vLLM/features/quantization.zh.md](vLLM/features/quantization.zh.md) 裡提出的疑慮:官方硬體相容表沒列 Blackwell,量化到底能不能用、值不值得用。
> 環境同 [quickstart.md](quickstart.md):WSL2 Ubuntu + RTX 5080 Laptop(16GB VRAM,sm_120)。
> 用到的腳本:[scripts/run_server.sh](scripts/run_server.sh)(1.5B 基準)、[scripts/run_server_7b_bf16.sh](scripts/run_server_7b_bf16.sh)(7B 未量化)、[scripts/run_server_fp8.sh](scripts/run_server_fp8.sh)(7B FP8 量化)、[scripts/wait_for_server.sh](scripts/wait_for_server.sh)(輪詢伺服器是否就緒的小工具)。

## 測試對象

| 模型 | 參數量 | 精度 | 來源 |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | bf16(未量化) | 官方原始模型 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | bf16(未量化) | 官方原始模型 |
| `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` | 7B | FP8(llm-compressor W8A8) | RedHat AI(LLM Compressor 維護方)在 Hugging Face 發布的量化版 |

## 結論先講:量化在這張卡上是「必需品」,不是「錦上添花」

**7B 模型不量化,直接在這張 16GB 顯卡上跑不起來**,錯誤訊息:

```
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
when initializing the engine...
```

白話翻譯:模型權重本身(bf16,約 15GB)幾乎把 16GB 顯存吃光,連 KV cache(推論過程中一定要用的暫存空間)都分不到半點空間,伺服器直接拒絕啟動,連一個請求都處理不了。實測時它甚至撐到把模型完整下載、載入 GPU(GPU 顯存用量一路衝到 15.3GB+),才在配置 KV cache 的最後一步失敗。

**換成 FP8 量化版,同一張卡直接跑起來,而且能正常回答問題。**

## 三個模型的實測數據對照

| 指標 | 1.5B(bf16) | 7B(bf16,未量化) | 7B(FP8 量化) |
|---|---|---|---|
| 能否啟動 | ✅ 正常 | ❌ `No available memory for the cache blocks` | ✅ 正常 |
| 權重 + 額外開銷佔用顯存 | 3.32 + 0.42 + 0.14 ≈ **3.9 GiB** | 權重本身就逼近 16GB 上限,無法完成 | 9.55 + 1.35 + 0.14 ≈ **11.0 GiB** |
| 可用 KV cache 顯存 | **10.9 GiB** | 0(這就是失敗原因) | **3.75 GiB** |
| KV cache 容量(token 數) | 408,144 tokens | — | 未印出(啟動階段更早) |
| 32K 上下文下的最大併發數 | **12.46x** | — | **2.14x** |
| 持續解碼速度(sustained decode) | **171.3 tok/s** | — | **77.2 tok/s** |
| 首字延遲(TTFT) | 0.013–0.018s | — | 0.021–0.035s |

(速度數據用已裝好的 [llm-eval-harness](.claude/skills/llm-eval-harness/) 的 `speed_probe.py` 實測,`--mode both`,取 3 輪 decode 的 peak/avg;完整原始輸出見下方「實測指令」。)

## 怎麼解讀這張表

- **1.5B 跟 7B(FP8)沒有直接的「誰比較好」關係**——它們是不同量級的模型。1.5B 天生小、天生快,7B 懂得更多、更聰明,但天生就要吃更多資源。量化讓「更聰明的 7B」有機會塞進「本來只夠養活 1.5B」的 16GB 顯存裡,這才是這次實測真正要驗證的事。
- 7B(FP8)的持續解碼速度(77.2 tok/s)大約是 1.5B 的 45%,但參數量是 1.5B 的 4.7 倍——換算下來,FP8 量化讓「每多一倍參數換來的減速幅度」明顯小於線性(不量化的話,7B 权重都塞不下,更別談速度)。
- 7B(FP8)的最大併發數只有 2.14x(對比 1.5B 的 12.46x),代表可用的 KV cache 空間相對緊繃——如果之後要撐多人同時使用,7B 這個量級在 16GB 卡上還是偏緊,需要搭配 [engine_args.zh.md](vLLM/configuration/engine_args.zh.md) 裡的 `--max-model-len`(縮短上下文)或 `--gpu-memory-utilization` 去擠出更多 KV cache 空間。
- **FP8 沒有踩到任何 Blackwell 專屬的相容性問題**——這點跟 quickstart.md 記錄的「Blackwell 生態系還在補齊」形成對比:至少 FP8(llm-compressor W8A8)這條路線,在這張卡、這個 vLLM 版本(0.29.0)上是穩的。NVFP4(NVIDIA 這代主推的格式)還沒測,留待下一步。

## 實測指令

啟動 FP8 量化版:

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo   # 換成你自己 clone 這個 repo 的實際路徑
bash "$REPO_DIR/scripts/run_server_fp8.sh"
```

啟動未量化 7B(會失敗,這是預期中的失敗,用來對照):

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo
bash "$REPO_DIR/scripts/run_server_7b_bf16.sh"
```

用 llm-eval-harness 測速度(伺服器要先跑起來,`--model` 換成對應的 `--served-model-name`):

```bash
cd ~/vllm-quickstart && source .venv/bin/activate
export DUMMY_KEY=EMPTY   # 本機伺服器沒設 API key,隨便給一個環境變數名稱即可
uv run --with openai python "$REPO_DIR/.claude/skills/llm-eval-harness/scripts/speed_probe.py" \
  --base-url http://localhost:8000/v1 --model qwen2.5-7b-fp8 --key-env DUMMY_KEY --mode both
```

## 下一步

- 找 NVFP4 量化版的模型測一次,對照 FP8 的結果(NVFP4 是 Blackwell 這代主推的格式,理論上該更快、更省)
- 用 `llm-eval-harness` 的 concurrency probe 實際測出這張卡在 FP8 7B 下的併發上限,而不是只看 vLLM 啟動時算出的理論值(2.14x)
