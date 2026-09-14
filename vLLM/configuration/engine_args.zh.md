# Engine Arguments(啟動參數)— 實用子集

> 這不是官方文件的完整翻譯。官方原頁面是從 vLLM 原始碼**自動生成**的參數參考,目前版本涵蓋約 200-250 個參數、超過 2000 行,而且會隨每次 vLLM 版本更新變動。
> 這裡只挑對**單張 GPU、本機部署**情境實用的參數子集翻譯,其餘保留英文,直接查官方頁面即可。
> 官方原文:[docs.vllm.ai — Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
> 對照的實測環境見:[../../quickstart.md](../../quickstart.md)

## 這是什麼

Engine arguments 控制 vLLM 引擎的行為:

- 做[離線推論](https://docs.vllm.ai/en/latest/serving/offline_inference.html)時,它們是 `LLM` 類別建構子的參數
- 做[線上服務](https://docs.vllm.ai/en/latest/serving/online_serving/)時,它們是 `vllm serve` 指令的命令列參數

這些參數類別(`EngineArgs`、`AsyncEngineArgs`)實際上是 `vllm.config` 裡多個設定類別(`ModelConfig`、`CacheConfig`、`SchedulerConfig` 等)組合而成。

## GPU 記憶體與 KV Cache(`CacheConfig`)

#### `--gpu-memory-utilization`
**型別:** float(0 到 1 之間)**預設值:** `0.92`
GPU 記憶體要撥多少比例給模型執行器使用。例如設 `0.5` 代表用 50% 的 GPU 記憶體。這台機器(16GB VRAM)如果要同時開其他程式佔用顯存,可以調低這個值,避免 vLLM 把記憶體佔滿導致其他程式 OOM。

#### `--kv-cache-dtype`
**型別:** 選項(`auto`、`bfloat16`、`float16`、`fp8` 系列等)**預設值:** `auto`
KV cache 儲存用的資料型別。想省顯存、能接受一點精度損失時可以改用 `fp8` 系列,讓同樣的 VRAM 能塞下更長的上下文或更大的 batch。

#### `--enable-prefix-caching`
**型別:** 布林開關
是否啟用 prefix caching(前綴快取)。多輪對話、system prompt 重複出現的場景開這個能省下重複計算,大幅提升吞吐量。詳細原理見[官方 design 文件](https://docs.vllm.ai/en/latest/design/prefix_caching/)。

#### `--block-size`
**型別:** int 或 `None`
KV cache 每個 block 包含幾個 token。一般不用手動調,`None` 代表用預設值。

## 模型載入與序列長度

#### `--max-model-len`
**型別:** 整數(支援 `32k`、`1m` 這種易讀寫法)**預設值:** 沒指定的話,自動從模型設定推算
模型的最大上下文長度(prompt + 輸出加起來)。這個值設太大,`gpu-memory-utilization` 分配到的 KV cache 空間就要對應撐得住,兩個參數要一起看,不然可能啟動時就報記憶體不足。

#### `--dtype`
**型別:** 選項(`auto`、`bfloat16`、`float16`、`float32` 等)**預設值:** `auto`
模型權重與運算用的資料型別。`auto` 會依模型原本存的精度自動選(FP32/FP16 模型用 FP16,BF16 模型用 BF16)。我們在 `run_server.sh` 沒特別指定,就是吃這個預設值。

#### `--trust-remote-code`
**型別:** 布林開關 **預設值:** `False`
下載模型/tokenizer 時,是否信任 Hugging Face 上模型倉庫自帶的自訂程式碼。用不熟悉來源的模型時要注意這個等於允許執行對方的程式碼。

#### `--load-format`
**型別:** 選項 **預設值:** `auto`
模型權重的載入格式。`auto` 會優先找 `safetensors` 格式,找不到才 fallback 用 pytorch 的 `.bin` 格式。

## 排程與批次(`SchedulerConfig`)

#### `--max-num-seqs`
**型別:** 整數
單一次迭代最多同時處理幾個請求(序列)。這個數字愈大,併發量愈高,但每個請求分到的運算資源相對變少,通常會影響單一請求的延遲。

#### `--max-num-batched-tokens`
**型別:** 整數(支援易讀寫法)
單一次迭代最多能處理的 token 總數(所有正在跑的請求加起來)。這個值連動著顯存使用量與吞吐量,是效能調校時最常碰的參數之一。

#### `--enable-chunked-prefill`
**型別:** 布林開關 **預設值:** `False`
是否允許把很長的 prompt(prefill 階段)拆成好幾塊,依照剩餘的 `max_num_batched_tokens` 額度分批處理,而不是一次全部塞進去。長 prompt 的場景開這個能讓短請求不用一直等長請求的 prefill 跑完。

## 編譯 / Eager 模式

#### `--enforce-eager`
**型別:** 布林開關 **預設值:** `False`
是否強制永遠使用 PyTorch 的 eager 模式,關閉 `torch.compile` 與 CUDA Graph 編譯優化。

> 這台機器(Blackwell RTX 5080)踩過的坑就跟這個參數有關:一開始因為 WSL 沒裝 `build-essential` 缺 C 編譯器,`torch.compile` 會直接報錯,當時用 `enforce_eager=True` 繞過去(見 [quickstart.md](../../quickstart.md) 的踩坑對照表)。裝好 `build-essential` 之後,`vllm serve` 預設(`enforce_eager=False`)的完整編譯路徑其實也能正常跑完,只是第一次啟動因為要編譯+預熱多種 batch size 的 CUDA Graph,會明顯比較慢(我們實測約 6 分鐘,其中大半是下載模型 + 編譯耗掉的)。

## 量化

#### `--quantization`
**型別:** 選項 **預設值:** `None`
指定權重量化方式。設成 `None` 時,vLLM 會先檢查模型設定檔裡有沒有 `quantization_config` 屬性,自動判斷要不要量化、用哪種格式。各種量化格式(AWQ / GPTQ / FP8 / GGUF / NVFP4...)的細節不在這裡展開,是下一個學習主題,到時候會另外開一篇。

## 服務相關

#### `--served-model-name`
**型別:** 字串(可多個)**預設值:** 跟 `--model` 參數相同
API 對外回應時使用的模型名稱。可以給好幾個名字,伺服器會接受其中任何一個名字的請求,方便 API 相容性或改名不動用戶端程式碼。

## 還有更多參數

以上只是 200 多個參數裡,對「單 GPU 本機部署」這個情境比較常用到的一小部分。完整清單(包含分散式部署的 `ParallelConfig`、多模態的 `MultiModalConfig`、LoRA 的 `LoRAConfig`、可觀測性的 `ObservabilityConfig` 等)請直接查官方頁面:[docs.vllm.ai/en/latest/configuration/engine_args](https://docs.vllm.ai/en/latest/configuration/engine_args/)。之後學到分散式部署、LoRA 等主題時,會再回來從裡面挑相關子集補充翻譯。
