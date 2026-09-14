# Quantization(量化)

> 官方原文:[docs.vllm.ai — Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
> 本文翻譯了官方索引頁的核心內容(支援格式清單、硬體相容表)。頁尾「Out-of-Tree Quantization Plugins」是給要自己開發新量化方法的人看的開發者文件,不是部署使用者會用到的,這裡略過,有需要再查原文。
> 各量化格式各自的細節頁面(AutoAWQ、GPTQModel、LLM Compressor 等)是獨立文件,尚未翻譯,見文末連結。

## 這是什麼

量化(Quantization)是用模型精度換取更小的記憶體佔用,讓大型模型能在更多種裝置上跑起來。

> **提示**
> 如果不知道從哪裡開始,官方建議先看 [LLM Compressor](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/README.md)——一個用來把模型最佳化成適合 vLLM 部署格式的函式庫,支援 FP8、INT8、INT4 等多種量化格式。

vLLM 支援的量化格式:

- [AutoAWQ](https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html)
- [BitsAndBytes](https://docs.vllm.ai/en/latest/features/quantization/bnb.html)
- [GPTQModel](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel.html)
- [Intel Neural Compressor](https://docs.vllm.ai/en/latest/features/quantization/inc.html)
- [LLM Compressor](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/README.html)
    - FP8 W8A8
    - INT4 W4A16
    - INT8 W4A8
    - INT8 W8A8
- [NVIDIA Model Optimizer](https://docs.vllm.ai/en/latest/features/quantization/modelopt.html)
- [Online Quantization](https://docs.vllm.ai/en/latest/features/quantization/online.html)(執行期即時量化,不用事先轉檔)
- [AMD Quark](https://docs.vllm.ai/en/latest/features/quantization/quark.html)
- [Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)(對應 [engine_args.zh.md](../configuration/engine_args.zh.md) 裡的 `--kv-cache-dtype`)
- [TorchAO](https://docs.vllm.ai/en/latest/features/quantization/torchao.html)
- [FP8 ViT Encoder Attention](https://docs.vllm.ai/en/latest/features/quantization/fp8_vit_attn.html)

## 支援的硬體

官方硬體相容表(原文逐字翻譯,只翻欄位標題與註解,方法名稱與符號保留原樣):

| 量化實作方式 | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Marlin(GPTQ/AWQ/FP8/FP4) | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| llm-compressor INT8(W8A8) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| llm-compressor INT8(W4A8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| llm-compressor FP8(W8A8) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| bitsandbytes | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| DeepSpeedFP | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

- Volta 對應 SM 7.0、Turing 對應 SM 7.5、Ampere 對應 SM 8.0/8.6、Ada 對應 SM 8.9、Hopper 對應 SM 9.0。
- ✅ 代表該硬體支援這個量化方法;❌ 代表不支援。
- Intel Gaudi 的量化支援已經全部搬到 [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi) 專案。
- \*Turing 不支援 Marlin MXFP4。

> **⚠️ 這台機器(RTX 5080,Blackwell,SM 12.0)的重要提醒**
> 上面這張官方相容表**完全沒有列出 Blackwell**(SM 10.0/12.0)這一代顯卡,最新只列到 Hopper(SM 9.0)。這不代表 Blackwell 不能用量化,而是文件更新速度跟不上硬體世代——跟我們在 [quickstart.md](../../quickstart.md) 裡記錄的狀況一致(FlashInfer、cutlass kernel 等好幾個元件對 sm_120 的支援都還在補齊中)。
>
> 實務上,Blackwell 要吃的是 **NVFP4**(NVIDIA 這代主推的 4-bit 浮點量化格式),走的是 [NVIDIA Model Optimizer](https://docs.vllm.ai/en/latest/features/quantization/modelopt.html) 或 llm-compressor 的路線,不在上面這張(舊版)表格涵蓋範圍內。
>
> **已經親自測過 FP8 跟 NVFP4 這兩條路線,結果是「FP8 能用、NVFP4 目前不能用」**——詳見 [quantization-benchmark.md](../../quantization-benchmark.md):
> - FP8(llm-compressor W8A8):7B 模型不量化直接在這張 16GB 卡上啟動失敗(`No available memory for the cache blocks`),換成 FP8 量化版就正常跑起來,沒有遇到任何 Blackwell 專屬的相容性問題。
> - NVFP4:啟動失敗,原因是 vLLM/FlashInfer 對 SM120(消費級 Blackwell)的 NVFP4 GEMM kernel 支援目前確實還沒做完(有多個上游 issue 追蹤,不是設定錯誤)。

> **備註**
> Google TPU 上的量化支援,請參考 [TPU-Inference 支援的模型與功能](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) 文件。

> **備註**
> 這張相容表會隨 vLLM 版本持續更新,最新資訊請直接查 vLLM 原始碼裡的 [`vllm/model_executor/layers/quantization`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization) 目錄。

## 開發者專屬內容(略過)

官方文件後半段說明如何用 `@register_quantization_config` 裝飾器,自己註冊一個「域外」(out-of-tree)的自訂量化方法(繼承 `QuantizationConfig`、實作 `get_quant_method` 等抽象方法)。這是給要**開發新量化演算法**的人看的,對「用現成量化格式跑模型」的部署情境用不到,這裡不展開,有需要直接看[官方原文](https://docs.vllm.ai/en/latest/features/quantization/#out-of-tree-quantization-plugins)。

## 下一步

這篇只翻了索引頁,還沒細看個別量化格式的實際操作步驟(例如怎麼把一個 Hugging Face 模型轉成 AWQ 格式、`--quantization` 參數該填什麼值)。之後真的要在這台機器上跑量化模型時,會挑跟 Blackwell 最相關的 [NVIDIA Model Optimizer](https://docs.vllm.ai/en/latest/features/quantization/modelopt.html) 或 [LLM Compressor FP8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8.html) 回來補一篇實測記錄。
