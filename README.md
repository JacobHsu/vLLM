# vLLM 學習筆記

個人學習 [vLLM](https://github.com/vllm-project/vllm) 的筆記與實測紀錄,內容基於官方 Quickstart 文件,並在自己的機器(Windows 11 + WSL2 + NVIDIA RTX 5080)上實際安裝、執行過。

來源文件:[docs.vllm.ai — Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)

## 目錄內容

| 檔案 | 說明 |
|---|---|
| [學習計畫.md](學習計畫.md) | 學習進度地圖,追蹤已學/待學的主題 |
| [quickstart.md](quickstart.md) | **本機實測紀錄**:只保留這台機器實際用到的 NVIDIA CUDA 路徑,記錄真正執行過的指令、遇到的錯誤,以及對應的解法(尤其是 RTX 50 系列 Blackwell 顯卡在 WSL2 上的相容性問題) |
| [scripts/](scripts/) | 實測用的腳本,已內建修正過的環境變數設定,可直接執行 |
| [vLLM/](vLLM/) | 官方文件逐篇繁體中文翻譯 |
| ├─ [quickstart.zh.md](vLLM/quickstart.zh.md) | 官方 Quickstart 文件的完整翻譯,涵蓋所有平台(NVIDIA CUDA / AMD ROCm / Intel GPU / Google TPU / Ascend NPU / Apple Silicon) |
| ├─ [configuration/engine_args.zh.md](vLLM/configuration/engine_args.zh.md) | Engine Arguments(啟動參數)實用子集翻譯,官方頁面自動生成、規模達 200+ 參數,只挑單 GPU 本機部署會用到的 |
| ├─ [features/quantization.zh.md](vLLM/features/quantization.zh.md) | Quantization(量化)索引頁翻譯,含硬體相容表;已實測驗證 Blackwell 相容性(見下) |
| ├─ [features/structured_outputs.zh.md](vLLM/features/structured_outputs.zh.md) | Structured Outputs(結構化輸出)完整翻譯:`choice`/`regex`/`json`/`grammar` 幾種模式,含線上服務、離線推論、搭配 reasoning 的用法 |
| └─ [features/tool_calling.zh.md](vLLM/features/tool_calling.zh.md) | Tool Calling(工具呼叫)完整翻譯:含 20+ 家模型專屬 parser 對照表,Qwen 系列適用 `hermes` parser |
| [quantization-benchmark.md](quantization-benchmark.md) | **量化實測紀錄**:1.5B / 7B(bf16)/ 7B(FP8)/ 8B(NVFP4)四個模型的實測對照,含 tok/s、VRAM、能否啟動;NVFP4 在這張卡上目前啟動失敗(上游 SM120 支援未完成) |
| [.claude/skills/llm-eval-harness/](.claude/skills/llm-eval-harness/) | 從 [daymade/claude-code-skills](https://github.com/daymade/claude-code-skills)(MIT)複製的 skill,用來測 OpenAI-compatible 端點的可用性/速度/併發/品質 |

## 快速開始

這台機器的完整安裝與踩坑記錄請直接看 [quickstart.md](quickstart.md)。重點結論:

- 環境:Windows 11 + WSL2(Ubuntu)+ NVIDIA GPU,用 `uv` 建立 Python 環境並安裝 vLLM
- vLLM **不支援原生 Windows**,所有指令都要在 WSL2 的 Linux shell 裡執行
- 在 Blackwell(RTX 50 系列,sm_120)顯卡上會遇到 3 個已知相容性問題,已在腳本中修好,細節見 quickstart.md 的對照表

## 授權與版權

本筆記內容參考自 vLLM 官方文件([Apache-2.0 授權](https://github.com/vllm-project/vllm/blob/main/LICENSE)),翻譯與實測記錄僅供個人學習使用。
