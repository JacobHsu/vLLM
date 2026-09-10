# vLLM 學習筆記

個人學習 [vLLM](https://github.com/vllm-project/vllm) 的筆記與實測紀錄,內容基於官方 Quickstart 文件,並在自己的機器(Windows 11 + WSL2 + NVIDIA RTX 5080)上實際安裝、執行過。

來源文件:[docs.vllm.ai — Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)

## 目錄內容

| 檔案 | 說明 |
|---|---|
| [vllm-quickstart-zh-tw.md](vllm-quickstart-zh-tw.md) | 官方 Quickstart 文件的完整繁體中文翻譯,涵蓋所有平台(NVIDIA CUDA / AMD ROCm / Intel GPU / Google TPU / Ascend NPU / Apple Silicon) |
| [quickstart.md](quickstart.md) | **本機實測紀錄**:只保留這台機器實際用到的 NVIDIA CUDA 路徑,記錄真正執行過的指令、遇到的錯誤,以及對應的解法(尤其是 RTX 50 系列 Blackwell 顯卡在 WSL2 上的相容性問題) |
| [scripts/](scripts/) | 實測用的腳本,已內建修正過的環境變數設定,可直接執行 |

## 快速開始

這台機器的完整安裝與踩坑記錄請直接看 [quickstart.md](quickstart.md)。重點結論:

- 環境:Windows 11 + WSL2(Ubuntu)+ NVIDIA GPU,用 `uv` 建立 Python 環境並安裝 vLLM
- vLLM **不支援原生 Windows**,所有指令都要在 WSL2 的 Linux shell 裡執行
- 在 Blackwell(RTX 50 系列,sm_120)顯卡上會遇到 3 個已知相容性問題,已在腳本中修好,細節見 quickstart.md 的對照表

## 授權與版權

本筆記內容參考自 vLLM 官方文件([Apache-2.0 授權](https://github.com/vllm-project/vllm/blob/main/LICENSE)),翻譯與實測記錄僅供個人學習使用。
