# Tool Calling 本機實測記錄

> 對照 [vLLM/features/tool_calling.zh.md](vLLM/features/tool_calling.zh.md) 裡的 `get_weather` 快速開始範例,在這台機器上實際跑一次。
> 環境同 [quickstart.md](quickstart.md):WSL2 + RTX 5080,用 `Qwen/Qwen2.5-1.5B-Instruct`(未量化)測試——Qwen2.5 系列的聊天範本已內建 Hermes 風格 tool use,所以直接用 `hermes` parser,不用額外找/寫聊天範本。
> 用到的腳本:[scripts/run_server_tool_calling.sh](scripts/run_server_tool_calling.sh)、[scripts/test_tool_calling.py](scripts/test_tool_calling.py)

## 結果:一次就成功,跟官方文件範例輸出一致

啟動指令:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

丟一句「舊金山天氣如何?」,搭配定義好的 `get_weather` 工具,`tool_choice="auto"`:

```
=== raw message ===
ChatCompletionMessage(content=None, ..., tool_calls=[ChatCompletionMessageFunctionToolCall(
    id='chatcmpl-tool-8d0a6d47d4d1219c',
    function=Function(arguments='{"location": "San Francisco, CA", "unit": "fahrenheit"}', name='get_weather'),
    type='function')], ...)

=== parsed ===
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

模型正確判斷「這個問題該呼叫 `get_weather` 工具」,參數 `location`/`unit` 也都符合定義好的 schema(`unit` 剛好落在 `["celsius", "fahrenheit"]` 這個 enum 裡)。跟官方文件範例輸出完全一致,沒有遇到任何 Blackwell 相關問題。

## 為什麼這個功能也沒踩到 Blackwell 的坑

跟 structured outputs 是同一類——tool calling 的底層機制其實就是「用 structured outputs 限制模型輸出符合工具參數的 JSON schema」,加上一個 parser(`hermes`)把生成的文字解析成結構化的 `tool_calls` 物件。同樣不需要 GPU 專屬 kernel,是 CPU 端的取樣限制 + 文字解析邏輯,跟量化(需要專屬矩陣運算硬體)是完全不同的技術路線。

## 實測指令

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo   # 換成你自己 clone 這個 repo 的實際路徑
bash "$REPO_DIR/scripts/run_server_tool_calling.sh"
```

另開一個 shell:

```bash
cd ~/vllm-quickstart && source .venv/bin/activate
python "$REPO_DIR/scripts/test_tool_calling.py"
```

## 下一步

- 這次只測了單一工具、單一輪對話;平行工具呼叫(一次判斷要呼叫多個工具)、多輪對話裡把工具執行結果餵回去給模型,還沒測
