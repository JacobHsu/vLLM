# Tool Calling(工具呼叫)

> 官方原文:[docs.vllm.ai — Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
> 結尾「How to Write a Tool Parser Plugin」是給要自己開發新 tool parser 的人看的開發者文件,略過不翻,有需要查[官方原文](https://docs.vllm.ai/en/latest/features/tool_calling/#how-to-write-a-tool-parser-plugin)。
> 範例假設已經有 vLLM 伺服器在跑,可參考 [quickstart.md](../../quickstart.md)。

vLLM 目前支援具名函式呼叫(named function calling),以及聊天補全(chat completion)API 裡 `tool_choice` 欄位的 `auto`、`required`(`vllm>=0.8.3` 起)、`none` 選項。

## 快速開始

啟動伺服器時開啟 tool calling。這個範例用 Meta 的 Llama 3.1 8B 模型,所以要指定 vLLM 範例目錄裡的 `llama3_json` tool calling 聊天範本:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

接著發一個會觸發模型使用工具的請求:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."
tool_functions = {"get_weather": get_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"],
            },
        },
    },
]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
```

範例輸出:

```text
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

這個範例示範了:

* 啟動伺服器時開啟 tool calling
* 定義一個實際處理工具呼叫的函式
* 用 `tool_choice="auto"` 發送請求
* 處理結構化回應並執行對應的函式

也可以用具名函式呼叫(named function calling)指定特定函式,設定 `tool_choice={"type": "function", "function": {"name": "get_weather"}}`。注意這會用到結構化輸出後端——所以第一次使用時,因為要編譯 FSM(有限狀態機),會有幾秒鐘(或更久)的延遲,編譯完之後會被快取,之後的請求就不會再有這個延遲。

呼叫端(caller)有責任要做到:

1. 在請求裡定義好合適的工具
2. 在聊天訊息裡放進相關的上下文
3. 在應用程式邏輯裡處理工具呼叫

更進階的用法(包含平行工具呼叫、各家模型專屬的 parser)見下面各節。

## 具名函式呼叫(Named Function Calling)

vLLM 預設就支援聊天補全 API 裡的具名函式呼叫,大多數 vLLM 支援的結構化輸出後端都能搭配使用。這種方式保證回傳的函式呼叫**格式一定合法可解析**,但不保證內容**品質一定好**。

vLLM 會用結構化輸出,確保回應符合 `tools` 參數裡 JSON schema 定義的工具參數物件。為了得到最好的結果,建議在 prompt 裡也說明預期的輸出格式/schema,確保模型原本想生成的內容,跟結構化輸出後端強制它生成的 schema 是一致的。

要使用具名函式,你需要在聊天補全請求的 `tools` 參數裡定義函式,並在 `tool_choice` 參數裡指定其中一個工具的 `name`。

## 強制函式呼叫(Required Function Calling)

vLLM 支援聊天補全 API 裡的 `tool_choice='required'` 選項。跟具名函式呼叫類似,這也是用結構化輸出實現的,所以預設就會啟用,任何支援的模型都能用。不過其他解碼後端的支援,列在 V1 引擎的[路線圖](https://docs.vllm.ai/en/latest/usage/v1_guide.html#features)上。

設定 `tool_choice='required'` 時,模型保證會根據 `tools` 參數裡指定的工具清單,生成一個以上的工具呼叫。工具呼叫的數量取決於使用者的查詢內容,輸出格式會嚴格遵守 `tools` 參數定義的 schema。

## 不呼叫工具(None Function Calling)

vLLM 支援聊天補全 API 裡的 `tool_choice='none'` 選項。設定這個選項時,即使請求裡有定義工具,模型也不會生成任何工具呼叫,只會用一般文字內容回應。

> **備註**
> 當請求裡有指定工具時,vLLM 預設會把工具定義放進 prompt 裡,不管 `tool_choice` 設什麼。如果想在 `tool_choice='none'` 時排除工具定義,可以用 `--exclude-tools-when-tool-choice-none` 選項。

## 限制解碼(Constrained Decoding)行為

vLLM 在生成時會不會強制套用工具參數的 schema,取決於 `tool_choice` 模式,以及每個工具各自的 `strict` 欄位:

| `tool_choice` 值 | Schema 限制解碼 | 行為 |
| --- | --- | --- |
| 具名函式 | 是(透過結構化輸出後端) | 保證引數(arguments)是符合函式參數 schema 的合法 JSON |
| `"required"` | 是(透過結構化輸出後端) | 跟具名函式一樣,模型必須至少生成一個工具呼叫 |
| `"auto"` | 只有當至少一個工具設定 `strict: true` 時才會 | 當有工具選擇加入(opt-in)`strict: true` 時,structural-tag parser 會限制工具呼叫的引數;沒設的話,模型自由生成,工具呼叫是從原始文字裡抽取出來的 |
| `"none"` | 不適用 | 不會產生任何工具呼叫 |

### Strict 模式

對 `tool_choice="required"` 或具名函式呼叫來說,不管 `strict` 欄位設什麼,一律都會套用 structural-tag 限制。對 `tool_choice="auto"` 來說,只要至少一個工具設定 `strict: true`,就會加入 structural-tag 限制;沒設的話,模型自由生成,工具呼叫是從原始文字抽取出來的。`strict` 欄位在三種 API 介面(Chat Completion、Responses、Anthropic Messages)都支援。

為了讓 strict schema 限制的相容性最好,建議用 OpenAI 的 strict-schema 風格定義工具參數 schema:

* `parameters` 裡每個物件的 `additionalProperties` 都設成 `false`
* `properties` 裡所有欄位都標成 required
* 選填欄位改用允許 `null` 的方式表示,例如 `{"type": ["string", "null"]}`

vLLM 也提供一個全域開關 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 環境變數(預設 `true`)。設成 `false` 時,不管每個工具各自的 `strict` 欄位是什麼,vLLM 都不會附加 tool calling 用的 structural tag。這個環境變數只影響 structural-tag 型的 tool calling,不會影響具名函式呼叫或 `tool_choice="required"` 用到的、由 schema 衍生的結構化輸出。

```bash
VLLM_ENFORCE_STRICT_TOOL_CALLING=false vllm serve ...
```

## 自動函式呼叫(Automatic Function Calling)

要開啟這個功能,需要設定以下旗標:

* `--enable-auto-tool-choice` —— **必要**。告訴 vLLM 你要讓模型在它認為合適的時候自己生成工具呼叫。
* `--tool-call-parser` —— 選擇要用的 tool parser(列在下面),之後還會持續加入新的 parser。也可以透過 `--tool-parser-plugin` 註冊自己寫的 tool parser。
* `--tool-parser-plugin` —— **選填**。用來把使用者自訂的 tool parser 註冊進 vLLM 的外掛,註冊後的 parser 名稱可以填在 `--tool-call-parser` 裡。
* `--chat-template` —— **選填**,給自動 tool choice 用。這是聊天範本的路徑,用來處理 `tool` 角色的訊息,以及包含先前生成的工具呼叫的 `assistant` 角色訊息。Hermes、Mistral、Llama 模型的 `tokenizer_config.json` 裡已經有相容 tool 的聊天範本,但你也可以指定自訂範本。如果你的模型在 `tokenizer_config.json` 裡設定了專屬的 tool use 聊天範本,這個參數可以填 `tool_use`,這時會照 `transformers` 的規範使用該範本。詳情見 HuggingFace 的[說明](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates),範例可參考[這個 `tokenizer_config.json`](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)。

如果你喜歡的 tool-calling 模型還沒被支援,歡迎貢獻 parser 跟對應的 tool use 聊天範本!

> **備註**
> 當 `tool_choice="auto"` 時,schema 層級的限制需要同時滿足兩個條件:`VLLM_ENFORCE_STRICT_TOOL_CALLING=true`(預設值)、且至少一個工具設定 `strict: true`。滿足這兩個條件、且選用的 parser 支援 structural tag 時,vLLM 才會限制工具呼叫的引數。否則 vLLM 會從原始文字抽取工具呼叫,引數偶爾可能格式不對或不符合函式的參數 schema。

### Hermes 系列模型(`hermes`)

Nous Research Hermes 系列裡,比 Hermes 2 Pro 新的模型應該都支援。

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_注意:Hermes 2 **Theta** 模型因為建立過程中的合併(merge)步驟,已知工具呼叫的品質跟能力會下降。_

旗標:`--tool-call-parser hermes`

### Mistral 系列模型(`mistral`)

支援的模型:

* `mistralai/Mistral-7B-Instruct-v0.3`(已確認)
* 其他支援函式呼叫的 Mistral 模型應該也相容

已知問題:

1. Mistral 7B 不太會正確生成平行工具呼叫。
2. **僅限 Transformers tokenization 後端**:Mistral 的 `tokenizer_config.json` 聊天範本要求工具呼叫 ID 剛好是 9 位數,比 vLLM 生成的短很多。因為不符合這個條件會丟例外,所以另外提供了下面這些聊天範本:

    * [examples/tool_chat_template_mistral.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_mistral.jinja) —— 這是「官方」的 Mistral 聊天範本,但調整過讓它能跟 vLLM 的工具呼叫 ID 搭配使用(`tool_call_id` 欄位會被截斷成最後 9 位數)
    * [examples/tool_chat_template_mistral_parallel.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_mistral_parallel.jinja) —— 這是「更好」的版本,有提供工具時會加上一段 tool-use 的 system prompt,在平行工具呼叫上可靠度明顯更好

建議旗標:

1. 使用 Mistral AI 官方格式:

    `--tool-call-parser mistral`

2. 有 Transformers 格式可用時:

    `--tokenizer_mode hf --config_format hf --load_format hf --tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

> **備註**
> Mistral AI 官方發布的模型有兩種可能的格式:
>
> 1. 官方格式,`auto` 或 `mistral` 參數預設會用這個:
>
>     `--tokenizer_mode mistral --config_format mistral --load_format mistral`
>     這個格式用的是 Mistral AI 自己的 tokenizer 後端 [mistral-common](https://github.com/mistralai/mistral-common)。
>
> 2. Transformers 格式(如果有的話),`hf` 參數會用這個:
>
>     `--tokenizer_mode hf --config_format hf --load_format hf --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 系列模型(`llama3_json`)

支援的模型:

Llama 3.1、3.2、4 的所有模型應該都支援。

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

支援的是 [JSON 格式的 tool calling](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)。Llama-3.2 引入的 [pythonic tool calling](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling) 見下面的 `pythonic` tool parser。Llama 4 模型則建議用 `llama4_pythonic` tool parser。

其他像內建 python tool calling 或自訂 tool calling 的格式不支援。

已知問題:

1. Llama 3 不支援平行工具呼叫,但 Llama 4 支援。
2. 模型可能會生成格式不正確的參數,例如把陣列序列化成字串,而不是真的陣列。

vLLM 為 Llama 3.1 跟 3.2 提供兩個 JSON 格式的聊天範本:

* [examples/tool_chat_template_llama3.1_json.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja) —— 這是 Llama 3.1 模型「官方」的聊天範本,調整過讓它跟 vLLM 搭配更順
* [examples/tool_chat_template_llama3.2_json.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.2_json.jinja) —— 在 Llama 3.1 聊天範本基礎上,加了圖片支援

建議旗標:`--tool-call-parser llama3_json --chat-template {見上}`

vLLM 也為 Llama 4 提供 pythonic 跟 JSON 兩種聊天範本,但建議用 pythonic tool calling:

* [examples/tool_chat_template_llama4_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama4_pythonic.jinja) —— 基於 Llama 4 模型的[官方聊天範本](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)

Llama 4 模型請用 `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja`。

### IBM Granite

支援的模型:

* `ibm-granite/granite-4.0-h-small` 及其他 Granite 4.0 模型

    建議旗標:`--tool-call-parser granite4`

* `ibm-granite/granite-3.0-8b-instruct`

    建議旗標:`--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

    [examples/tool_chat_template_granite.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_granite.jinja):修改自 Hugging Face 原版聊天範本,支援平行函式呼叫。

* `ibm-granite/granite-3.1-8b-instruct`

    建議旗標:`--tool-call-parser granite`

    可以直接用 Huggingface 上的聊天範本,支援平行函式呼叫。

* `ibm-granite/granite-20b-functioncalling`

    建議旗標:`--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

    [examples/tool_chat_template_granite_20b_fc.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_granite_20b_fc.jinja):修改自 Hugging Face 原版(原版不相容 vLLM),混合了 Hermes 範本裡函式描述的元素,並沿用[論文](https://arxiv.org/abs/2407.00121)裡 "Response Generation" 模式的 system prompt。支援平行函式呼叫。

### InternLM 系列模型(`internlm`)

支援的模型:

* `internlm/internlm2_5-7b-chat`(已確認)
* 其他 internlm2.5 支援函式呼叫的模型應該也相容

已知問題:

* 雖然這個實作也支援 InternLM2,但用 `internlm/internlm2-chat-7b` 測試時,工具呼叫的結果不太穩定。

建議旗標:`--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba 系列模型(`jamba`)

支援 AI21 的 Jamba-1.5 模型。

* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`

旗標:`--tool-call-parser jamba`

### xLAM 系列模型(`xlam`)

xLAM tool parser 設計用來支援輸出多種 JSON 格式工具呼叫的模型,能偵測好幾種不同的輸出風格:

1. 直接輸出 JSON 陣列:以 `[` 開頭、`]` 結尾的字串
2. Thinking 標籤:用 `<think>...</think>` 標籤包住 JSON 陣列
3. 程式碼區塊:放在 ```json ...``` 程式碼區塊裡的 JSON
4. 工具呼叫標籤:用 `[TOOL_CALLS]` 或 `<tool_call>...</tool_call>` 標籤

支援平行函式呼叫,parser 能有效把文字內容跟工具呼叫分開。

支援的模型:

* Salesforce Llama-xLAM 模型:`Salesforce/Llama-xLAM-2-8B-fc-r`、`Salesforce/Llama-xLAM-2-70B-fc-r`
* Qwen-xLAM 模型:`Salesforce/xLAM-1B-fc-r`、`Salesforce/xLAM-3B-fc-r`、`Salesforce/Qwen-xLAM-32B-fc-r`

旗標:

* Llama 系的 xLAM 模型:`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_llama.jinja`
* Qwen 系的 xLAM 模型:`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_qwen.jinja`

### Qwen 系列模型

Qwen2.5 的 `tokenizer_config.json` 裡的聊天範本已經內建支援 Hermes 風格的 tool use,所以 Qwen 模型可以直接用 `hermes` parser 開啟 tool calling。更詳細的說明請參考官方 [Qwen 文件](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)。

* `Qwen/Qwen2.5-*`
* `Qwen/QwQ-32B`

旗標:`--tool-call-parser hermes`

> 這正好對應到我們在 [quickstart.md](../../quickstart.md)、[quantization-benchmark.md](../../quantization-benchmark.md) 裡一路測到現在的 `Qwen2.5-1.5B-Instruct` / `Qwen2.5-7B-Instruct-FP8-dynamic` ——之後要在這台機器上實測 tool calling,直接加 `--tool-call-parser hermes` 就能用,不用額外找聊天範本。

### DeepSeek-V3 系列模型(`deepseek_v3`)

支援的模型:

* `deepseek-ai/DeepSeek-V3-0324`(搭配 [examples/tool_chat_template_deepseekv3.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_deepseekv3.jinja))
* `deepseek-ai/DeepSeek-R1-0528`(搭配 [examples/tool_chat_template_deepseekr1.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_deepseekr1.jinja))

旗標:`--tool-call-parser deepseek_v3 --chat-template {見上}`

### DeepSeek-V3.1 系列模型(`deepseek_v31`)

支援的模型:

* `deepseek-ai/DeepSeek-V3.1`(搭配 [examples/tool_chat_template_deepseekv31.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_deepseekv31.jinja))

旗標:`--tool-call-parser deepseek_v31 --chat-template {見上}`

### OpenAI OSS 系列模型(`openai`)

支援的模型:

* `openai/gpt-oss-20b`
* `openai/gpt-oss-120b`

旗標:`--tool-call-parser openai`

### Kimi-K2 系列模型(`kimi_k2`)

支援的模型:

* `moonshotai/Kimi-K2-Instruct`

旗標:`--tool-call-parser kimi_k2`

### Hunyuan 系列模型(`hunyuan_a13b`)

支援的模型:

* `tencent/Hunyuan-A13B-Instruct`(聊天範本已內建在 Hugging Face 模型檔案裡)

旗標:

* 非 reasoning 模式:`--tool-call-parser hunyuan_a13b`
* reasoning 模式:`--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`

### Cohere Command 系列模型(`cohere_command3` / `cohere_command4`)

支援的模型:

* [`CohereLabs/command-a-reasoning-08-2025`](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025)(`cohere_command3`)
* [`CohereLabs/command-a-plus-05-2026`](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16) 與 [`CohereLabs/North-Mini-Code-1.0`](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)(`cohere_command4`)

旗標:`--tool-call-parser cohere_command4 --reasoning-parser cohere_command4`

備註:Cohere 的 parser 需要 `cohere_melody` 套件,預設不會安裝,使用前請先安裝 [cohere_melody](https://pypi.org/project/cohere-melody/)。

### LongCat-Flash-Chat 系列模型(`longcat`)

支援的模型:

* `meituan-longcat/LongCat-Flash-Chat`
* `meituan-longcat/LongCat-Flash-Chat-FP8`

旗標:`--tool-call-parser longcat`

### GLM-4.5 系列模型(`glm45`)

支援的模型:

* `zai-org/GLM-4.5`
* `zai-org/GLM-4.5-Air`
* `zai-org/GLM-4.6`

旗標:`--tool-call-parser glm45`

### GLM-4.7 系列模型(`glm47`)

支援的模型:

* `zai-org/GLM-4.7`
* `zai-org/GLM-4.7-Flash`

旗標:`--tool-call-parser glm47`

### FunctionGemma 系列模型(`functiongemma`)

Google 的 FunctionGemma 是個專為函式呼叫設計的輕量模型(2.7 億參數),基於 Gemma 3,針對筆電、手機這類邊緣裝置部署做了最佳化。

支援的模型:

* `google/functiongemma-270m-it`

FunctionGemma 用的是獨特的輸出格式,以 `<start_function_call>` 跟 `<end_function_call>` 標籤包住:

```text
<start_function_call>call:get_weather{location:<escape>London<escape>}<end_function_call>
```

這個模型設計上是要針對特定函式呼叫任務微調(fine-tune)過,才能得到最好的結果。

旗標:`--tool-call-parser functiongemma --chat-template examples/tool_chat_template_functiongemma.jinja`

> **備註**
> FunctionGemma 設計上就是要給你針對自己的函式呼叫任務去微調。基礎模型提供的是通用的函式呼叫能力,但要達到最好的效果,還是要做任務專屬的微調。微調指南請參考 Google 的 [FunctionGemma 文件](https://ai.google.dev/gemma/docs/functiongemma)。

### Qwen3-Coder 系列模型(`qwen3_xml`)

支援的模型:

* `Qwen/Qwen3-Coder-480B-A35B-Instruct`
* `Qwen/Qwen3-Coder-30B-A3B-Instruct`

旗標:`--tool-call-parser qwen3_xml`

### Olmo 3 系列模型(`olmo3`)

Olmo 3 模型輸出工具呼叫的格式跟下面 `pythonic` parser 預期的格式很類似,但有幾點不同:每個工具呼叫都是一個 pythonic 字串,但平行工具呼叫是用換行分隔,而且整段呼叫會包在 `<function_calls>..</function_calls>` 這樣的 XML 標籤裡。另外,除了 pythonic 的寫法(`True`、`False`、`None`)之外,parser 也接受 JSON 的布林值跟 null 字面值(`true`、`false`、`null`)。

支援的模型:

* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-32B-Think`

旗標:`--tool-call-parser olmo3`

### Gigachat 3 系列模型(`gigachat3`)

使用 Hugging Face 模型檔案裡的聊天範本。

支援的模型:

* `ai-sage/GigaChat3-702B-A36B-preview`
* `ai-sage/GigaChat3-702B-A36B-preview-bf16`
* `ai-sage/GigaChat3-10B-A1.8B`
* `ai-sage/GigaChat3-10B-A1.8B-bf16`

旗標:`--tool-call-parser gigachat3`

### Apertus 系列模型(`apertus`)

使用 examples 資料夾裡的聊天範本,修正了幾個 OpenAI 相容性問題:`--chat-template /vllm-workspace/examples/tool_chat_template_apertus.jinja`

支援的模型:

* `swiss-ai/Apertus-8B-Instruct-2509`
* `swiss-ai/Apertus-70B-Instruct-2509`

旗標:`--tool-call-parser apertus`

### 支援 Pythonic 工具呼叫的模型(`pythonic`)

愈來愈多模型改用 python 清單(list)來表示工具呼叫,而不是用 JSON。這樣做的好處是天生就支援平行工具呼叫,也不用擔心工具呼叫的 JSON schema 有沒有寫對。`pythonic` tool parser 就是用來支援這類模型的。

舉個具體例子,這類模型查舊金山跟西雅圖的天氣時可能會生成:

```python
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]
```

限制:

* 模型不能在同一次生成裡同時輸出文字內容跟工具呼叫。對特定模型來說這可能不難改,但社群目前對「開始跟結束工具呼叫時該生成哪些 token」還沒有共識(尤其 Llama 3.2 模型完全不會生成這類 token)。
* Llama 比較小的模型不太會有效使用工具。

支援的模型範例:

* `meta-llama/Llama-3.2-1B-Instruct` ⚠️(搭配 [examples/tool_chat_template_llama3.2_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.2_pythonic.jinja))
* `meta-llama/Llama-3.2-3B-Instruct` ⚠️(搭配 [examples/tool_chat_template_llama3.2_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.2_pythonic.jinja))
* `Team-ACE/ToolACE-8B`(搭配 [examples/tool_chat_template_toolace.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_toolace.jinja))
* `fixie-ai/ultravox-v0_4-ToolACE-8B`(搭配 [examples/tool_chat_template_toolace.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_toolace.jinja))
* `meta-llama/Llama-4-Scout-17B-16E-Instruct` ⚠️(搭配 [examples/tool_chat_template_llama4_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama4_pythonic.jinja))
* `meta-llama/Llama-4-Maverick-17B-128E-Instruct` ⚠️(搭配 [examples/tool_chat_template_llama4_pythonic.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama4_pythonic.jinja))

旗標:`--tool-call-parser pythonic --chat-template {見上}`

> **警告**
> Llama 比較小的模型常常無法用正確的格式生成工具呼叫,結果依模型而異。

## Tool-Calling 效能基準測試

要在真實的 tool-calling 流量下測量服務延遲與吞吐量,可以用 BFCL(Berkeley Function Calling Leaderboard)資料集搭配 `vllm bench serve`。完整的伺服器+客戶端指令請見 [BFCL benchmark 範例](https://docs.vllm.ai/en/latest/benchmarking/cli.html#bfcl-tool-calling-benchmark)。

## 如何自己寫 Tool Parser 外掛(略過)

官方文件這節說明怎麼寫一個 tool parser 外掛(繼承 `ToolParser`、實作 `extract_tool_calls`/`extract_tool_calls_streaming` 等方法,再用 `ToolParserManager.register_lazy_module` 註冊)。這是給要**自己開發新 parser**(支援一個目前還沒被支援的模型)的人看的,對「用現成 parser 接現成模型」的部署情境用不到,這裡不展開,有需要直接看[官方原文](https://docs.vllm.ai/en/latest/features/tool_calling/#how-to-write-a-tool-parser-plugin)。
