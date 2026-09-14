# Structured Outputs(結構化輸出)

> 官方原文:[docs.vllm.ai — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
> 這篇篇幅適中(跟 quickstart 同量級),官方內容全部翻譯,程式碼區塊保持原樣可直接執行。
> 範例假設已經有 vLLM 伺服器在跑,可參考 [quickstart.md](../../quickstart.md) 或 [scripts/run_server.sh](../../scripts/run_server.sh)。

vLLM 支援用 [xgrammar](https://github.com/mlc-ai/xgrammar) 或 [guidance](https://github.com/guidance-ai/llguidance) 當後端來產生結構化輸出。本文示範幾種產生結構化輸出的不同做法。

> **警告**
> 如果你還在用以下這些在 v0.12.0 已經移除的舊版 API 欄位,請改用本文示範的 `structured_outputs`:
>
> - `guided_json` → `{"structured_outputs": {"json": ...}}` 或 `StructuredOutputsParams(json=...)`
> - `guided_regex` → `{"structured_outputs": {"regex": ...}}` 或 `StructuredOutputsParams(regex=...)`
> - `guided_choice` → `{"structured_outputs": {"choice": ...}}` 或 `StructuredOutputsParams(choice=...)`
> - `guided_grammar` → `{"structured_outputs": {"grammar": ...}}` 或 `StructuredOutputsParams(grammar=...)`
> - `guided_whitespace_pattern` → `{"structured_outputs": {"whitespace_pattern": ...}}` 或 `StructuredOutputsParams(whitespace_pattern=...)`
> - `structural_tag` → `{"structured_outputs": {"structural_tag": ...}}` 或 `StructuredOutputsParams(structural_tag=...)`
> - `guided_decoding_backend` → 直接從請求裡移除這個欄位

## 線上服務(OpenAI API)

可以用 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 與 [Chat](https://platform.openai.com/docs/api-reference/chat) API 產生結構化輸出。

支援以下參數,要用 extra parameters 的方式加進請求:

- `choice`:輸出必定是給定選項裡的其中一個
- `regex`:輸出會符合指定的正規表示式
- `json`:輸出會符合指定的 JSON schema
- `grammar`:輸出會符合指定的上下文無關文法(context free grammar)
- `structural_tag`:在產生的文字裡,特定標籤範圍內的內容要符合 JSON schema

完整的支援參數清單可以查看 [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server.html) 頁面。

OpenAI-Compatible Server 預設就支援結構化輸出。可以透過 `vllm serve` 的 `--structured-outputs-config.backend` 參數指定要用哪個後端,預設是 `auto`,會依請求內容自動選一個合適的後端。也可以指定特定後端搭配一些選項,完整選項清單可查 `vllm serve --help`。

先從最簡單的 `choice` 開始看範例:

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)
model = client.models.list().data[0].id

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
)
print(completion.choices[0].message.content)
```

接下來是 `regex` 的用法。支援的正規表示式語法依結構化輸出後端而異:例如 `xgrammar`、`guidance`、`outlines` 用的是 Rust 風格正規表示式,而 `lm-format-enforcer` 用的是 Python 的 `re` 模組。這個範例是用簡單的 regex 樣板產生一個電子郵件地址:

```python
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
        }
    ],
    extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
)
print(completion.choices[0].message.content)
```

結構化文字生成裡最實用的功能之一,就是能產生符合預先定義好的欄位跟格式的合法 JSON。`json` 參數有兩種用法:

- 直接給一個 [JSON Schema](https://json-schema.org/)
- 定義一個 [Pydantic model](https://docs.pydantic.dev/latest/),再從裡面萃取出 JSON Schema(通常這樣比較省事)

下面示範用 Pydantic model 搭配 `response_format` 參數:

```python
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema()
        },
    },
)
print(completion.choices[0].message.content)
```

> **提示**
> 雖然不是必要的,但通常在 prompt 裡也順便說明一下 JSON schema 跟各欄位該怎麼填,大多數情況下都能明顯改善結果品質。

最後是 `grammar` 選項,大概是最難用、但也最強大的一種——可以用來定義完整的語言,例如 SQL 查詢語法。它是用上下文無關的 EBNF 文法運作的。以下範例定義了一種簡化版 SQL 查詢的格式:

```python
simplified_sql_grammar = """
    root ::= select_statement

    select_statement ::= "SELECT " column " from " table " where " condition

    column ::= "col_1 " | "col_2 "

    table ::= "table_1 " | "table_2 "

    condition ::= column "= " number

    number ::= "1 " | "2 "
"""

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
        }
    ],
    extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
)
print(completion.choices[0].message.content)
```

另見:[完整範例](https://github.com/vllm-project/vllm/tree/main/examples/features/structured_outputs)

## 搭配 Reasoning(推理)輸出

結構化輸出也能搭配推理模型的 reasoning 功能一起用:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

推理功能可以搭配任何一種結構化輸出功能使用。下面用 JSON schema 示範:

```python
from pydantic import BaseModel


class People(BaseModel):
    name: str
    age: int


completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the name and age of one random person.",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "people",
            "schema": People.model_json_schema()
        }
    },
)
print("reasoning: ", completion.choices[0].message.reasoning)
print("content: ", completion.choices[0].message.content)
```

另見:[完整範例](https://github.com/vllm-project/vllm/tree/main/examples/features/structured_outputs)

> **備註**
> 使用 Qwen3 Coder 系列模型並開啟 reasoning 時,如果 reasoning 內容沒有被正確解析進獨立的 `reasoning` 欄位,結構化輸出可能會被停用(v0.11.2+ 適用)。要讓兩個功能同時運作,必須明確在 reasoning 模式下啟用結構化輸出,啟動 vLLM 伺服器時加上:`--structured-outputs-config.enable_in_reasoning=True`。另見 Reasoning Outputs 文件。

## 實驗性的自動解析功能(OpenAI API)

這一節說明 OpenAI 用戶端函式庫裡,針對 `client.chat.completions.create()` 方法的 beta 版包裝,能跟 Python 特有的型別做更深入的整合。

寫這篇文件時(`openai==1.54.4`),這在 OpenAI 用戶端函式庫裡還是 "beta" 功能,程式碼參考見[這裡](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104)。

以下範例假設 vLLM 是用 `vllm serve meta-llama/Llama-3.1-8B-Instruct` 啟動的。

用 Pydantic model 取得結構化輸出的簡單範例:

```python
from pydantic import BaseModel
from openai import OpenAI

class Info(BaseModel):
    name: str
    age: int

client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
model = client.models.list().data[0].id
completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
    ],
    response_format=Info,
)

message = completion.choices[0].message
print(message)
assert message.parsed
print("Name:", message.parsed.name)
print("Age:", message.parsed.age)
```

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

再看一個比較複雜的範例,用巢狀的 Pydantic model 處理數學解題的逐步過程:

```python
from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str

completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful expert math tutor."},
        {"role": "user", "content": "Solve 8x + 31 = 2."},
    ],
    response_format=MathResponse,
)

message = completion.choices[0].message
print(message)
assert message.parsed
for i, step in enumerate(message.parsed.steps):
    print(f"Step #{i}:", step)
print("Answer:", message.parsed.final_answer)
```

輸出:

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
Step #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
Step #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
Step #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
Answer: x = -29/8
```

`structural_tag` 的用法範例見這裡:[examples/features/structured_outputs](https://github.com/vllm-project/vllm/tree/main/examples/features/structured_outputs)

## 離線推論

離線推論也支援同樣類型的結構化輸出。做法是用 `SamplingParams` 裡的 `StructuredOutputsParams` 類別來設定,`StructuredOutputsParams` 裡主要可用的選項有:

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

用法跟上面線上服務的範例一樣。以下示範 `choice` 參數的用法:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

另見:[完整範例](https://github.com/vllm-project/vllm/blob/main/examples/features/structured_outputs/structured_outputs_offline.py)
