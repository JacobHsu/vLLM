# Structured Outputs 本機實測記錄

> 對照 [vLLM/features/structured_outputs.zh.md](vLLM/features/structured_outputs.zh.md) 的四種模式(`choice`/`regex`/`json`/`grammar`),在這台機器上實際跑一次。
> 環境同 [quickstart.md](quickstart.md):WSL2 + RTX 5080,用 `Qwen/Qwen2.5-1.5B-Instruct`(未量化)測試,因為這個功能測的是格式限制邏輯,不需要大模型。
> 測試腳本:[scripts/test_structured_outputs.py](scripts/test_structured_outputs.py)

## 結果:四種模式全部正常,沒有遇到任何 Blackwell 相關問題

| 模式 | 請求內容 | 實際輸出 | 結果 |
|---|---|---|---|
| `choice` | 判斷「vLLM is wonderful!」的情緒,只能選 positive/negative | `positive` | ✅ 完全符合限定選項 |
| `regex` | 產生一個 email,格式要符合 `\w+@\w+\.com\n` | `alan_turing@enigma.com` | ✅ 符合 regex(模型自己把空格換成底線,regex 允許) |
| `json` | 產生符合 `CarDescription` schema(brand/model/car_type)的 JSON | `{"brand": "Ford", "model": "Mustang", "car_type": "Coupe"}` | ✅ 合法 JSON,`car_type` 也剛好落在 enum 允許值裡 |
| `grammar` | 用簡化版 SQL 文法產生查詢 | `SELECT col_1  from table_1  where col_2 = 1` | ✅ 完全符合自訂 EBNF 文法 |

## 為什麼這個功能沒踩到 Blackwell 的坑

跟量化(需要 GPU 上專屬的矩陣運算 kernel)不一樣,structured outputs 是在**取樣(sampling)這一步做文章**——後端(xgrammar)在每個 token 生成的當下,把不符合 schema/regex/文法的 token 直接排除掉,讓模型只能從「合法」的選項裡選。這主要是 CPU 端的邏輯運算(維護一個有限狀態機/FSM),不需要新的 GPU kernel,所以跟顯卡世代沒什麼關係,難怪一次就過。

## 實測指令

```bash
wsl -d Ubuntu
export REPO_DIR=/mnt/x/path/to/vllm-repo   # 換成你自己 clone 這個 repo 的實際路徑
bash "$REPO_DIR/scripts/run_server.sh"
```

另開一個 shell 執行測試腳本:

```bash
cd ~/vllm-quickstart && source .venv/bin/activate
python "$REPO_DIR/scripts/test_structured_outputs.py"
```

## 下一步

- 這次測的是離線/線上服務都用得到的四種基本模式;`structural_tag`、搭配 reasoning 模型的用法(文件裡有提到)還沒測
