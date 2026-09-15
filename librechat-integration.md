# vLLM 對接 LibreChat 聊天後端

本文件說明:這台機器上跑的 vLLM(`Qwen/Qwen2.5-1.5B-Instruct`)除了被這個 repo 自己的腳本測試,也被另一個獨立專案 **LibreChat** 當成聊天後端串接使用。本文件只涵蓋「從 vLLM 這一側看,LibreChat 如何連入、vLLM 端需要配合哪些條件」;LibreChat 那邊完整的設定與 Agent/工具呼叫計畫,寫在它自己 repo 的文件裡(見文末〈完整細節在另一個 repo〉),不在此重複。

## 連線架構

LibreChat 跑在 Windows 主機上的 Docker 容器裡(不是 WSL2),vLLM 跑在 WSL2 裡,兩者中間隔了一層 Docker 網路 + 一層 WSL2 網路:

```
LibreChat API 容器(Docker,Windows 主機上)
   │  baseURL: http://host.docker.internal:8000/v1/
   ▼
host.docker.internal → Windows 主機
   │  (Docker 的 extra_hosts: host-gateway)
   ▼
Windows 主機的 8000 port
   │  (WSL2 的 localhost forwarding,自動轉發)
   ▼
vLLM(WSL2 Ubuntu,監聽 0.0.0.0:8000)
```

**已驗證(2026-09-15)**:`host.docker.internal` 這條路徑可直接連通,不需要另外查 WSL2 的內部 IP(`wsl hostname -I`)。跟 Windows 主機上其他服務(例如 Ollama)走的是同一條路徑。

## 怎麼確認 vLLM 服務有沒有在跑

**在 WSL2 裡直接確認：** 先用 `wsl -d Ubuntu` 進入 Ubuntu shell,再執行下面的指令(跟 quickstart.md 的慣例一致)。

```bash
# 行程是否存在
pgrep -fa vllm

# port 8000 是否有服務在監聽
ss -ltnp | grep 8000

# 呼叫 API 確認回應正常，並列出目前載入的模型
curl -s http://localhost:8000/v1/models
```

**從 Windows 主機確認**（驗證 WSL2 的 localhost forwarding 有生效，這是 LibreChat 那條路徑實際會走的方向）：

```powershell
curl http://localhost:8000/v1/models
```

兩邊都要回傳 JSON、列出 `Qwen/Qwen2.5-1.5B-Instruct`，才代表〈連線架構〉裡那條路徑真的通。也可以用更輕量的 liveness check：`curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health`，回 `200` 就是活的。

**沒有回應時：**

- 看背景啟動時導出的 log 檔（例如 `/tmp/vllm-server.log`，見下方〈vLLM 端要配合的條件〉第 2 點的啟動指令)有沒有錯誤訊息。
- 確認啟動方式有沒有用 `setsid`（見下方第 2 點）：只用 `nohup ... & disown` 而沒加 `setsid`，行程會在啟動指令的 shell 結束時被系統回收，`pgrep` 會直接查不到。

## vLLM 端要配合的條件

1. **一定要監聽 `0.0.0.0`,不能只監聽 `127.0.0.1`**——WSL2 的 localhost forwarding 只轉發綁在所有介面的服務。`scripts/run_server.sh`、`scripts/run_server_tool_calling.sh` 都沒有加 `--host`,vLLM 預設值就是 `0.0.0.0`,不用額外處理,但如果之後改腳本手動指定 host,要記得別改成 `127.0.0.1`。
2. **背景啟動要用 `setsid`,不能只靠 `nohup ... & disown`**——這是在幫 LibreChat 那邊測試時踩到的坑:用 `wsl.exe -d Ubuntu -- bash -lc "nohup ... & disown"` 這種一次性指令啟動,`wsl.exe` 呼叫本身一結束,vLLM 行程就被系統回收了,即使有 `nohup`/`disown` 也擋不住。要完全脫離終端機存活,得再加一層 `setsid`:
   ```bash
   setsid nohup bash scripts/run_server.sh > /tmp/vllm-server.log 2>&1 < /dev/null &
   disown
   ```
3. **要支援 tool calling(agent 呼叫工具)時,必須用 `scripts/run_server_tool_calling.sh`,不能用 `scripts/run_server.sh`**——後者沒有 `--enable-auto-tool-choice --tool-call-parser hermes`,LibreChat 那邊的 agent 送出帶 `tools` 參數的請求時,vLLM 不會把輸出解析成 `tool_calls`,agent 會直接把工具定義當成一般文字回覆,不會觸發工具呼叫。這件事在 [tool-calling-test.md](tool-calling-test.md) 裡已經用 vLLM 自己的 Python client 驗證過工具呼叫本身沒問題,但**目前跟 LibreChat 對接測試時,vLLM 是用不帶這兩個參數的 `run_server.sh` 啟動的**,只驗證了純聊天,還沒驗證 LibreChat agent 呼叫工具這條路。

## 目前驗證進度(從 LibreChat 那一側做的測試)

- ✅ 網路連通性:LibreChat 的 Docker 容器內部直接打 `http://host.docker.internal:8000/v1/models` 有回應。
- ✅ LibreChat 的模型清單動態抓取(`fetch: true`)成功抓到 `Qwen/Qwen2.5-1.5B-Instruct`。
- ✅ 純聊天端到端測試:透過 LibreChat 正式的聊天 API 送出訊息,vLLM 產生的回覆正確送達並顯示。
- ⬜ Agent 呼叫工具(需要先把 vLLM 換成 `run_server_tool_calling.sh` 啟動,並在 LibreChat 那邊建好 agent + 掛工具,兩邊都還沒做)。

## 完整細節在另一個 repo

LibreChat 那邊的 `librechat.yaml` 設定、Agent Builder 操作步驟、工具怎麼掛(Actions vs MCP)的完整計畫,寫在 LibreChat 專案自己的文件裡。以下僅標示檔名,跨 repo 無法用相對連結指過去:

- `docs/study/vllm-tool-agent-demo.md`:整個 demo 的目標、已驗證事實、原始執行步驟
- `docs/study/vllm-librechat-integration-plan.md`:打開 Agents、建 agent、掛工具、端到端測試的分階段手動驗證計畫
