# Automatic Prefix Caching(自動前綴快取)

> 官方原文:[docs.vllm.ai — Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
> 這篇描述的是 vLLM v1 目前實際在用的架構,不是過時文件,全文完整翻譯。

Prefix caching(前綴快取)是 LLM 推論裡很常見的一種最佳化手法,用來避免重複計算相同的 prompt 內容。核心概念很簡單——把已經處理過的請求的 KV cache block 快取起來,當新請求進來、跟之前的請求有相同前綴時,就重複使用這些 block。因為 prefix caching 幾乎是「白吃的午餐」(不會改變模型輸出),很多公開服務(例如 OpenAI、Anthropic)跟大多數開源 LLM 推論框架(例如 SGLang)都廣泛採用。

實作 prefix caching 的方式有很多種,vLLM 選擇的是基於雜湊(hash)的做法。具體來說,每個 KV cache block 的雜湊值,是用「這個 block 裡的 token」加上「這個 block 之前所有的前綴 token」一起算出來的:

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

以上面這個例子來說,第一個 block 的 KV cache 可以單靠 token「A gentle breeze stirred」就唯一辨識出來。第三個 block 則要靠這個 block 本身的 token「laughed in the distance」,加上前面所有的前綴 token「A gentle breeze stirred the leaves as children」才能唯一辨識。因此,我們可以建立 block 的雜湊值 `hash(tuple[components])`,其中 components 包含:

* **父雜湊值(Parent hash value)**:上一個 block 的雜湊值
* **Block tokens**:這個 block 裡的 token 序列(用具體的 token 內容而不只是位置,是為了降低雜湊碰撞的機率)
* **額外雜湊值(Extra hashes)**:其他讓這個 block 保持唯一性所需的值,例如 LoRA ID、多模態輸入的雜湊值(見下方範例)、還有用來在多租戶環境裡隔離快取的 cache salt

> **備註 1**
> 只有完整填滿的 block 才會被快取。

> **備註 2**
> 舊版本裡,雜湊金鑰不保證不會碰撞。從 v0.11 開始,預設的雜湊演算法改成 `sha256`,解決了碰撞風險的問題。
>
> 用 `vllm serve` 時,可以透過 `--prefix-caching-hash-algo` 控制要用哪種雜湊演算法:
>
> - `sha256`(預設):用 Python 的 `pickle` 做序列化。雜湊值在不同 Python 或 vLLM 版本之間不保證能重現。
> - `sha256_cbor`:用 `cbor2` 做序列化,提供可重現、跨語言相容的雜湊值。要在不同環境間需要確定性快取的話,推薦用這個。
> - `xxhash`:用 Pickle 序列化搭配 xxHash(128 位元)做更快、非加密用途的雜湊,需要額外安裝 `xxhash` 套件。**重點提醒**:使用不具備密碼學安全性的雜湊演算法,理論上會提高雜湊碰撞的風險,在多租戶環境下可能導致不可預期的行為,甚至洩漏私人資訊。即使碰撞機率仍然很低,開啟這個選項前還是要衡量自己能接受的安全風險,對比它帶來的效能好處。
> - `xxhash_cbor`:結合標準 CBOR 序列化與 xxHash,提供可重現的雜湊。同樣需要額外安裝 `xxhash` 套件。

**多模態輸入的雜湊範例**
這個例子示範多模態輸入(例如圖片)時 prefix caching 怎麼運作。假設有一個請求帶著以下訊息:

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

會變成以下的 prompt:

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

可以看到,tokenize 之後,`[IMG]` 會被替換成一連串的佔位 token(placeholder),這些佔位 token 會在 prefill 階段被替換成圖片的 embedding。這種情況下 prefix caching 遇到的挑戰是:要能區分「圖片」跟「佔位符」。解法是把前端圖片處理器產生的圖片雜湊值也編碼進去。舉例來說,假設 block size 是 16、總共有 41 個佔位 token,上面這個 prompt 裡各 block 的雜湊值會是這樣:

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 2
    Parent hash: Block 1 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 3
    Parent hash: Block 2 hash
    Token IDs: <p>, ..., <p>, 4
    Extra hash: <image hash>
```

本文接下來會先介紹 vLLM v1 裡 prefix caching 用到的資料結構,再說明主要 KV cache 操作(配置、附加、釋放、驅逐)的 prefix caching 工作流程,最後用一個範例示範完整的端對端 prefix caching 流程。

**為了安全性而做的快取隔離**
為了在共享環境裡提升隱私,vLLM 支援透過選填的「每個請求各自加鹽(salt)」來隔離前綴快取的重複使用。在請求裡加入 `cache_salt`,這個值會被加進第一個 block 的雜湊值計算裡,確保只有帶著相同 salt 的請求才能重複使用快取的 KV block。這可以防止「透過觀察延遲差異來推測快取內容」這種計時攻擊(timing attack),而且不會犧牲效能。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

這樣設定之後,快取共享就只會發生在明確約定同一組 salt 的使用者或請求之間,讓信任群組內可以重複使用快取,同時把其他人隔離開來。

## 資料結構

vLLM v1 的 prefix caching 是在 KV cache manager 裡實作的。最基本的建構單位是(簡化過的)`Block` 資料類別:

```python
class KVCacheBlock:
    # block 的 ID(不可變)
    block_id: int
    # block 的雜湊值(block 填滿時才會賦值,
    # block 被驅逐時會重設)
    block_hash: BlockHash
    # 目前有幾個請求正在使用這個 block
    ref_cnt: int

    # 用來組成「空閒佇列」雙向鏈結串列的指標
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

有兩個設計重點值得特別說明:

1. KV cache manager 初始化時,會一次配置好所有的 `KVCacheBlock`,形成一個 block pool。這樣可以避免 Python 物件建立的額外開銷,而且隨時都能輕鬆追蹤所有 block。
2. 直接在 `KVCacheBlock` 裡放雙向鏈結串列的指標,讓我們能直接組出一個空閒佇列(free queue)。這帶來兩個好處:
    1. 把中間的元素移到尾端,只需要 O(1) 的複雜度。
    2. 不用額外引入像 Python 的 `deque` 這種還要包一層 wrapper 的佇列。

因此,KV cache manager 初始化時會有以下幾個元件:

![元件總覽](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/overview.png)

* **Block Pool**:一個 `KVCacheBlock` 的清單
* **Free Block Queue(空閒 block 佇列)**:只儲存頭尾 block 的指標,方便操作
* **Cache blocks(快取 block)**:雜湊金鑰對應到 block ID 的映射表
* **Request blocks(請求 block)**:請求 ID 對應到已配置 block ID 的映射表

## 操作

### Block 配置

**新請求:** 排程器(scheduler)為新請求配置 KV cache block 的流程:

1. 排程器呼叫 `kv_cache_manager.get_computed_blocks()`,取得一串已經算過的 block。做法是把請求裡的 prompt token 做雜湊,再去查詢快取的 block。
2. 排程器呼叫 `kv_cache_manager.allocate_slots()`,會執行以下步驟:
    1. 算出還需要多少新的 block,如果沒有足夠的 block 可以配置就返回。
    2. 「碰觸(touch)」已經算好的 block:把該 block 的參照計數(reference count)加一,如果這個 block 沒有被其他請求使用,就把它從空閒佇列裡移除。這是為了避免這些已算好的 block 被驅逐——見下一節的範例說明。
    3. 從空閒佇列的頭端彈出 block 來配置成新 block。如果彈出的 block 剛好是已被快取的 block,這個動作同時也會「驅逐(evict)」該 block,之後其他請求就不能再重複使用它了。
    4. 如果某個新配置的 block 剛好已經被 token 填滿,會立刻把它加進快取 block 裡,這樣同一批次裡的其他請求就能馬上重複使用它。

**執行中的請求:** 排程器為執行中的請求配置 KV cache block 的流程:

1. 排程器呼叫 `kv_cache_manager.allocate_slots()`,會執行以下步驟:
    1. 算出還需要多少新的 block,如果沒有足夠的 block 可以配置就返回。
    2. 從空閒佇列的頭端彈出 block 來配置成新 block。如果彈出的 block 剛好是已被快取的 block,同樣會把它「驅逐」。
    3. 把新產生的 token ID 附加到現有 block 跟新 block 的空位裡。如果某個 block 滿了,就把它加進快取 block 裡。

**重複的 block**
假設 block size 是 4,你送出一個請求(請求 1),prompt 是 ABCDEF,解碼長度是 3:

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

現在 block 0 跟 block 1 都已經被快取了,這時我們用貪婪取樣(greedy sampling)再送出同一個請求(請求 2),它會產生跟請求 1 完全一樣的輸出:

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

可以看到,block 3 是一個新的完整 block,而且也被快取了。但它其實跟 block 1 是重複的內容,等於同樣的內容被快取了兩次。在 v0 裡,偵測到 block 3 重複時,會直接釋放 block 3,讓請求 2 改用 block 1,所以它的 block table 在 Time 1 會變成 `[0, 1]`。但 vLLM v1 的 block table 是「只能附加(append-only)」的,不允許把 block table 從 `[0, 3]` 改成 `[0, 1]`。所以雜湊金鑰 E-H 會有重複的 block 並存,這個重複要等到請求被釋放時才會消除。

### 釋放(Free)

當一個請求結束時,如果它的 block 沒有其他請求在用(參照計數為 0),我們會釋放它所有的 block。在這個範例裡,我們釋放請求 1,以及跟它相關的 block 2、3、4、8。可以看到,被釋放的 block 是以**反向**順序加到空閒佇列的尾端——因為一個請求裡最後面的 block,雜湊時用到的 token 最多,被其他請求重複使用的機率也最低,所以應該優先被驅逐。

![請求釋放後的空閒佇列](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/free.png)

### 驅逐(LRU)

當空閒佇列頭端的 block(也就是最近最少使用、LRU 的 block)剛好是已快取的 block 時,就必須把它驅逐,避免它繼續被其他請求使用。具體來說,驅逐包含以下步驟:

1. 從空閒佇列頭端彈出 block,這就是要被驅逐的 LRU block。
2. 從快取 block 裡移除這個 block ID。
3. 移除這個 block 的雜湊值。

## 範例

這個範例假設 block size 是 4(每個 block 能快取 4 個 token),KV cache manager 裡總共有 10 個 block。

**時間 1:快取是空的,一個新請求進來。** 我們配置 4 個 block,其中 3 個已經滿了、被快取,第 4 個 block 只填了 4 個 token 裡的 3 個。

![範例時間 1](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-1.png)

**時間 2:請求 0 把 block 3 填滿,並要求新的 block 繼續解碼。** 我們快取 block 3,並配置 block 4。

![範例時間 2](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-3.png)

**時間 3:請求 1 進來,帶著 14 個 prompt token,其中前 10 個 token 跟請求 0 一樣。** 可以看到只有前 2 個 block(8 個 token)命中快取,因為第 3 個 block 只有 4 個 token 裡的 2 個相符。

![範例時間 3](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-4.png)

**時間 4:請求 0 結束並釋放。** Block 2、3、4 以反向順序加進空閒佇列(但 block 2 跟 3 仍然是被快取的狀態)。Block 0 跟 1 因為還在被請求 1 使用,不會加進空閒佇列。

![範例時間 4](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-5.png)

**時間 5:請求 1 結束並釋放。**

![範例時間 5](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-6.png)

**時間 6:請求 2 進來,帶著 29 個 prompt token,其中前 12 個跟請求 0 一樣。** 注意雖然空閒佇列裡 block 的順序是 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`,但快取命中的 block(也就是 0、1、2)在配置前就已經被「碰觸」並從佇列中移除,所以空閒佇列變成 `7 - 8 - 9 - 4 - 3 - 6 - 5`。因此,最後配置到的 block 是 0(快取命中)、1(快取命中)、2(快取命中)、7、8、9、4、3(被驅逐)。

![範例時間 6](https://docs.vllm.ai/en/latest/assets/design/prefix_caching/example-time-7.png)
