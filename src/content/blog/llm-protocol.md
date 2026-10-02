---
title: '大语言模型接口协议'
description: '本文对比介绍了 Anthropic Messages API、OpenAI Chat Completions API 和 OpenAI Responses API 三种主流的大语言模型接口协议，并探讨了在考虑到 KV cache 复用的情况下如何更好的使用协议'
pubDate: '2026-10-01'
tags: ['llm', 'inference', 'openai api', 'anthropic api']
---

大语言模型的主流接口协议，主要有 Anthropic Messages API、OpenAI Chat Completions API 和 OpenAI Responses API，下面我们对比一下这三种 API，并探讨一下在考虑到 KV cache 复用的情况下如何更好的使用协议。

## 1 Anthropic Messages API v.s. OpenAI Chat Completions API

| 方面 | OpenAI Chat Completions | Anthropic Messages |
|---|---|---|
| 鉴权 | `Authorization: Bearer` | `x-api-key`，外加必填的 `anthropic-version` 头 |
| 系统提示词 | 作为 `role: "system"`（或 `developer`）消息放进 messages | 独立的顶层 `system` 字段，messages 里只有 user/assistant |
| `max_tokens` | 可选 | 必填 |
| 消息内容 | 字符串，或 parts 数组（`text`、`image_url` 等） | 字符串，或 content blocks 数组（`text`、`image`、`document`、`tool_use`、`tool_result`、`thinking` 等） |
| 响应结构 | `choices[].message`，可用 `n` 一次生成多个候选 | 单个 `content` 数组，没有多候选 |
| 流式输出 | 每个 chunk 都是 data: {...}，里面的 choices[0].delta 带增量文本或增量的工具参数，最后以 data: [DONE] 结束 | 带类型的事件流：message_start → content_block_start → 若干 content_block_delta → content_block_stop →（下一个块）→ message_delta（带 stop_reason 和用量）→ message_stop |
| 工具定义 | 包在 {"type": "function", "function": {name, description, parameters}} 里 | 更扁平，{name, description, input_schema} 定义一个工具 |
| 工具调用返回结果 | 返回的 message 里带 tool_calls，其中 arguments 是一个 JSON 字符串，需要自己再 parse 一次 | 返回的 content 里出现 tool_use 块，input 直接是 JSON 对象。 |
| 工具调用结果回传 | 工具结果用一条 role: "tool" 的消息回传，靠 tool_call_id 对应 | 工具结果放在一条 user 消息里，作为 tool_result 块，靠 tool_use_id 对应 |
| 字段不同 | 结束原因：`finish_reason`：`stop` / `length` / `tool_calls`，用量字段：`prompt_tokens` / `completion_tokens` | 结束原因：`stop_reason`：`end_turn` / `max_tokens` / `tool_use` / `stop_sequence`，用量字段：`input_tokens` / `output_tokens`|

其中，两者在系统提示词、工具调用、流式输出上的不同较大。 

对于 Anthropic 协议，顶层 system 字段可以是 text block 数组，存放多段系统提示词。但是如果在对话中途出现的 system 消息，OpenAI 允许在 messages 任意位置插入 role: "system" 的消息，对于 Anthropic 则需要特殊处理，常见处理有两种：

- 提到顶层合并。 这是 LiteLLM 这类转换网关的默认做法：把所有 system 消息抽出来拼进顶层 system。好处是简单，坏处有两个：一是丢失了位置语义，原本“第 20 轮之后才生效”的指令变成了从头就有；二是每出现一条新的中途 system，顶层 system 就变了，整个前缀缓存失效。
- 原地转成 user 消息。 把中途的系统指令包一层标签，塞进当前位置的 user 消息里，比如 <system-reminder>...</system-reminder>。Claude Code 就大量使用这种方式来注入动态提醒。它保留了位置语义，也不会动前缀。

对于两个协议在工具调用、流式输出上的不同，可以使用下面的请求是实验：

<table>
<tr><th>方面</th><th>OpenAI Chat Completions</th><th>Anthropic Messages</th></tr>
<tr>
<td>工具调用（模型发起调用）</td>
<td>

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1-mini",
    "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询某个城市的当前天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

</td>
<td>

```bash
curl https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-5-5",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
    "tools": [{
      "name": "get_weather",
      "description": "查询某个城市的当前天气",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }]
  }'
```
</td>
</tr>
<tr>
<td>工具调用（回传工具结果）</td>
<td>

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1-mini",
    "messages": [
      {"role": "user", "content": "北京现在天气怎么样？"},
      {"role": "assistant", "content": null, "tool_calls": [{
        "id": "call_demo123", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}
      }]},
      {"role": "tool", "tool_call_id": "call_demo123",
       "content": "{\"temp_c\": 22, \"condition\": \"晴\"}"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询某个城市的当前天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
      }
    }]
  }'
```

</td>
<td>

```bash
curl https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-5-5",
    "max_tokens": 512,
    "messages": [
      {"role": "user", "content": "北京现在天气怎么样？"},
      {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_demo123", "name": "get_weather", "input": {"city": "北京"}}
      ]},
      {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_demo123",
         "content": "{\"temp_c\": 22, \"condition\": \"晴\"}"}
      ]}
    ],
    "tools": [{
      "name": "get_weather",
      "description": "查询某个城市的当前天气",
      "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }]
  }'
```
</td>
</tr>
<tr>
<td>流式输出</td>
<td>

```bash
curl -N https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1-mini",
    "stream": true,
    "messages": [{"role": "user", "content": "数到5，每个数字一行。"}]
  }'
```

</td>
<td>

```bash
curl -N https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-5-5",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "数到5，每个数字一行。"}]
  }'
```
</td>
</tr>
</table>

## 2 OpenAI Chat Completions API vs OpenAI Responses API

OpenAI 官方把 Responses (/v1/responses) 描述为 Chat Completions (/v1/chat/completions) 的演进版本，带来了更简洁的接口和面向 agent 的原语，并表示 Chat Completions 仍会继续支持，但所有新项目都推荐使用 Responses。

两者的区别有：

| 方面 | Chat Completions | Responses |
|---|---|---|
| 端点 | `POST /v1/chat/completions` | `POST /v1/responses` |
| 输入字段 | `messages`（消息数组） | `input`（字符串，或 item 数组） |
| 系统提示词 | `system` / `developer` 角色消息 | 顶层 `instructions`，或 `developer` 角色 item |
| 输出长度上限 | `max_tokens`（推理模型须用 `max_completion_tokens`） | `max_output_tokens` |
| 输出结构 | `choices[].message` | `output[]`，由多种类型的 item 平铺组成 |
| 多个候选 | 支持 `n` | 不支持 |
| 取文本 | `choices[0].message.content` | SDK 提供 `output_text`；原始 JSON 需在 `output` 中找 `message` item |
| 对话状态 | 无状态，每次传完整历史 | 可无状态，也可用 `previous_response_id` 或 Conversations API |
| 工具定义 | `{"type":"function","function":{name, parameters}}` | `{"type":"function","name":...,"parameters":...}` |
| strict 默认值（函数调用是否采用严格模式） | 关闭 | 开启 |
| 模型发起调用 | `message.tool_calls[]`，`finish_reason: "tool_calls"` | `function_call` item（带 `call_id`） |
| 回传工具结果 | `role: "tool"` 消息 + `tool_call_id` | `function_call_output` item + `call_id` |
| 推理强度 | `reasoning_effort` | `reasoning.effort` |
| 推理内容 | 无标准字段 | `reasoning` item，可选摘要和加密内容，能跨轮回传 |
| 结构化输出 | `response_format` | `text.format` |
| 流式 | `data:` chunk + `delta`，以 `[DONE]` 结束 | 类型化事件，如 `response.output_text.delta` |
| 内置工具 | 无 | 网页搜索、文件搜索、远程 MCP 等 |
| 用量字段 | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` |


## 3 考虑 KV cache 下的协议使用

了解大模型的人都知道，KV cache 的复用能提升推理速度和吞吐。上述 API request 中的 JSON 最终都会被 chat template 渲染成一条线性的 token 序列，而 KV cache 只认这条序列的前缀看是否能够复用，考虑到这一点，我们看一下在处理 system、tools 以及复制历史对话时要注意什么。

### 3.1 chat template 里，system 和 tools 放在哪

主流做法几乎一致：tools 和 system 都放在序列最开头，通常 tools 会被直接并进 system 区块。以 Qwen 系列的模板为例，一次带工具的请求渲染出来大致是：

```
<|im_start|>system
你是一个编程助手。

# Tools
You may call one or more functions...
<tools>
{"type": "function", "function": {"name": "get_weather", ...}}
</tools>
...调用格式说明...<|im_end|>
<|im_start|>user
北京现在天气怎么样？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp_c": 22}
</tool_response><|im_end|>
<|im_start|>assistant
```

可以看到：工具定义序列化成 JSON 文本，追加在 system 内容后面。Anthropic 闭源，看不到模板，但官方文档里写明了缓存前缀的顺序是 tools → system → messages。

### 3.2 从 KV cache 复用角度推出一些实用原则

前缀缓存的规则很简单：从第一个 token 开始逐个比对，遇到第一个不同的 token，从那里往后全部重算。在 chat template 中，tools 和 system 放在最前面，后面是历史信息。有些模型 template 支持中途 system，有些会忽略。

由此我们可以推出一些实践原则：
1. 不要在 system 里放动态内容，system 保持稳定
2. 工具列表保持稳定、有序。 中途增删工具、调整顺序、修改描述，都会导致全量重算。
3. 序列化要确定。 同样的工具，如果 JSON key 的顺序不同、空格不同，渲染出来就是不同的 token。自己拼请求或写网关时，要保证序列化结果每次一致
4. 历史消息尽量不要被改写。

### 3.3 如何践行实践原则

#### 3.3.1 对于 Anthropic 协议，对于中途 system，推荐原地转成 user 信息

对于 Anthropic 协议，把中途 system 提到顶层合并，等于每次都在改前缀，是缓存不友好的做法，推荐使用原地转成 user 消息。

#### 3.3.2 支持按需加载工具

尽量保证对话过程中，工具列表稳定、有序，将全部工具直接放进 `tools` 参数最简单。

如果中途增删工具、修改描述怎么处理？答案是支持按需加载工具。

最典型的就是 Anthropic API 的 tool search tool 配合 defer_loading：

在工具列表里放一个 tool search 工具，其他工具全部标记 defer_loading: true，这样 Claude 一开始只能看到搜索工具。带 defer_loading: true 的工具在计算缓存 key 之前就会从渲染后的 tools 区块里剥离，根本不出现在 system prompt 前缀中；当搜索命中某个延迟工具并返回 tool_reference 时，这个工具的完整定义会在对话正文的那个位置内联展开（定义被渲染在对应 tool_result 的位置），而不是插进前缀。

自己部署模型时，可以这样做：

tools 参数里永远只放两个固定的“元工具”：
- search_tools(query)：搜索有哪些工具可用，返回工具的说明书（名字、描述、参数 schema）；
- call_tool(name, arguments)：调用某个工具，name 是要调用的工具名，arguments 是传给它的参数。

真正干活的那 200 个工具（get_weather、send_email 等）从来不出现在 tools 参数里。模型通过 search_tools 读到它们的说明书，再通过 call_tool 间接调用它们。你的代码收到 call_tool 后，根据 name 把请求转发给真正的实现，这就是“调度层按 name 路由”的意思。

`tools` 参数（每次请求都完全相同）：

```json
"tools": [
  {
    "type": "function",
    "function": {
      "name": "search_tools",
      "description": "按关键词搜索可用工具，返回工具名、用途和参数格式。调用任何工具前必须先搜索。",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "call_tool",
      "description": "调用通过 search_tools 找到的工具。name 为工具名，arguments 为符合该工具参数格式的对象。",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"]
      }
    }
  }
]
```

然后对话逐步变成这样（OpenAI Chat Completions 格式）：

```json
"messages": [
  {"role": "user", "content": "帮我查一下北京的天气"},

  {"role": "assistant", "content": null, "tool_calls": [{
    "id": "call_1", "type": "function",
    "function": {"name": "search_tools", "arguments": "{\"query\": \"天气\"}"}
  }]},
  {"role": "tool", "tool_call_id": "call_1",
   "content": "[{\"name\": \"get_weather\", \"description\": \"查询城市当前天气\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]"},

  {"role": "assistant", "content": null, "tool_calls": [{
    "id": "call_2", "type": "function",
    "function": {"name": "call_tool",
                 "arguments": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"}
  }]},
  {"role": "tool", "tool_call_id": "call_2",
   "content": "{\"temp_c\": 22, \"condition\": \"晴\"}"}
]
```

最后模型回答“北京现在 22 度，晴”。

你的应用收到模型的工具调用后，这样处理：

```python
REGISTRY = {
    "get_weather": {"schema": {...}, "fn": get_weather_impl},
    "send_email":  {"schema": {...}, "fn": send_email_impl},
    # …… 200 个工具
}

def handle_tool_call(name, args):
    if name == "search_tools":
        hits = search_registry(args["query"])          # 关键词或 embedding 检索
        return json.dumps(
            [{"name": n, **REGISTRY[n]["schema"]} for n in hits],
            ensure_ascii=False,
        )

    if name == "call_tool":
        target, inner_args = args["name"], args["arguments"]
        if target not in REGISTRY:
            return f"错误：没有名为 {target} 的工具，请先用 search_tools 搜索。"
        err = validate(inner_args, REGISTRY[target]["schema"])   # 自己做 schema 校验
        if err:
            return f"参数错误：{err}，请按工具的参数格式重试。"
        return json.dumps(REGISTRY[target]["fn"](**inner_args), ensure_ascii=False)
```

这个方案也有几个需要接受的缺点：

1. **参数没有约束解码保护。** `call_tool` 的 `arguments` 声明的是任意 object，模型生成时不会被 `get_weather` 的 schema 约束，所以必须像上面代码那样自己校验，并把错误信息返回给模型让它重试。相应地，`call_tool` 不能开 strict 模式（strict 要求 `additionalProperties: false`，无法表达任意 object），如果一定要开，可以把 `arguments` 声明成 JSON 字符串，再自己解析。
2. **模型要多走一步，并且“记住”说明书。** 每个新工具都要先搜索再调用，多一轮请求；而且说明书在对话中部，对话很长时，模型可能记不清参数格式。
3. **略偏离训练分布。** 模型训练时见惯了直接调用工具，这种两层包装的方式不太常见。能力强的模型通常没问题，弱一些的模型可能会忘记先搜索，或者把参数嵌套错。在元工具的 `description` 里把用法写清楚会有很大帮助。

所以总的来说：工具数量不多时，直接全部放进 `tools` 参数最简单。


#### 3.3.3 保证序列化一致

如何保证工具的序列化一致？首先要明确进到 template 前序列化在哪里发生，然后盯以下几处：

- 工具列表的顺序。最简单的办法是发请求前按名字排序。
- 每个工具内部 key 的顺序。 确保 schema 来源是确定的：Python 3.7+ 的 dict 和 JS 的对象都保留插入顺序，只要构造过程确定就没问题；JS 中整数形式的 key 会被自动提前，要留意。如果没法控制来源，可以统一做一次规范化（递归排序 key），只要每次都这样做，结果就是一致的。
- 中文转义。 ensure_ascii=True 会把“北京”变成 \u5317\u4eac，token 完全不同。在自己拼字符串（比如方案 A 里把 schema 写进 tool result）的地方要统一设置。
- 历史工具调用的参数不要重新序列化。 OpenAI 格式里 arguments 是字符串，模板一般会原样输出。拿到模型生成的 arguments 后，回传时要原样放回，不要 json.loads 再 json.dumps，否则空格和 key 顺序可能和模型当初生成的不一致，会从这一轮开始缓存失效。
- 内容本身不能有动态值。 工具描述里别出现时间、版本号、计数这类每次都会变的东西。


#### 3.3.4 是否删掉历史轮次的思考内容

有些模型比如 Qwen3、deepseek 需要删除历史轮次的思考内容（它不是删除所有历史 assistant 轮的思考，而是以最后一条 user 消息为界。在这条 user 消息之后的 assistant 轮会保留思考，之前的才删除。这样设计是因为，在一次工具调用循环中，模型需要看到自己之前的推理才能接着做下去）。一旦新的 user message 出现，思考都会被删掉，与历史消息不同，这一整段工具调用历史（工具返回的文件内容、搜索结果等往往非常长）都要重算。

如果你自己部署这类模型做 agent，可以考虑两个方向：
- 保留全部思考。 改模板或者使用模型提供的选项，让历史轮的思考始终保留。序列只追加、不改写，缓存命中最好，模型也能看到完整的推理链。代价是上下文增长得更快。较新的、面向 agent 的模型越来越倾向于这个方向：Anthropic 也要求在工具调用循环中把 thinking 块原样回传，较新的 Claude 模型默认会在上下文里保留历史轮的思考块。
- 一直删除。 上下文更省，但要接受每个 user 回合开始时重算一次上个回合的工具循环。如果单个回合内的工具步数不多，这个代价是可以接受的。











