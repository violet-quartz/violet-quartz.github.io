---
title: 'Pi Agent Harness 介绍'
description: 'Pi 是一个极简的 Agent Harness 框架，按照分层 pi-ai -> pi-agent-core -> pi-coding-agent 循序渐进，本文我们结合源码探讨了 Pi 中的消息与工具协议的统一、Agent 循环的核心实现、会话的持久化设计与压缩机制以及Extension 扩展系统的相关问题'
pubDate: '2026-10-03'
tags: ['agent', 'infra', 'pi', 'agent-harness']
---

[pi](https://github.com/earendil-works/pi) 是一个极简的 Agent Harness 框架。可以阅读 [How to Build a Custom Agent Framework with PI: The Agent Stack Powering OpenClaw](https://gist.github.com/dabit3/e97dbfe71298b1df4d36542aceb5f158) 这篇社区教程，按照分层 pi-ai -> pi-agent-core -> pi-coding-agent -> pi-tui 循序渐进，了解如何使用。

下面我们结合源码，对 Pi Agent Harness 实现的一些重点问题进行探讨。

## 1 如何在本地调试 pi

环境准备：

```bash
npm install --ignore-scripts
npm run hydrate:model-data      # 首次运行需要；以后想更新模型列表时再跑
./pi-test.sh # 可以本地跑一个命令行窗口
```

通过命令行 debug：

```bash
cd packages/coding-agent
node --import ./src/experimental/source-resolver.ts examples/sdk/my-example.ts # 运行自己写的 my-example.ts
node --import --inspect-brk ./src/experimental/source-resolver.ts examples/sdk/my-example.ts # debug 
```

也可以通过配置使用 vscode 进行 debug，下述示例为对 cli “ls 一下” 进行 debug 的配置。
```launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "type": "node",
            "request": "launch",
            "name": "Debug pi",
            "runtimeArgs": [
                "--stack-trace-limit=100",
                "--import",
                "${workspaceFolder}/packages/coding-agent/src/experimental/source-resolver.ts"
            ],
            "program": "${workspaceFolder}/packages/coding-agent/src/experimental/cli.ts",
            "args": ["-p", "--mode", "json", "ls 一下"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "skipFiles": ["<node_internals>/**"]
        },
        {
            "type": "node",
            "request": "launch",
            "name": "Debug pi (to file)",
            "runtimeExecutable": "/bin/sh",
            "runtimeArgs": [
                "-c",
                "node --stack-trace-limit=100 --import ./packages/coding-agent/src/experimental/source-resolver.ts ./packages/coding-agent/src/experimental/cli.ts -p --mode json 'ls 一下' > out.jsonl 2> err.log"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "skipFiles": ["<node_internals>/**"]
        }
    ]
}
```

## 2 消息与工具协议的统一

不同的协议（比如 OpenAI Chat Completion 和 Anthropic Messages）表示方式会有不同，pi 的做法是在 packages/ai 中定义一套自己的中间协议。上层只使用这套协议，各家 API 的差异全部由适配器处理。适配器负责把统一格式转换成该家 API 的请求，再把流式响应转换回统一事件。

```
coding-agent  AgentMessage（agent 包定义，coding-agent 扩展了具体类型）  ─convertToLlm→
pi-ai         Message（统一协议）               ─transformMessages + 适配器→
provider      Anthropic / OpenAI / Google / Bedrock …… 的原生格式
```

### 2.1 pi-ai 中统一的中间协议

**统一的消息格式**
```typescript
type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

// assistant 消息的内容是一个块数组，文本、思考、工具调用都是平级的块：

interface AssistantMessage {
  role: "assistant";
  content: (TextContent | ThinkingContent | ToolCall)[];
  api: Api; provider: string; model: string;   // 记录这条消息是哪个模型生成的
  usage: Usage;                                // 统一的 token 用量和费用
  stopReason: "stop" | "length" | "toolUse" | "error" | "aborted" | ...;
}

interface ToolCall { type: "toolCall"; id: string; name: string; arguments: JsonObject }

interface ToolResultMessage {
  role: "toolResult"; toolCallId: string; toolName: string;
  content: (TextContent | ImageContent)[];   // 给模型看的内容
  details?: JsonValue;                       // 给程序用的结构化数据，不发给模型
  isError: boolean;
}
```

**统一的工具定义**

工具的参数用 TypeBox 定义，它本身就会生成标准的 JSON Schema：

```typescript
interface Tool { name: string; description: string; parameters: TSchema }
```

**统一的流式事件**

各家流式协议的格式五花八门，适配器统一转换成同一组事件（types.ts:732）：

```
start
→ text_start / text_delta / text_end
→ thinking_start / thinking_delta / thinking_end
→ toolcall_start / toolcall_delta / toolcall_end
→ done | error
```

每个事件都带着当前的不完整 assistant 消息（partial），UI 直接渲染它即可。工具参数在流式传输时只是一段不完整的 JSON，适配器用 parseStreamingJson 尽力解析（utils/json-parse.ts）。

### 2.2 跨模型切换：transformMessages

有了统一格式，就能在一个会话中途换模型：先用 Claude 写代码，再切到 GPT 审查，历史记录直接复用。但不同模型之间有些内容不能直接通用。每个适配器在发送请求前都会调用 transformMessages（api/transform-messages.ts），它根据每条消息记录的来源决定怎么处理：

| 情况 | 处理方式 | 原因 |
|---|---|---|
| 思考块来自同一个模型 | 原样保留，包括签名 | 有些 API 要求把签名传回去，才能延续推理 |
| 思考块来自其他模型 | 转换成普通文本 | 签名只对生成它的模型有效 |
| 被安全过滤的思考块（加密内容）来自其他模型 | 丢弃 | 其他模型无法解读 |
| 工具调用 id 格式不兼容 | 重写 id，并同步修改对应的 toolResult | 例如 OpenAI Responses 的 id 有几百个字符，Anthropic 要求不超过 64 个字符 |
| 当前模型不支持图片 | 替换成占位文本 | 避免请求被拒绝 |
| 报错或被中断的 assistant 消息 | 跳过 | 内容不完整，重放可能导致 API 报错 |
| 工具调用没有对应的结果 | 补一个 `isError` 的结果 | 多数 API 要求每个工具调用都必须有结果 |

这一层使得会话历史和具体模型解耦：会话里存的是 pi 的统一格式，只在每次发送请求时，按当前目标模型转换。

### 2.3 应用层的扩展：AgentMessage 和 convertToLlm

应用层还需要一些模型不认识的消息类型。pi 没有把它们硬塞进 LLM 协议，而是在 agent 包里留了一个扩展点：AgentMessage 是标准 Message 和一个默认为空的 CustomAgentMessages 接口的联合类型（agent/src/types.ts:361-370）。agent 循环内部始终使用 AgentMessage，只在每次请求模型前调用 convertToLlm，把它转换成标准 Message。默认实现会直接丢弃不认识的类型。

coding-agent 通过声明合并，往 CustomAgentMessages 里加入了压缩摘要、分支摘要、! 执行的 bash 命令和 extension 的自定义消息（coding-agent/src/core/messages.ts:70），并提供了自己的 convertToLlm，把它们转换成标准消息：

```
compactionSummary → user："以下是之前对话的摘要……"
branchSummary     → user
bashExecution     → user（命令和输出）
custom            → user
system/user/assistant/toolResult → 原样保留
```


## 3 Agent 循环的核心实现

pi 的 agent 循环核心在 packages/agent/src/agent-loop.ts，入口是 runLoop。

### 3.1 最小模型

```
用户消息 → [调用 LLM → 有工具调用？→ 执行工具 → 结果追加到上下文] ↺ → 没有工具调用，结束
```

一次"调用 LLM + 执行它请求的工具"称为一个 turn。一次用户输入可能经历多个 turn。

### 3.2 双层循环

```
  while (true) {                                    // 外层：处理 follow-up 消息
    while (hasMoreToolCalls || pendingMessages.length > 0) {   // 内层：一个个 turn
      // 1. 把排队的消息（steering）追加到上下文
      // 2. prepareRequest：请求前的钩子（pi 在这里检查是否需要压缩）
      // 3. streamAssistantResponse：流式调用 LLM
      // 4. 出错或被中止 → 直接结束
      // 5. 有 toolCall → executeToolCalls，结果追加到上下文，hasMoreToolCalls = true
      // 6. finishTurn 钩子，然后读取新的 steering 消息
    }
    // 模型不再调用工具时：如果有 follow-up 消息就 continue，否则 break
  }
```

两种用户消息的区别：
- steering：在 agent 运行过程中插入的消息。它会在下一个 turn 开始前注入上下文，用来中途纠正方向。
- follow-up：等 agent 自然停下后才处理，相当于排队的下一条指令。

### 3.3 事件驱动：循环和外部系统解耦

循环本身不做持久化、不渲染 UI、也不做压缩。它只通过 emit 发出事件：

```
  agent_start → turn_start → message_start → message_update* → message_end
              → tool_execution_start/end → turn_end → … → agent_end
```

外部系统订阅这些事件来完成各自的工作：
- AgentSession 在收到 message_end 时调用 sessionManager.appendMessage，这就是前面讲的持久化挂点。
- TUI 根据 message_update 实时渲染。

需要影响循环行为的地方，则通过 AgentLoopConfig 里的钩子注入：prepareRequest、prepareNextTurn、finishTurn、getSteeringMessages、getFollowUpMessages、beforeToolCall、afterToolCall。agent-loop.ts 只定义"什么时候"可以介入，pi 的产品逻辑（压缩、会话投影、扩展、权限）通过这些钩子决定"做什么"。

## 4 会话的持久化设计与压缩机制

从打开一个 cli 到退出是一个会话（session），每个会话会存成一个 JSONL 文件，只追加写入。 可以使用如下命令，去看真实的会话文件：

```bash
./pi-test.sh  # 打开一个 cli，开始会话，会话文件默认存储到 ~/.pi/agent/sessions/--your_path--/
./pi-test.sh --session-dir /tmp/pi-sessions # 打开一个 cli，开始会话，指定会话文件存储到 /tmp/pi-sessions
```

给一个会话文件的示例：
```
{"type":"session","version":3,"id":"01a0e80e-7309-702d-8c34-36674ca667e1","timestamp":"2026-10-02T10:00:00.000Z","cwd":"/Users/me/demo"}
{"type":"model_change","id":"a1","parentId":null,"timestamp":"2026-10-02T10:00:00.100Z","provider":"anthropic","modelId":"claude-sonnet-5-5"}
{"type":"message","id":"b2","parentId":"a1","timestamp":"2026-10-02T10:00:01.000Z","message":{"role":"user","content[{"type":"text","text":"这个项目怎么跑测试？"}]}}
{"type":"message","id":"c3","parentId":"b2","timestamp":"2026-10-02T10:00:03.000Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_1","name":"read","arguments":{"path":"package.json"}}],"stopReason":"toolUse"}}
{"type":"message","id":"d4","parentId":"c3","timestamp":"2026-10-02T10:00:03.100Z","message":{"role":"toolResult","toolCallId":"call_1","toolName":"read","content":[{"type":"text","text":"{\"scripts\":{\"test\":\"vitest\"}}"}],"isError":false}}
{"type":"message","id":"e5","parentId":"d4","timestamp":"2026-10-02T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"运行 npm test 即可。"}],"stopReason":"stop"}}
{"type":"message","id":"f6","parentId":"e5","timestamp":"2026-10-02T10:01:00.000Z","message":{"role":"user","content":[{"type":"text","text":"怎么只跑一个文件？"}]}}
{"type":"message","id":"g7","parentId":"f6","timestamp":"2026-10-02T10:01:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"npx vitest run path/to/file.test.ts"}],"stopReason":"stop"}}
{"type":"message","id":"h8","parentId":"e5","timestamp":"2026-10-02T10:02:00.000Z","message":{"role":"user","content":[{"type":"text","text":"帮我加一个 lint 脚本"}]}}
{"type":"message","id":"i9","parentId":"h8","timestamp":"2026-10-02T10:02:04.000Z","message":{"role":"assistant","content":[{"type":"text","text":"已在 package.json 中添加 \"lint\": \"biome check .\"。"}],"stopReason":"stop"}}
{"type":"compaction","id":"j10","parentId":"i9","timestamp":"2026-10-02T10:03:00.000Z","summary":"## Goal\n了解测试方式并添加 lint脚本。\n## Progress\n- 测试命令为 npm test (vitest)","firstKeptEntryId":"h8","tokensBefore":5321}
{"type":"message","id":"k11","parentId":"j10","timestamp":"2026-10-02T10:03:10.000Z","message":{"role":"user","content":[{"type":"text","text":"再加一个 format 脚本"}]}}
```

**格式**：第一行是 SessionHeader（type: "session"，包含 id、cwd、version=3），后面每行是一个 entry。
**树结构**：每个 entry 都有 id 和 parentId。SessionManager 维护一个 leafId 指针，新 entry 的 parent 就是当前 leaf。分支时只需移动 leaf 指针，旧路径仍留在文件里。所以一个文件里存的是一棵树，而不是一条线性记录。

```
  root(user) ─ a(assistant) ─ b(user) ─ c(assistant)     <- 旧分支
                           └ b'(branch_summary) ─ d(user)  <- 当前 leaf
```

**分支**：可以通过 /tree 命令回到之前某节点，如上示例文件所示，f6 和 h8 的 parentId 都是 e5。用户先问了"怎么只跑一个文件"（f6 → g7），然后用 /tree 回到 e5，改问"加一个 lint脚本"（h8）。旧分支没有被删除，仍留在文件里，用 /tree 切换分支时，Pi 会询问是否要总结你即将离开的那条分支，只有当你选择总结时，才会生成 BranchSummaryEntry 并挂到新分支上；不总结就直接切过去，不产生任何摘要。

```
  a1 ─ b2 ─ c3 ─ d4 ─ e5 ─┬ f6 ─ g7                    (旧分支)
                          └ h8 ─ i9 ─ j10 ─ k11         (当前分支)
```

**压缩**：压缩可以通过 /compact 命令手动触发，当上下文过长以及 provider 返回上下文溢出错误后都会触发压缩。

压缩不删除也不改写旧 entry，而是在当前 leaf 后追加一个 compaction entry

  { type: "compaction", summary, firstKeptEntryId, tokensBefore, details?, systemMessage? }

其中 summary 为对压缩部分的总结，压缩的内容是从上一次压缩的 entry id 到 firstKeptEntryId（不包含），从 firstKeptEntryId 开始是保留的原始信息。

如上示例文件所示，j10 是一个 compaction entry。它不修改旧记录，只追加一条摘要 summary，并用 firstKeptEntryId: "h8" 标出从哪里开始保留原文。tokensBefore 记录压缩前的上下文大小。

模型实际看到的内容：pi 从最后一个 entry k11 沿 parentId 往上走，得到当前路径，再按最新的 compaction 裁剪：

  [compactionSummary] ## Goal 了解测试方式并添加 lint 脚本 ...
  [user]      帮我加一个 lint 脚本        (h8，从 firstKeptEntryId 开始保留)
  [assistant] 已在 package.json 中添加 ... (i9)
  [user]      再加一个 format 脚本        (k11)

压缩只作用于当前分支。压缩的算法可以参考 packages/coding-agent/docs/compaction.md，大致如下：
1. 找切割点：从投影后的上下文从后往前走，累计 token 估算，直到凑够 keepRecentTokens（默认 20k）。切割点之前的是                     
    messagesToSummarize，之后的是 kept messages。                                                                                    
2. 提取消息：从上一个保留边界（或 session 起点）到切割点，收集待压缩消息。                                                          
3. 生成摘要：调用 LLM，用结构化格式生成摘要；若已有上次摘要，作为迭代上下文传入。                                                   
4. 追加 entry：写入 CompactionEntry，含 summary 和 firstKeptEntryId（保留区的起点 ID）


**Entry 类型**：message、model_change、thinking_level_change、compaction、branch_summary、custom（扩展状态，不进入 LLM
  上下文）、custom_message（会进入上下文）、context_edit、label、session_info、usage。

**恢复上下文**：从 leaf 沿 parentId 一直走到 root，得到当前路径。再按压缩规则裁剪，然后把每个 entry 转换成 LLM 消息，同时从路径上还原 model 和 thinkingLevel。


## 5 Extension 扩展系统

Pi 的 extension 是一个 TypeScript 模块，和 pi 跑在同一个进程里，用来给 pi 增加可执行的行为：工具、命令、事件拦截、模型 provider、会话状态和终端 UI。

**最小示例**

一个 extension 只需默认导出一个工厂函数，函数参数是 ExtensionAPI：

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
pi.registerCommand("hello", {
    description: "Show a greeting",
    handler: async (name, ctx) => ctx.ui.notify(`Hello, ${name || "world"}!`, "info"),
});
}
```

ExtensionAPI 的能力：

| 能力 | API |
  |---|---|
  | 订阅或拦截生命周期事件 | `pi.on(event, handler)` |
  | 注册模型可调用的工具 | `pi.registerTool()` |
  | 注册 `/` 命令、快捷键、CLI 参数 | `pi.registerCommand()`、`pi.registerShortcut()`、`pi.registerFlag()` |
  | 发送用户消息或自定义消息 | `pi.sendUserMessage()`、`pi.sendMessage()` |
  | 持久化数据但不进入上下文 | `pi.appendEntry()` |
  | 切换工具、模型、思考级别 | `pi.setActiveTools()`、`pi.setModel()`、`pi.setThinkingLevel()` |
  | 接入新的模型 provider | `pi.registerProvider()` |
  | 自定义渲染和 UI | `pi.registerMessageRenderer()`、`pi.registerEntryRenderer()`、`ctx.ui` |
  | extension 之间通信 | `pi.events` |


可以通过看 packages/coding-agent/examples/extensions/ 中的例子，来理解 extension，比如其中的：
- **todo.ts**: 注册的是一个工具。它和内置的 read、bash 一样，作为 agent 循环里的一个普通工具运行，由模型决定什么时候调用。同时也注册一个命令。完成的是 todo 工具的功能，长任务里，模型容易忘记计划，或者做到一半偏离方向，使用 todo 工具能避免这一点
- **custom-compaction.ts**：通过 session_before_compact 事件替换默认的压缩摘要

extension 在函数里登记两类东西：
  - 能力：工具、命令、快捷键、provider 等，pi 把它们合并进自己的对应列表。
  - 事件处理函数：通过 pi.on(事件名, 函数) 登记，存进一个按事件名分组的 Map。

此后，extension 只在被调用时执行：要么模型调用了它注册的工具，要么用户执行了它注册的命令，要么 pi 发出了它订阅的事件。

事件是 pi 预留给 extension 的介入点。pi 运行到关键节点时，会按加载顺序依次调用并等待订阅了该事件的处理函数，然后根据返回值决定下一步怎么做。

下面是 custom-compaction 起作用的例子：

```typescript
// 1. pi 先自己计算切点（这一步 extension 无法跳过）
const preparation = prepareCompaction(pathEntries, settings);

// 2. 如果有 extension 订阅了这个事件，先把 preparation 交给它
if (runner.hasHandlers("session_before_compact")) {
const result = await runner.emit({ type: "session_before_compact", preparation, branchEntries, reason, signal, ... });
if (result?.cancel) throw new Error("Compaction cancelled");   // 取消压缩
if (result?.compaction) extensionCompaction = result.compaction;  // 使用 extension 的结果
}

// 3. 决定用哪份摘要
if (extensionCompaction) {          // 如果 extension 返回了摘要
summary = extensionCompaction.summary; ...        // 使用 extension 的摘要
} else {
const result = await this._runDefaultCompaction(...);   // 使用默认摘要
}

// 4. 两种情况都一样写入会话
sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore, details, fromExtension, usage);
```

多个 extension 同时订阅时：按加载顺序依次调用，最后一个有返回值的结果生效。如果某个处理函数返回了 cancel，会立即停止，不再调用后面的处理函数。