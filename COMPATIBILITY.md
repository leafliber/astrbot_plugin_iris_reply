# iris_reply 与 iris_chat_memory 兼容性指南

## 架构关系

```
用户消息
  |
  v
on_message (iris_reply) ─── 信号门控评估，设置 event.set_extra("iris_mode")
on_all_message (iris_chat_memory) ─── 添加到 L1 buffer
  |
  v
[iris_reply 门控命中时]
handle_llm_request (iris_reply) ─── 统一决策 llm_generate (不触发 hooks)
  |  决策发言
  v
handle_llm_request (iris_chat_memory) ─── 清空 contexts, 注入 L1/L2/L3 (mark_as_temp)
handle_llm_request (iris_reply) ─── 注入发言提示 (mark_as_temp)
  |
  v
主 LLM 调用: system_prompt(人格) + L1(含bot回复) + L2/L3 + 发言提示
  |
  v
handle_llm_response (iris_chat_memory) ─── 添加 bot 回复到 L1
handle_llm_response (iris_reply) ─── 从 event 读取模式, 记录回复
  |
  v
on_message_sent (iris_reply) ─── 从 event 读取模式, 写入对话锚点

[定时器主动发起通路（initiate）]
ProactiveEngine 定时扫描 ─── 统一决策 llm_generate (不触发 hooks)
  |
  v
context.send_message 直发 ─── 不经过任何事件钩子
  |
  v
iris_reply 手动记账（入窗 / 锚点 / 接话 pending）
```

## iris_reply 侧

iris_reply 通过 `event.set_extra` 传递发言模式，不依赖共享状态变量，
与 iris_chat_memory 的 hooks 完全兼容，两个插件的 `on_llm_request` /
`on_llm_response` / `after_message_sent` 钩子互不干扰。

关键设计：
- `iris_mode` 取值：`chime_in`（跟话）/ `follow_up`（跟进）/ `passive`（被动回复），
  跟随 event 生命周期，无跨事件竞态
- 统一决策与被动评估的 `llm_generate` 直接调用 `prov.text_chat()`，不触发 hooks，
  不会触发 iris_chat_memory 的注入逻辑
- 所有注入内容均使用 `mark_as_temp()`，不污染对话历史
- **主动发起（initiate）为直发通路**：`context.send_message` 不经过事件管线，
  不触发 `on_llm_request` / `on_llm_response` / `after_message_sent` 任何钩子

## iris_chat_memory 侧建议修改

### 1. 检测 iris_reply 的主动发言（推荐）

iris_chat_memory 的 `on_llm_request` 钩子可以通过检测 `event.get_extra("iris_mode")`
来判断当前是否为 iris_reply 触发的主动发言，从而优化行为：

```python
# iris_chat_memory 的 on_llm_request 中
iris_mode = event.get_extra("iris_mode")
if iris_mode in ("chime_in", "follow_up"):
    # iris_reply 注入了发言提示，此处仍注入 L1/L2/L3
    # 但可以降低 L2/L3 的 token 预算，因为 iris_reply 的滑动窗口
    # 已在统一决策中提供了近期上下文
    pass
```

这不是必须的——即使不做此检测，两个插件也能正常协作。

### 2. 跳过冗余的被动触发检测（可选）

iris_chat_memory 的 `_detect_passive_trigger` 检查 `event.is_at_or_wake_command`，
而 iris_reply 在触发主动发言时也设置此标志。这不会产生冲突（iris_chat_memory 将其
视为"主动触发"而非"被动触发"是正确行为），但如果想更精确：

```python
# iris_chat_memory 的被动触发检测中
if event.get_extra("iris_mode") in ("chime_in", "follow_up", "passive"):
    # iris_reply 已处理触发逻辑，跳过 iris_chat_memory 的被动触发检测
    return False
```

### 3. 主动发起消息对 L1 不可见（重要，新增）

iris_reply 的主动发起（initiate）通过 `context.send_message` 直发，
**不会触发任何事件钩子**，因此 iris_chat_memory 的 L1 buffer 中不会出现
这类发起消息，只会在群友后续发言时间接体现。对绝大多数场景影响可忽略；
若需要完整记忆，可在 iris_chat_memory 侧通过平台层消息回执（如可用）补充。

### 4. L1 buffer 中 bot 回复的格式（重要，无需修改）

iris_chat_memory 的 `on_llm_response` 将 bot 回复添加到 L1 buffer，格式为
`[Bot昵称/时间]: 回复内容`。这正好解决了 iris_reply 的第三人称问题——LLM 能在
L1 上下文中看到自己的发言，维持自我身份认同。

**当前实现已正确，无需修改。**

## AstrBot 配置要求

### 必须配置：禁用内置 group_chat_context

当使用 iris_chat_memory 时，**必须禁用** AstrBot 内置的 `group_chat_context`，
否则会产生以下问题：

1. **重复注入**：group_chat_context 和 iris_chat_memory 的 L1 都注入群聊上下文
2. **第三人称问题**：group_chat_context 不记录 bot 回复，创造"观察者框架"
3. **历史膨胀**：group_chat_context 不使用 `mark_as_temp()`，上下文累积到历史

配置方法：在 AstrBot 管理面板中，将 `provider_ltm_settings` 下的
`group_message_max_cnt` 设为 0，或直接关闭群聊上下文功能。

### 不使用 iris_chat_memory 时

如果不安装 iris_chat_memory，iris_reply 仍能正常工作：
- 统一决策使用自己的 `SlidingWindow` + `ContextPackager` 构建上下文
- 发言提示已强化身份认同（"请保持你的人格，像平时在群里说话一样回复"）
- 但第三人称问题的修复效果有限——AstrBot 内置 group_chat_context 仍不包含
  bot 回复。建议在不使用 iris_chat_memory 时也考虑禁用 group_chat_context，
  或调低其 `group_message_max_cnt`

## 不兼容场景

| 场景 | 是否兼容 | 说明 |
|------|---------|------|
| 两插件同时安装 + 禁用 group_chat_context | 完全兼容 | 推荐配置 |
| 两插件同时安装 + 启用 group_chat_context | 部分兼容 | 重复注入 + 第三人称问题 |
| 仅 iris_reply + 启用 group_chat_context | 部分兼容 | 第三人称问题（提示修改后减轻） |
| 仅 iris_reply + 禁用 group_chat_context | 完全兼容 | 滑动窗口自供上下文 |
| 仅 iris_chat_memory | 完全兼容 | 与 iris_reply 无关 |
