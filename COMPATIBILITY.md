# iris_reply 与 iris_chat_memory 兼容性指南

## 架构关系

```
用户消息
  |
  v
on_message (iris_reply) ─── 评估触发，设置 event.set_extra("iris_mode")
on_all_message (iris_chat_memory) ─── 添加到 L1 buffer
  |
  v
[iris_reply 触发时]
handle_llm_request (iris_reply) ─── 触发判断 llm_generate (不触发 hooks)
  |  触发通过
  v
handle_llm_request (iris_chat_memory) ─── 清空 contexts, 注入 L1/L2/L3 (mark_as_temp)
handle_llm_request (iris_reply) ─── 注入主动提示 (mark_as_temp)
  |
  v
主 LLM 调用: system_prompt(人格) + L1(含bot回复) + L2/L3 + 主动提示
  |
  v
handle_llm_response (iris_chat_memory) ─── 添加 bot 回复到 L1
handle_llm_response (iris_reply) ─── 从 event 读取模式, 记录回复
  |
  v
on_message_sent (iris_reply) ─── 从 event 读取模式, 设置 follow-up
```

## iris_reply 侧（已完成）

重构后的 iris_reply 通过 `event.set_extra` 传递回复模式，不再依赖共享状态变量。
这确保了与 iris_chat_memory 的 hooks 完全兼容，两个插件的 `on_llm_request` /
`on_llm_response` / `after_message_sent` 钩子互不干扰。

关键设计：
- `llm_generate` 直接调用 `prov.text_chat()`，不触发 hooks，触发判断和被动评估
  的内部 LLM 调用不会触发 iris_chat_memory 的注入逻辑
- 所有注入内容均使用 `mark_as_temp()`，不污染对话历史
- `iris_mode` 通过 event 传递，跟随 event 生命周期，无跨事件竞态

## iris_chat_memory 侧建议修改

### 1. 检测 iris_reply 的主动回复（推荐）

iris_chat_memory 的 `on_llm_request` 钩子可以通过检测 `event.get_extra("iris_mode")`
来判断当前是否为 iris_reply 触发的主动回复，从而优化行为：

```python
# iris_chat_memory 的 on_llm_request 中
iris_mode = event.get_extra("iris_mode")
if iris_mode == "proactive":
    # iris_reply 注入了主动提示，此处仍注入 L1/L2/L3
    # 但可以降低 L2/L3 的 token 预算，因为 iris_reply 的滑动窗口
    # 已在触发判断中提供了近期上下文
    pass
```

这不是必须的——即使不做此检测，两个插件也能正常协作。

### 2. 跳过冗余的被动触发检测（可选）

iris_chat_memory 的 `_detect_passive_trigger` 检查 `event.is_at_or_wake_command`，
而 iris_reply 在触发主动回复时也设置此标志。这不会产生冲突（iris_chat_memory 将其
视为"主动触发"而非"被动触发"是正确行为），但如果想更精确：

```python
# iris_chat_memory 的被动触发检测中
if event.get_extra("iris_mode") in ("proactive", "passive"):
    # iris_reply 已处理触发逻辑，跳过 iris_chat_memory 的被动触发检测
    return False
```

### 3. L1 buffer 中 bot 回复的格式（重要，无需修改）

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
- 触发判断使用自己的 `SlidingWindow` + `ContextPackager` 构建上下文
- 主动提示已修改为强化身份认同（"请保持你的人格，像平时在群里说话一样回复"）
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
