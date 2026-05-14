---
title: OpenClaw架构
date: 2026-05-14 12:00:00 +0800
categories: [Agent]
tags: [OpenClaw]
toc: true
layout: post
---

# OpenClaw是什么

相信大家已经在日常生活中经常使用豆包等AI工具来查资料和手机信息。这些工具确实提高了我们的资料收集和整合的速度。然而，我们发现我们获得了这些信息之后仍然需要自己按照AI给出的操作去一步一步的进行修改，AI充当的仍然只是一个更加聪明的检索系统的这样一个角色。这距离AGI的差距仍然很大。于是人们就想，那么我们是否可以让AI来帮助我们直接处理一些事情呢？不是只是给出一些指导和建议，而是输入命令之后直接能帮我们完成整件事情，就像漫威电影中的钢铁侠一样。于是，基于这样的动机，OpenClaw诞生了。**OpenClaw 是一个运行在你自己电脑上的 AI 助手平台。**它能够真实地帮你写邮件，改代码 — 做一些能够真真实实地改变设备上资源的一些事情，而不再仅仅是一个chatbot。

# OpenClaw特点

那么就有小伙伴会问了，之前Anthropic公司的的Claude Code，国内字节的TRAE，以及阿里的通义灵码不也能写代码，查资料吗，OpenClaw功能也差不多，为什么这么火呢？在我看来，OpenClaw的巨大优势主要体现在以下几点上：

1. **让不懂技术的人也能使用高级AI：**过去你要用 高级一点能做事的，像Claude Code一样的AI，你得了解CLI怎么使用以及这款工具的一些入门使用技巧。这对于非程序员出生的人们是十分困难的。而OpenClaw你只需要和它聊天即可使用，对用户的要求仍然像使用豆包一样低，而能做的事大大地增加了。
2. **让AI的使用便捷化：**同时，你只能在电脑上用，一旦你的电脑不在你的身边，你对于AI的使用权力又变回了只能使用豆包这样的chatbot。而OpenClaw允许你在飞书、Telegram、Slack等软件上通过聊天的方式使用AI来处理事务，极大地提高了AI在各种环境下的便捷性，让更多的人用日常生活中更加习惯的方式使用AI。

# 架构

但是OpenClaw也不是魔法。它的本质也可以说是是Harness Engineering一种体现—给大模型这个大脑装上的一副手脚架和方向盘，让大模型的能力能够更好的发挥。接下来，我们先从OpenClaw的架构谈起，看看OpenClaw长什么样。
我们先给出OpenClaw的一个大致框架图：
![image-20260514102211964](D:\Typora\imgs\image-20260514102211964.png)

大致来说，我们在CLI/WebChat UI/飞书等消息通道发送消息，消息被传入到Gatway进行分发，之后分发到合适的Agent进行处理，Agent可以操作设备上的工具来进行处理，之后将最终结果/需用户确认的提示返回给用户进行进一步处理或者结束。因此，整个架构我们需要重点了解的是Gateway和Agent Runtime。

## 控制平面-Gateway

Gateway（控制平面）可以理解成OpenClaw的前台，它是连接消息通道和后端Agent系统的中心枢纽。任何消息都必须要经过Gateway的检查和分发才能进入特定的Agent中进行处理。就像一个公司要经过了大门口安检的检查才能进入公司的某些特定区域一样。Gateway是所有消息进和出的唯一通道。
![image-20260514104147808](D:\Typora\imgs\image-20260514104147808.png)

Gateway默认跑在本机127.0.0.1:18789端口上，所有的客户端和Agent Runtime都以客户端的身份回连。每条消息通过**通道适配器**传入Gateway之后，Gateway主要做这五件事：**消息格式转换、路由、鉴权/权限控制、广播、断点恢复**。

### 消息格式转换-通道适配器

通道适配器可以理解为一个**万能插头**：它处理各个平台特有的消息格式，把平台特有消息转换成 OpenClaw 内部统一的 `InboundMessage` 对象供Agent Runtime处理；也把Agent处理结果`OutboundMessage` 对象转换成各个平台特有的消息格式供平台返回输出。OpenClaw内部不再关心平台层面的差异，Agent不会因为连接平台的变化而变化，只专注负责消息的处理逻辑。因此，OpenClaw的核心逻辑只有一套(一套核心逻辑+N套适配器)。

OpenClaw中定义了一个 **Protocol（协议）** — **`ChannelPlugin`** 来确保收到的消息格式能够被成功地转换。Gateway只关心是否实现了这个协议，而不关心某一个具体的平台。该协议要求要接入OpenClaw的所有聊天平台必须实现这个统一接口。我们可以将协议内容概括如下：

| 字段名             | 类型 / 返回值         | 说明                                                         |
| ------------------ | --------------------- | ------------------------------------------------------------ |
| `id`               | `str`                 | 渠道唯一标识，如 `"WhatsApp"`                                |
| `meta`             | `ChannelMeta`         | 显示名 / 文档路径 / 是否支持 markdown                        |
| `capabilities`     | `ChannelCapabilities` | 能力清单：支持哪些聊天类型（dm/group）、是否能编辑、是否阻塞流式 |
| `list_account_ids` | `list[str]`           | 从配置里读出该渠道有几个账号（支持多账号）                   |
| `start`            | `async None`          | 启动这个账号的入站流（连 WebSocket / 起 webhook 服务）       |
| `stop`             | `async None`          | 停止入站流，清理连接                                         |
| `send`             | `async None`          | 把统一的 `OutboundMessage` 翻译成平台 API 格式发出去         |

通道适配器的实现有三种方式，每一种都实现了**`ChannelPlugin`**：

| 类型                                                | 特点                                                   | ChannelPlugin实现                                            |
| --------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Bot API<br />(Telegram、飞书、Slack、Discord、钉钉) | 用平台官方提供的机器人 API 实现，最稳定                | start() 里订阅/发起 WebSocket 事件流/ webhook 服务。send(） 里调官方 SDK 发消息。 |
| Bridge<br />(QQ、微信)                              | 通过桥接服务（比如 go-cqhttp、NapCat）间接接入，风险高 | start(）起一个独立子进程或 WebSocket 桥接器，适配器只负责跟它说话。send（）通过桥发消息。 |
| Native<br />(WhatsApp、iMessage、Matrix)            | OpenClaw自己实现的通道，直接模拟客户端协议接入         | start/stop/send 全 no-op——客户端直连 Gateway, Gateway 推 EventFrame 就是发消息。 |

我们现在知道了通道适配器需要遵守的协议和三种实现方式，最后，我们需要来看看在通道适配器中传输的内容`InboundMessage`和`OutboundMessage`。`InboundMessage`和`OutboundMessage`格式如下：

**`InboundMessage`**

| 字段名            | 示例值                                       | 说明                                                     |
| ----------------- | -------------------------------------------- | -------------------------------------------------------- |
| `channel`         | `"feishu"` | `"wechat"` | `"webchat"` \| ... | 来自哪个渠道（就是 `plugin.id`）                         |
| `account_id`      | `"app_xxx"` | `"tenant_xxx"`                 | 这条消息进的是哪个机器人 / 账号（一个 Agent 可挂多个号） |
| `sender_id`       | `"ou_a1b2c3..."` / `"wxid..."`               | 发送者在该渠道里的 ID（原样，不跨渠道映射）              |
| `conversation_id` | `"oc_xxxxx"` / `"wxid_room"`                 | 会话 ID — Runtime 用它隔离上下文 / 记忆                  |
| `text`            | `"帮我建个文档"`                             | 纯文本内容（渠道负责把富文本 / 卡片解平）                |
| `attachments`     | `[{kind, url, ...}, ...]`                    | 图片 / 文件 / 语音 — 统一格式                            |
| `thread_id`       | `"thr_xxx"` | `None`                         | 可选 — 如果这条消息属于某个 thread                       |
| `chat_type`       | `"dm"` | `"group"` | `"channel"`             | 私聊 / 群聊 / 频道 — 让 Runtime 判断要不要 @ 前缀等      |
| `raw`             | `{ ... 渠道原始 payload ... }`               | 保底逃生口                                               |

**`OutboundMessage`**

| 字段名         | 示例值                    | 说明                                         |
| -------------- | ------------------------- | -------------------------------------------- |
| `text`         | `"已建好，文档地址：..."` | 回复的纯文本内容                             |
| `media_urls`   | `[ url1, url2, ... ]`     | 要附带的图片 / 文件链接列表                  |
| `reply_to_id`  | `"msg_xxx"` | `None`      | 可选，引用回复某条消息（仅支持的渠道生效）   |
| `channel_data` | `{ "card": {...} }`       | 渠道特有扩展数据，比如飞书卡片、Slack Blocks |

#### 加入一个新平台

1. 新建一个文件夹 `channels/feishu/`
2. 写一个`ChannelPlugin`实现类`channels/feishu/plugin.py`
3. 在在 boot 里注册一行`channel_registry.register(FeiShuChannelPlugin())`

#### 一句话理解

**渠道适配器 = 翻译官。** 
它做两次翻译: 渠道原始格式 → `InboundMessage`(进), `OutboundMessage` →渠道原始格式(出)。 
Agent Runtime 看到的永远是`InboundMessage`和`OutboundMessage`这两个 Pydantic 模型而从来不关心连接了哪些平台。

### 路由

### 鉴权/权限控制

### 广播

### 断点恢复

## Agent Runtime

### Session模型

# 记忆

# 工具

# 安全

# 生态

# 完整端到端消息流
