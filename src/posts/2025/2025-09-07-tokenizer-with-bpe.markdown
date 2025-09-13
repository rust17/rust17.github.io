---
title: "使用 BPE 构建 Tokenizer"
layout: post
date: 2025-09-07 11:00
headerImage: false
tag:
- LLM
- BPE
- GPT
- CS336
- Language Modeling from Scratch
category: blog
hidden: false
author: circle
description: using bpe to build a tokenizer
---

### Tokenizer 的作用与意义

在[之前的文章](/2025/2025-08-29-build-bpe-from-scratch)中，我们实现了 BPE 算法并[进行了多进程优化](/2025/2025-09-06-parallel-bpe)。本文将在这些基础上，探讨如何构建一个完整的 Tokenizer 类。

在一次大模型的完整请求-响应流程中，Tokenizer 在前后两个关键环节发挥着核心作用：
- **编码（Encode）**：将自然语言文本转换为模型可理解的 Token ID 序列
- **解码（Decode）**：将模型生成的 Token ID 序列转换回人类可读的自然语言

本文主要探索如何实现这个两个方法：`encode`、`decode`。

### 编码（Encode）过程详解

编码阶段负责将输入文本转换为模型可处理的数字序列。以句子 "What's the weather<|endoftext|>like today?" 为例，其处理流程如下：

#### 1. 预处理与特殊标记识别
首先对输入文本进行初步分割，确保特殊标记（如 `<|endoftext|>`）被完整识别并保留：
```python
["What's the weather", '<|endoftext|>', 'like today?']
```

#### 2. 分片段处理
对每个文本片段单独应用 BPE 合并算法。以 "What's the weather" 为例：
- 如果是特殊标记，直接跳过处理
- 对普通文本转换为适合 BPE 处理的字节单元列表：
```python
[[b'W', b'h', b'a', b't'], [b"'", b's'], [b' ', b't', b'h', b'e'], [b' ', b'w', b'e', b'a', b't', b'h', b'e', b'r']]
```

#### 3. BPE 合并操作
对每个字节列表应用合并规则：
1. 找出所有相邻字节对：`(b'W', b'h'), (b'h', b'a'), (b'a', b't')`
2. 使用[预训练](/2025/2025-08-29-build-bpe-from-scratch)阶段得到的 `merges` 列表查找优先级最高的字节对
3. 执行合并操作，更新字节列表：`[b'W', b'h', b'at']`
4. 重复此过程直到无法继续合并

#### 4. Token 到 ID 的映射
将合并后的 Token 序列转换为[预训练](/2025/2025-08-29-build-bpe-from-scratch)阶段得到的词汇表中的对应 ID：
```python
[100, 201, 305, 400, 502, 607, 720, 801]
```

代码实现如下：
```python
def encode(text: str) -> list[int]:
    # 预分词 - 保留 special_tokens，处理重叠情况
    pattern = '|'.join(map(re.escape, special_tokens))
    words = re.split(f"({pattern})", text) if special_tokens else [text]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    all_ids = []

    for word in words:
        # 如果是 special token，直接转换为对应的 ID
        if word in special_tokens_map:
            all_ids.append(special_tokens_map[word])
            continue

        # 转成字节列表：[b't', b'h', b'e']
        pre_tokens = [
            [bytes([i]) for i in match.group().encode('utf-8')]
            for match in re.finditer(PAT, word)
        ]

        # 对每个 token 应用合并规则
        for pre_token in pre_tokens:
            # 循环应用合并规则
            while len(pre_token) > 1:
                # 1. 找到所有相邻对
                pairs = [(pre_token[i], pre_token[i+1]) for i in range(len(pre_token) - 1)]

                # 2. 查找这些对在 merges_map 里的优先级，找到优先级最高的
                best_pair = None
                highest_priority = float('inf')  # 优先级越小越高

                for i, pair in enumerate(pairs):
                    priority = merges_map.get(pair)
                    if priority is not None and priority < highest_priority:
                        highest_priority = priority
                        best_pair = pair

                # 3. 如果找不到优先级，结束循环
                if best_pair is None:
                    break

                # 4. 如果找到了优先级，则执行合并
                i = 0
                new_token = []
                while i < len(pre_token):
                    if i + 1 < len(pre_token) and (pre_token[i], pre_token[i+1]) == best_pair:
                        # 执行合并
                        merged = pre_token[i] + pre_token[i+1]
                        new_token.append(merged)
                        i += 2
                    else:
                        new_token.append(pre_token[i])
                        i += 1
                pre_token = new_token

            for token in pre_token:
                if token in vocab_rev:
                    all_ids.append(vocab_rev[token])

    return all_ids
```

完整的代码[在这](https://github.com/rust17/assignment1-basics/blob/main/cs336_basics/BPE/demo3/tokenizer.py)

### 模型处理阶段

模型接收到 Token ID 序列后，通过 Transformer 等神经网络结构进行计算，逐次预测下一个最可能出现的 Token，直到生成结束标记为止。

### 解码（Decode）过程

模型计算完后，输出的是一个 Token ID 序列，例如：`[23456, 34567, 45678, ..., 20000]`，需要转回人类能看懂的话：

查阅词汇表，将每个 ID 转换成 Token 字符串，然后将字符串合成一个完整的句子，返回给用户。

代码实现比较简单：
```python
def decode(ids: list[int]) -> str:
    res = [vocab[id] for id in ids if id in vocab]
    all_bytes = b''.join(res)
    return all_bytes.decode('utf-8', errors='ignore')
```

### 总结

本文详细介绍了基于 BPE 算法的 Tokenizer 实现，涵盖了编码和解码两个核心过程。通过构建完整的 Tokenizer，我们为后续的语言模型训练和推理奠定了坚实的基础。
