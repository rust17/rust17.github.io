---
title: "使用多进程优化 BPE 分词器"
layout: post
date: 2025-09-06 11:00
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
description: using parrallel to speed up bpe
---

### 背景

在[之前的文章](/2025/2025-08-29-build-bpe-from-scratch)中，我们实现了 BPE 算法并借助索引优化显著降低了算法的时间复杂度，使其能够应对大规模语料训练。然而，在处理超大文本文件时，直接全量加载进内存仍容易引发内存不足（OOM）的问题：：

```python
with open(input_path, "r", encoding="utf-8") as f: text = f.read()
```

为此，我们尝试引入分块读取与多进程处理，进一步提升程序的吞吐能力和扩展性。

### 分块读取与边界处理

首先考虑将文件切分为多个块（chunk）逐步读取，每次仅处理一个块，从而避免一次性内存占用过高：
```python
chunk_size = 1024 * 1024 * 10 # 10MB
pre_token_freqs = defaultdict(int)

with open(input_path, "r", encoding="utf-8") as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        chunk_freqs = regex_pretokenize(chunk, special_tokens)
        for token, freqs in chunk_freqs.items():
            pre_token_freqs[token] += freqs
```
但这种方法存在一个问题：如果某个词恰好处在两个 chunk 的边界，就可能被错误切分。因此，我们需要在合理的位置进行切分，例如特殊标记或换行符处，以保证词汇完整性。

改进后的策略如下：

1. 每次读取固定大小的块
2. 从这个块的末尾开始往前找，找到第一个合理的断点，例如：换行符、空格或者 special_token
3. 以断点为界，将当前块分为有效部分与剩余部分，处理有效部分
4. 将剩余部分与下一块拼接，继续处理

代码实现如下：
```python
chunk_size = 1024 * 1024 * 10 # 10MB
pre_token_freqs = defaultdict(int)

with open(input_path, "r", encoding="utf-8") as f:
    leftover = ''
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        # 将上一次剩余内容与当前块合并
        chunk = leftover + chunk
        leftover = ''

        # 查找任意特殊标记的位置
        end = chunk.rfind("<|endoftext|>")

        if end != -1:
            # 找到断点，按照断点进行切分
            text = chunk[:end]
            leftover = chunk[end:]
        else:
            # 没有找到，就找最近的换行符
            end = chunk.rfind('\n')
            if end != -1:
                text = chunk[:end]
                leftover = chunk[end:]
            else:
                text = chunk
                leftover = ''

        chunk_freqs = regex_pretokenize(text+leftover, special_tokens)
        for token, freqs in chunk_freqs.items():
            pre_token_freqs[token] += freqs
```
### 引入多进程并行处理

尽管分块读取避免了内存问题，但串行处理大文件仍然耗时较长。为进一步提升效率，我们引入多进程并行处理机制：

需要注意的是，我们在拆分的过程不再需要一次性把文件读如内存了，我们可以利用文件指针，直接跳到理论分割点，让我们来模拟一下整个工作流程：

1. 确定大致分割点：例如，希望将 100 G 的文件分成 4 块，理论分割点分别是 0 GB，25 GB，50 GB，75 GB，100 GB
2. 寻找第一个分割点
    - 利用文件指针一下子定位到 25 GB
    - 从这个位置开始往后找合理断点，假设在 25.01 GB 找到了
    - 第一个块的边界就是 (0GB, 25.01GB)
3. 以此类推，寻找第二个分割点，直到文件末尾
4. 拿到所有分割点后，我们就可以把这些边界信息分发给多个子进程了，这样每个子进程只读取一小部分数据

来看看经过多进程改造后的代码：
```python
def find_split_points(input_path: str, nums_of_split: int, special_token: str) -> list[int]:
    file_size = os.path.getsize(input_path)

    splits = [i * file_size // nums_of_split for i in range(nums_of_split + 1)]

    actual_splits = [0]

    special_token_bytes = special_token.encode('utf-8')

    with open(input_path, 'rb') as f:
        for i in range(1, nums_of_split):
            # 定位到理论分割点
            f.seek(splits[i])

            # 读取足够数据来查找断点
            data = f.read(1024)

            pos = data.find(special_token_bytes)

            if pos != -1:
                # 找到断点，保存起来
                actual_splits.append(splits[i] + pos)
            else:
                # 没有找到，就找最近的换行符
                pos = data.find(b'\n')
                if pos != -1:
                    actual_splits.append(splits[i] + pos + 1)
                else:
                    actual_splits.append(splits[i])

        actual_splits.append(file_size)
    return actual_splits

split_points = find_split_points(input_path, multiprocessing.cpu_count(), "<|endoftext|>")

args = [(input_path, split_points[i], split_points[i+1], special_tokens) for i in range(len(split_points) - 1)]

pre_token_freqs = defaultdict(int)
with multiprocessing.Pool() as pool:
    pre_iterator = pool.imap_unordered(regex_pretokenize, args)

    for i, freqs in enumerate(pre_iterator):
        for token, freqs in chunk_freqs.items():
            pre_token_freqs[token] += freqs
```
完整代码地址在[这里](https://github.com/rust17/assignment1-basics/blob/main/cs336_basics/demo3/bpe_processing.py)。

### 总结
通过分块读取与多进程并行处理，我们成功解决了 BPE 训练中的内存瓶颈问题，并显著提升了处理大规模语料的效率。该方法不仅适用于 BPE 算法，也可推广到其他需要处理大文件的任务中。
