---
title: "从零开始实现一个 BPE 分词器"
layout: post
date: 2025-08-29 11:00
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
description: cs336 spring2024 assignment1 basics
---

### 1. 引言：为什么我们需要分词（Tokenization）？

作为一名程序员，肯定每天都要跟 JSON 打交道，JSON 就是把一个复杂的数据结构转成一串字符串，Tokenization 其实有点像这个过程的反向操作——把非结构化的文本，转换成模型能理解的结构化数字序列。

首先，怎样让模型认识一段文本呢？模型不是人肯定不认识文字，只认识数字。怎样把文本转成数字呢？用 UTF-8 把文本转成字节，每个字节不就是 0-255 的数字吗？对，这完全正确，这样的好处是词汇表是固定的，大小只有 256，模型永远不会遇到“不认识的字符”，因为任何文本都可以拆成字节。但是这样有个问题，对于一个很常见的单词 "hello" 来说，如果用字节分词法，就会得到 5 个数字，那一篇文章呢？这个数字就会变得很长。对于计算机来说，处理一个超长的数字序列，计算量可能还会翻倍！字节分词的问题就是：效率太低了。

既然如此，如果用一个单词对应一个数字的分词法（单词分词），不就好了吗？用空格或者标点把句子切开，一个单词对应一个数字，例如：`["Hello", ",", "world", "!"] -> [15, 3, 22, 4]`，这样就不会造成数字序列很大了。问题是如果遇到了不认识的字咋办呢，单词分法的前提是我们已经知道了每个词对应的数字，可以将已知的单词转成数字，如果我给你一个新词 "supercalifragilisticexpialidocious"，你要怎么处理？可以动态扩充词典。听起来可行，问题是工程上会带来很多麻烦，所以通常把不认识的词统统标记为一个特殊的“未知词”，叫 UNK（Unknown）。好的，如果一句话里 UNK 太多，你觉得模型会有什么问题？就好像人一样，不认识的字太多，只能靠瞎猜，自然就懵了。

好，我们来想一个折中的办法，既能像单词分词那样把常见的词标注为一个单独的数字（token），又能像字节分词那样，遇到不认识的词，可以把它拆成更小、能认识的单元。这听着有点像我们处理自然语言的方式，遇到一个复杂的词，我们会尝试把它拆解成词根、前缀、后缀（英文语境）或者偏旁部首（中文语境）来理解。这正是 BPE（Byte Pair Encoding）的解决方式。

### 2. BPE 算法的核心思想

BPE 的核心思想其实非常简单：让数据自己告诉我们，哪些字符最常出现，最值得被合并成 token。

我们来模拟一下 BPE 算法的训练过程，听起来很高大上？别担心，其实逻辑比听起来要简单。

假设我们要把一句话转成数字表示："the cat in the hat"，

第一步：我们用之前说过的字节分词来作为起点，把这句话拆成字节序列（这里为了直观，将字节前面的 b 简化掉了）：
```
['t', 'h', 'e', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'h', 'a', 't']
```
我们的初始词汇表就是所有出现过的字节：
```
{'t', 'h', 'e', ' ', 'c', 'a', 'i', 'n'}
```
第二步：寻找出现次数最多的相邻字节对（pair），例如：('t', 'h') 和 ('h', 'e')。发现：('t', 'h')、('h', 'e')、('a', 't') 都出现了 2 次，这种情况下，我们使用字典序来选，('t', 'h') 字典序最大，我们选中了它。

第三步：也就是最关键的一步，合并！我们把序列中所有出现的 ('t', 'h') 都替换成一个新的单元：'th'，得到：
```
['th', 'e', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 'th', 'e', ' ', 'h', 'a', 't']
```
同时，我们把词汇表也进行更新：
```
{' ', 'a', 'c', 'e', 'h', 'i', 'n', 't', 'th'}
```
好，我们接着来进行第二轮，这次 ('th', 'e')、('a', 't') 都出现了 2 次，同样，根据字典序，我们选中 ('th', 'e')，进行合并后得到：
```
['the', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 'the', ' ', 'h', 'a', 't']
```
同时，我们把词汇表也进行更新：
```
{' ', 'a', 'c', 'e', 'h', 'i', 'n', 't', 'th', 'the'}
```
最终经过 5 轮合并后，所有对都只出现 1 次，无法继续进行合并，原始文本 "the cat in the hat" 被编码成序列：
```
['the ', 'c', 'at', ' ', 'i', 'n', ' ', 'the ', 'h', 'at']
```
最终词汇表：
```
{' ', 'a', 'at', 'c', 'e', 'h', 'i', 'n', 't', 'th', 'the', 'the '}
```
让我们来总结一下规律：将给定文本得到一份字节序列，根据文本动态构建词汇表，这样就不用担心有不认识的词，同时，不断的合并意味着压缩，让高频的词可以用一个单元表示。这个过程有点像消消乐，把最常见的相邻对合并，词汇表也越来越丰富，从单个字节慢慢扩展到常用的词根，甚至完整的单词。这样就能得到一开始说的“融合了单词分词和字节分词优点”的方法。

### 3. 初版实现：一个能跑但不够快的版本

好的，说了那么多，相信你已经对 BPE 有了一个基本的理解了，我们来实现一下这个算法的核心逻辑吧。

BPE 的核心就是不断重复这个过程：

1. 找出当前序列里最常见的一对邻居。
2. 合并它们，得到一个新单元。
3. 更新序列和词汇表。
4. 重复 1-3 步循环，直到达到我们预设的词汇表大小（比如 5 万个）。

我们拆成三部分实现

**1. 统计相邻对频率**
```python
def get_stats(tokens: list[list[bytes]]) -> dict[tuple, int]:
    """
    统计所有相邻字节对的出现频率

    1. 输入 tokens 是一个列表，每个元素是字节序列，代表当前文本的 token 序列
    2. 输出是一个字典，key 是相邻的 token 对（一个元组），value 是这对 token 出现的次数
    """
    pair_stats = defaultdict(int)

    for token in tokens:
        for i in range(len(token) - 1):
            pair_stats[(token[i], token[i+1])] += 1

    return pair_stats
```

**2. 合并函数**
```python
def merge(tokens: list[list[bytes]], pair: tuple[bytes, bytes]) -> list[list[bytes]]:
    """
    在所有字节序列中合并最频繁的字节对

    1. 输入 tokens 是一个列表，每个元素是字节序列，代表当前文本的 token 序列
    2. 输入 pair 是我们要合并的 token 对，比如 ('t', 'h')
    3. 输出是合并后新的字节序列
    """
    new_tokens = []
    for token in tokens:
        new_token = []
        i = 0

        while i < len(token):
            if (i < len(token) - 1 and
                token[i] == pair[0] and
                token[i+1] == pair[1]):
                new_token.append(pair[0] + pair[1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_tokens.append(new_token)

    return new_tokens
```

**3. 主循环**
```python
vocab_size = 276 # 假设我们想学 20 个新 token (256 个字节 + 20 个新合并)
num_merges = vocab_size - 256

for i in range(num_merges):
    stats = get_stats(pre_tokens) # 统计相邻对的频率
    top_pair = max(k for k in stats.keys() if stats[k] == max(stats.values())) # 从统计频率里找到频率最高的相邻对
    vocabs[len(vocabs)] = top_pair[0] + top_pair[1] # 更新词汇表
    pre_tokens = merge(pre_tokens, top_pair) # 合并
```

好了，我们实现了第一个版本，为了演示，这里放出的只是部分代码，完整代码可以在[这里](https://github.com/rust17/assignment1-basics/blob/main/cs336_basics/demo1/bpe_processing.py)查看。

这段代码有个问题，每一次合并，都会遍历当前文本的所有字节列表，成本是 O(V * L * T)，其中 V 是合并次数，L 是每个单元的长度，T 是所有单元的数量。如果文本很大，这里的成本就高得惊人！

### 4. 性能优化

我们来思考一下怎么提高上面这段代码的效率，关键要是降低每次合并的复杂度。

可以观察到，其实，每次合并真正受影响的单元只是其中的一小部分，还是以 "the cat in the hat" 为例，在上面的代码中，假如我们要合并 ('t', 'h')，实际上受影响的只有 "the"，而代码会遍历整个文本序列 `['the', 'cat', 'in', 'the', 'hat']`；要合并 ('a', 't') 也是，实际上受影响的只有 "cat" 和 "hat"，同样代码会遍历整个文本，这样执行，时间不高才怪。

那有没有什么办法可以降低每次遍历的范围呢？假如我已经知道了每个相邻对会影响的单元，那每次合并不就可以只遍历这些单元而不用遍历整个文本了吗？是的，聪明的你可能已经想到了，利用字典。

我们有一个字典如下，左边是每次合并的相邻对，右边是受影响的单元：
```python
{('t', 'h'): ['the'], ('a', 't'): ['cat', 'hat']}
```
第一次合并时，我们只需要遍历 `['the']`，而第二次只需要遍历 `['cat', 'hat']` 即可，这样，每次合并的复杂度从整个文本降低到有限个数的单元，这个思路其实就是利用空间换取时间。

我们来看看重构过后的代码：

**1. 统计相邻对频率**
```python
def get_stats(token_freqs: dict[tuple, int]) -> tuple[dict[tuple, int], dict[tuple, set]]:
    """
    统计所有相邻字节对的出现频率

    参数：
    - token_freqs: 字节序列到频率的映射，格式如 {(b'h', b'e', b'l', b'l', b'o'): 5}

    返回值：
    - pair_freqs: 字节对到总频率的映射，如 {(b'h', b'e'): 10, (b'e', b'l'): 15}
    - pair_tokens: 字节对到包含该字节对的所有字节序列集合的映射
    """
    pair_freqs = defaultdict(int)
    pair_tokens = defaultdict(set)

    for token, freq in token_freqs.items():
        token_set = tuple(token)
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_freqs[pair] += freq
            pair_tokens[pair].add(token_set)

    return pair_freqs, pair_tokens
```

**2. 合并函数**
```python
def merge(token: list, top_pair: tuple, pair_freqs: dict[tuple, int], pair_tokens: dict[tuple, set], pre_token_freqs: dict[tuple, int]):
    """
    合并指定 token 中的最频繁字节对

    参数：
    - token: 要合并的字节序列
    - top_pair: 要合并的字节对 (byte1, byte2)
    - pair_freqs: 字节对频率统计字典
    - pair_tokens: 字节对到包含该字节对的所有字节序列集合的映射
    - pre_token_freqs: 字节序列频率统计字典
    """
    freq = pre_token_freqs.pop(token)

    new_token_merged = []
    j = 0
    while j < len(token):
        if (j < len(token) - 1 and
            token[j] == top_pair[0] and
            token[j+1] == top_pair[1]):
            new_token_merged.append(top_pair[0] + top_pair[1])
            j += 2
        else:
            new_token_merged.append(token[j])
            j += 1

    new_token_merged = tuple(new_token_merged)
    pre_token_freqs[new_token_merged] = freq

    for j in range(len(token) - 1):
        pair = (token[j], token[j+1])
        pair_freqs[pair] -= freq
        pair_tokens[pair].discard(token)

    for j in range(len(new_token_merged) - 1):
        new_pair = (new_token_merged[j], new_token_merged[j+1])
        pair_freqs[new_pair] += freq
        pair_tokens[new_pair].add(new_token_merged)
```

**3. 主循环**
```python
pre_token_freqs = regex_pretokenize(text, special_tokens)

merges = []
pair_freqs, pair_tokens = get_stats(pre_token_freqs)

num_merges = vocab_size - len(vocabs)
for _ in range(num_merges):
    top_pair = None
    max_freq = 0
    for pair, freq in pair_freqs.items():
        if freq > max_freq or (freq == max_freq and (top_pair is None or pair > top_pair)):
            max_freq = freq
            top_pair = pair

    merges.append((top_pair[0], top_pair[1]))

    vocabs[len(vocabs)] = top_pair[0] + top_pair[1]

    affected_tokens = list(pair_tokens[top_pair])
    pair_tokens[top_pair].clear()
    for token in affected_tokens:
        merge(token, top_pair, pair_freqs, pair_tokens, pre_token_freqs)
```

相比于第一版，新的代码更精确了，每一步操作都只针对受影响的部分，而不是对整体做扫描，完整代码地址在[这里](https://github.com/rust17/assignment1-basics/blob/main/cs336_basics/demo2/bpe_processing.py)。

让我们来看看这一版算法的复杂度，每轮合并当中，需要

1. 找到最频繁字节对，假设 P 是不同字节对的数量，复杂度是 O(P)
2. 更新受影响字节序列，假设 L 是字节平均长度，A 是平均每轮受影响的字节序列数量，复杂度就是 O(A * L)
3. 总体而言，时间复杂度是 O(V * (P + A * L))，通常随着迭代进行，受影响 token 数量 A 会逐渐减少

### 5. 总结
我们从一个看似简单的问题——“如何让计算机读懂文字？”出发，经历了一趟从理论到实践，再到性能优化的工程之旅。我们没有止步于理论，而是亲手构建了两种不同效率的 BPE 分词器。第一版实现让我们理解了算法的本质，而第二版的优化过程，则让我们深刻体会到数据结构和算法对于解决实际性能瓶颈是多么关键。

BPE 虽然强大，但它只是“子词”分词世界的一员。还有许多有趣的变种，比如 Google T5 模型使用的 WordPiece（它根据提升数据“似然度”来选择合并对），以及更复杂的 Unigram 模型。幸运的是，我们不必每次都重新发明轮子。像 Hugging Face Tokenizers 这样的开源库已经为我们提供了工业级的、高度优化的分词器实现。在理解了 BPE 的底层原理之后，再去使用这些工具，我们就能更加游刃有余，知道在什么场景下该做什么样的选择。

### 参考资料：
- [Stanford CS336 Assignments](https://github.com/DK-Zhu/stanford-cs336-assignments)
- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/index.html)
