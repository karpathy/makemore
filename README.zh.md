# makemore

<p align="center">
  <a href="https://github.com/karpathy/makemore/blob/master/README.md">English</a> · <b>简体中文</b>
</p>

makemore 接收一个文本文件作为输入（假设每一行为一个训练样本），并生成更多与其类似的内容。在底层原理上，它是一个自回归字符级语言模型（autoregressive character-level language model），提供了从二元语法模型（Bigram）一直到 Transformer（架构完全与 GPT 一致）的丰富模型选型。例如，向其喂入一个名字数据库，makemore 便能生成富有创意、听起来像真实名字但实际上并不存在的婴儿姓名灵感；若输入公司名称库，它能为新公司命名提供创意点子；或者也可以直接输入拼字游戏（Scrabble）有效单词表，生成类似英文的合成词。

本项目无意成为拥有无数复杂配置项的臃肿库，而是一个结构清晰、易于魔改的单个独立文件，主要面向教学与算法理解。[PyTorch](https://pytorch.org) 是其唯一的依赖项。

当前代码实现参考了以下几篇里程碑式核心论文：

- 二元语法（Bigram）：通过统计计数查找表，根据当前单个字符预测下一个字符
- 多层感知机（MLP）：参考 [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 卷积神经网络（CNN）：参考 [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499)（开发中...）
- 循环神经网络（RNN）：参考 [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- 长短期记忆网络（LSTM）：参考 [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- 门控循环单元（GRU）：参考 [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer：参考 [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### 使用说明

项目中自带的 `names.txt` 数据集作为示例，收录了来自 [ssa.gov](https://www.ssa.gov/oact/babynames/) 的 2018 年度最常见的 3.2 万个姓名。数据格式如下：

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

直接运行训练脚本：

```bash
$ python makemore.py -i names.txt -o names
```

训练进度、日志与保存的模型权重均会存放在工作目录 `names` 中。默认模型是一个约 20 万参数的极简轻量级 Transformer；代码库中还提供了多种训练配置项 - 详见命令行参数（argparse）与源码实现。训练不需要任何特殊的高性能硬件，在普通 MacBook Air 上即可轻松运行，各类通用平台均能支持；若有可用 GPU，训练速度将会成倍飞升。在训练过程中，脚本会持续实时打印部分生成采样。若希望手动从已有模型中采样生成，可添加 `--sample-only` 参数，例如在另一个独立终端中执行：

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

该命令将加载当前保存的最优模型，并按需打印更多采样样本。以下是基于当前默认配置（测试集对数概率约为 1.92，若进一步微调超参数可获得更低损失）最终生成的部分独特婴儿姓名：

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

祝探索愉快！

### 开源许可证

MIT License

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区开发者维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](https://github.com/karpathy/makemore/blob/master/README.md) 为准。欢迎通过 PR 协助改进译文！
