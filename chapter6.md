# 第6章 文本摘要

在某一时刻，你可能需要对一份文件进行摘要，无论是研究文章、财务收益报告，还是一连串的电子邮件。如果你仔细想想，这需要一系列的能力，比如理解长篇大论，对内容进行推理，并制作出流畅的文本，将原始文件的主要议题纳入其中。此外，准确地摘要一篇新闻文章与摘要一份法律合同有很大的不同，所以能够做到这一点需要有复杂的领域概括能力。由于这些原因，文本摘要对于神经语言模型，包括Transformers来说是一项困难的任务。尽管有这些挑战，文本摘要还是为领域专家提供了大幅加快工作流程的前景，并被企业用来浓缩内部知识、摘要合同、为社交媒体发布自动生成内容等等。

为了帮助你了解其中的挑战，本章将探讨我们如何利用预训练的Transformers来摘要文档。摘要是一个经典的序列到序列（seq2seq）任务，有一个输入文本和一个目标文本。正如我们在第1章中所看到的，这正是编码器-解码器Transformers的优势所在。

在这一章中，我们将建立自己的编码器-解码器模型，将几个人之间的对话浓缩成一个简洁的摘要。但在这之前，让我们先来看看摘要的典型数据集之一：CNN/DailyMail语料库。

## CNN/DailyMail 数据集

CNN/DailyMail数据集由大约300,000对新闻文章及其相应的摘要组成，这些摘要由CNN和DailyMail在其文章中附加的要点组成。该数据集的一个重要方面是，摘要是抽象的，而不是摘录的，这意味着它们由新的句子而不是简单的摘录组成。该数据集可在Hub上找到；我们将使用3.0.0版本，这是一个为摘要而设置的非匿名版本。我们可以用类似于分割的方式来选择版本，我们在第四章中看到，用版本关键词来选择。因此，让我们潜入其中，看一看：

```
from datasets import load_dataset 
dataset = load_dataset("cnn_dailymail", version="3.0.0")
print(f"Features: {dataset['train'].column_names}") 

Features: ['article', 'highlights', 'id']

```

该数据集有三列：文章，其中包含新闻文章，亮点与摘要，以及唯一标识每篇文章的ID。我们来看看一篇文章的摘录：

```
sample = dataset["train"][1] 
print(f""" Article (excerpt of 500 characters, total length: {len(sample["article"])}): """) 
print(sample["article"][:500]) print(f'\nSummary (length: {len(sample["highlights"])}):')
print(sample["highlights"])

Article (excerpt of 500 characters, total length: 3192): 

(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze after Britain were disqualified for a faulty handover. The 26-year-old Bolt has n 

Summary (length: 180):

Usain Bolt wins third gold of world championship . Anchors Jamaica to 4x100m relay victory . Eighth gold at the championships for Bolt . Jamaica double up in women's 4x100m relay .

```

我们看到，与目标摘要相比，文章可能非常长；在这个特定的案例中，差异是17倍。长文章对大多数Transformers模型构成了挑战，因为上下文的大小通常被限制在1000个左右，这相当于几个段落的文字。处理这个问题的标准但粗略的方法是简单地截断超出模型上下文规模的文本。显然，在文本的结尾处可能会有重要的摘要信息，但是现在我们需要忍受模型结构的这种限制。



## 文本摘要流水线

让我们先从质量上看一下前面的例子的输出，看看几个最流行的Transformers模型在摘要上的表现。尽管我们要探索的模型架构有不同的最大输入规模，但我们把输入文本限制为2000个字符，以便所有模型都有相同的输入，从而使输出更具有可比性：

```
sample_text = dataset["train"][1]["article"][:2000] 
# We'll collect the generated summaries of each model in a dictionary 
summaries = {}

```

摘要中的一个惯例是用一个换行来分隔摘要句子。我们可以在每个句号之后添加一个换行符，但是对于像 "U.S. "或 "U.N. "这样的字符串，这种简单的启发式方法会失败。自然语言工具包（NLTK）软件包包括一个更复杂的算法，可以从缩写中出现的标点符号中区分出句子的结束：

```
import nltk from nltk.tokenize import sent_tokenize
nltk.download("punkt") 
string = "The U.S. are a country. The U.N. is an organization." 
sent_tokenize(string)

['The U.S. are a country.', 'The U.N. is an organization.']

```

**警告**

在下面的章节中，我们将加载几个大型模型。如果你的内存用完了，你可以用较小的模型（如 "gpt"、"t5small"）来替换大型模型，或者跳过本节，跳到 "在CNN/DailyMail数据集上评估PEGASUS"。
