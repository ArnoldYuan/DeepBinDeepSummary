## word2vec 
### 算法逻辑
输入一篇文章（article），输出其中每个词的词向量（embedding）。
为什么要词向量？因为在计算机里所有的逻辑都是通过数字实现的，字符串只是用一定的编码存成数字，而这个编码本身并不包含词汇的语义信息。比如对于两个近义词，他们的编码可能相差的很远。因此在自然语言处理（NLP）中不能直接用传统编码（ASCII，utf-8）这种来在计算机中表示词语。相应的，我们应该设计一种新的“编码”方案，使得其能够包含语义信息，这样近义词之间的编码也会很接近，这种编码在NLP中通常叫词向量。word2vec，BERT等都是可以输入文章生成每个词词向量的模型。

word2vec如何训练？即怎么样让这些词向量包含语义信息？发明word2vec的作者想了两种办法：CBOW（Continuous bag of words/连续词袋模型）和SG（skip-gram/跳字模型）。前者是让模型用一个词周围的词预测这个词本身（即给模型周围词的词向量，让模型判断中心词是什么，即输出中心词的预测概率）。后者反之，给模型中心词的词向量，预测周围词的概率。需要注意的是，模型只会输出预测的概率，而并不是所谓的词向量。

然后就是训练过程。一开始所有词的词向量都是随机初始化的，预测肯定不准确。但没关系，我们的算法知道正确答案是什么，这样就能用梯度下降的方法，不断优化这些词向量，让预测的结果越来越接近正确答案。举例来说，在CBOW模型中，我们输入一句话“1 2 3 4 5”，每个数字代表一个词。第一步，找到所有词的词向量（存在一个矩阵里）。第二步，用2和4预测3这个位置应该是什么词，模型会输出3这个位置所有词的可能性。例如模型输出3这个位置的词是2的概率是0.1，是3的概率是0.5。模型会计算一个损失函数（我们要让模型预测出3的概率尽可能地接近1才行），然后用损失函数来更新词向量，更新的幅度定义为学习率。具体怎么更新需要矩阵求导，但一般API会帮我们计算好，这里就不详细说了（不可能手算的）。

当训练完成之后，我们就得到了词向量。这里也是值得注意的，也就是我们最终要的是模型的中间产物（词向量），而不是模型的预测结果，这个特点在无监督学习中很常见（unsupervised learning）。

在DeepBinDiff的输入文章中，每一个词是一个索引（index），可以通过索引在`dictionary`中找到对应的字符串（例如索引1对应的字符串为“mov”）。这里索引相当于NLP中的one-hot vector，只是为了用数学形式代替字符串形式的表示，并没有什么实际的意义。

接着，程序会从一个超级大的矩阵（存着所有词的词向量）中根据索引去拿到相应的词向量，比如索引1对应的词向量是[0.1, 0.2, 0.3]。这个矩阵是用均匀分布初始化的，向量中的数值分布在-1到1之间。矩阵的形状是`dic_size`行，`embedding_size`列。`embedding_size`定义在`featureGen.py`最上面的configuration部分，是词向量维度，在这里是64。
```python
embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1.0, 1.0))
```

然后，在训练部分遍历每个词，因为是CBOW模型，我们首先要拿到周围词的词向量。到底拿多少个周围的词？论文里定义只拿前一条指令和后一条指令。拿完以后加起来取平均，存到temp_embeddings里。
```python
prevInsn_embedding = tf.cond(has_prev, 
    lambda: cal_insn_embedding(embeddings, prevInsn, prevInsn_size), 
    lambda: tf.random_uniform([2 * embedding_size], -1.0, 1.0))
nextInsn_embedding = tf.cond(has_next, 
    lambda: cal_insn_embedding(embeddings, nextInsn, nextInsn_size), 
    lambda: tf.random_uniform([2 * embedding_size], -1.0, 1.0))

currInsn = tf.div(tf.add(prevInsn_embedding, nextInsn_embedding), 2.0)
temp_embeddings = tf.concat([temp_embeddings, currInsn], 0)
```
之后用得到的`insn_embeddings`（这是模型根据上述步骤得到的结果)做预测（这里用的是negative sampling代替softmax，可以提高效率），然后计算损失函数。因为是负采样，所以这里不是预测概率了，预测的就是一个0/1的结果，即这个位置上的词是/否是给定的词。论文里给的损失函数公式是错误的，那个是没有用上负采样的公式，甚至都不是CBOW模型（是SG），在此表示质疑。

```python
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, # 预测模型的权重矩阵
                            biases=nce_biases, # 预测模型的偏移
                            labels=train_labels,
                            inputs=insn_embeddings,
                            num_sampled=num_sampled,
                            num_classes=dic_size))
```
使用1.0的学习率（怎么这么大hhh）最小化损失函数。因为每次计算损失函数要遍历所有的词求和然后更新网络参数，效率太低，文章里使用了随机梯度下降（SGD），每次选择一个batch计算。batch大小也定义在configuration中。
```python
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
```
上述过程循环`num_step`次。函数`buildAndTraining`的输出就是所有词的embedding（是一个tensorflow矩阵，每一行是一个词的embedding，词在矩阵中的顺序和dictionary中的顺序相同）。

## TADW
这个算法很神奇，我没搞懂他的原理。简单来说就是给算法一个图，以及图上每个节点的节点信息（这里就是embedding了），可以输出每个节点整合原来节点信息和图结构信息的embedding，也就是在原来只有语义信息的embedding的基础上加上了跟图相关的结构信息。算法说可以通过一个简单的矩阵分解就能实现（我数学不好看不懂这玩意的证明）。不过如果只要调用代码的话就很方便了！
```python
tadw_command = "python3 ./src/performTADW.py --method tadw --input " + bin_edgelist_file + " --graph-format edgelist --feature-file " + bin_features_file + " --output vec_all"
```
给两个file，一个是图的（这个时候应该两个图已经merge了）edgelist文件，还有一个前面的步骤得到的节点的feature vector文件，就能输出最终embedding到`vec_all`文件。