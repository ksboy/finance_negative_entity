# 金融负面实体识别

## 总体思路

采用pipeline的方式，采用五折交叉验证/划分数据集。

1. 预处理

2. 判断句子中是否存在负面实体，即二分类：正类表示句子中存在负面实体，负类则不存在。

   输入是句子，输出为句子类别。

3. 若第1步结果为正类（句子中存在负面实体），则对句中每个entity做二分类：正类表示entity是负面，负类是非负面。

   输入为 concat(sentence, entity)，输出为entity类别。

   key_entity为所有负面实体 list。若key_entity为空，则修正句子类别为负类。

4. 后处理

## 具体细节

### 预处理

对于英文和特殊字符（未出现在实体中），我们先分词后处理的方法，对于每个词：

1. 去掉英文

2. 去掉特殊符号，除了 ["，", "。", "：",",","、","？","@",'#']以外

由 "[]","{}","()","【】"，"（）"，"##"括起来的词语很多都是无意义的短语（未出现在实体中），则去掉

### 句子分类

采用Bert（chinese_wwm_ext_pytorch）提取特征，全连接层作为分类层。

Input_A : sentence
Input_B : None
Output: sentence label


### 实体分类

同样采用Bert（chinese_wwm_ext_pytorch）提取特征，全连接层作为分类层。

找出句子中实体所对应的向量（多个向量取加和平均，合成768维），与cls对应向量拼接。输入到分类器中向量维度768+768=1536维。

Input_A : sentence
Input_B : entity
Output: entity label

### 后处理

包含实体：实体B是实体A的子串，则删除实体B。

例如实体A="北京资易贷公司（小资钱包平台）"，实体B="资易贷"，则去掉实体B。

线上结果提升了4个百分点。

### 模型融合

我们采用五折划分数据集，训练5个模型；shuffle数据集后再次训练5个模型；共有10个模型，采用投票的方式进行融合。

分析实验结果，我们发现cls_sentence 的recall 较高，precision很低，因此我们在投票时适当增加阈值。

线上结果提升了1个千分点。

## 实验结果

### 交叉验证

cls_sentence = 0.9653

cls_entity =0.9514

### 线上

overall= 0.95676786

cls_sentence =  0.97496002

cls_entity = 0.94463975

### 其他

我们之前尝试通过ner来识别负面实体，但是f1不高，在0.9左右。

## 具体实现

代码采用AllenNLP框架，Github地址：https://github.com/ksboy/finance_negative_entity