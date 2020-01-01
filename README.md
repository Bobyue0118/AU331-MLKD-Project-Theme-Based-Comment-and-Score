# AU331-MLKD-Project-Theme-Based-Comment-and-Score
**主要功能：**输入电影主题（电影名+星级），自动输出相应影评；只继续输入影评，输出星级。
## Description
本项目为上海交通大学机器学习与知识发现（AU331）课程项目，主要模型框架见下图。
- 整体模型框架示意图
![image](https://github.com/Bobyue0118/AU331-MLKD-Project-Theme-Based-Comment-and-Score/blob/master/assets/%E6%95%B4%E4%BD%93%E6%A1%86%E6%9E%B6.png)
- 只基于主题限制的影评生成模型
![image](https://github.com/Bobyue0118/AU331-MLKD-Project-Theme-Based-Comment-and-Score/blob/master/assets/%E5%BD%B1%E8%AF%84%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B1.png)
- 基于主体限制 & 注意力机制的影评生成模型
![image](https://github.com/Bobyue0118/AU331-MLKD-Project-Theme-Based-Comment-and-Score/blob/master/assets/%E5%BD%B1%E8%AF%84%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B2.png)
## Idea
- 以Gated Recurrent Unit(GRU)为核心
- 结合注意力机制(attention mechanism)、集束搜索(beam search)、adaptive_softmax等策略
## To be Polished
1. 努力学习，广泛涉猎，改进网络
2. 适用其他文学体裁，比如：书评，音乐评论等
3. 适用其他语言，表情符号：英语，日语，emoji等
4. 扩大word2vec字典
5. 帮助评论网站识别恶意水军评论
## Project Participant
岳博，潘鼎，徐加声 上海交通大学自动化系
## Data Sources
豆瓣，主要爬取20部中国国产电影
