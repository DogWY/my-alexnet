# Alexnet 论文复现

[论文地址](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

AlexNet 首次证明了学习到的特征可以超过手工设计的特征。

AlexNet 使用ReLU而不是Sigmoid作为激活函数，ReLU的求导更加简单，并且比Sigmoid更能适应不同的初始化方法。

AlexNet 使用暂退法（Dropout）控制全连接层，增强模型的泛化性能。

AlexNet 强调模型的深度。