# HS_resnet18

## 参考
#### [paper](https://arxiv.org/abs/2010.07621)
#### [HS-ResNet](https://github.com/bobo0810/HS-ResNet)
## 环境
- python
- pytorch
- ubuntu

## 训练
- SGD
- cross_entropy
- base_lr:0.1 (cosine learning rate)

## 模型核心
![HS_module](https://github.com/2anchao/HS_resnet18/tree/main/pictures/HS_module.png)       
![compare](https://github.com/2anchao/HS_resnet18/tree/main/pictures/compare.png)       

## 其它
- 基本上都是参考代码：https://github.com/bobo0810/HS-ResNet 
- 修改得到了HS_resnet18
- HS_resnet18的Imagenet测试集正确率为：73.5
- 模型位置，可作为检测、分割等视觉任务的预训练模型，模型位置：
- 训练结果并不理想，模型的速度和存储大小都比较大，正确率也低于我训练得到的resnest18（后续会开源，resnest18的Imagenet测试集正确率为74.5）
- 等待百度开源代码
- 感觉模型的创新点并不是很强，和res2net很像，都是分组级联（分组减少参数、级联增加感受野可能性，特征交互融合、细粒度特征提取）

