#### HS_resnet18

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
<div align=center><img src="https://github.com/2anchao/HS_resnet18/blob/main/pictures/HS_module.png" width="500" height="380" /></div>.    
<div align=center><img src="https://github.com/2anchao/HS_resnet18/blob/main/pictures/compare.png" width="500" height="380" /></div>.  
 
## 其它
- 基本上都是参考代码：https://github.com/bobo0810/HS-ResNet. 修改得到了HS_resnet18。
- HS_resnet18的Imagenet测试集正确率为：73.5。
- 模型位置(链接: https://pan.baidu.com/s/1WKrPF_aGc-SN-JgkJjc1-w  密码: rivl)，可用作检测、分割等视觉任务的预训练模型。
- 训练结果并不理想，模型的速度和存储大小都比较大，正确率也低于我训练得到的resnest18（后续会开源，resnest18的Imagenet测试集正确率为74.5）。
- paper的代码还没更新，等待百度开源代码，吃瓜群众一枚。
- 感觉模型的创新点并不是很强，和res2net很像，都是分组级联（分组减少参数、级联增加感受野可能性、特征交互融合、细粒度特征提取）。
- 请给贫困山区孩子献上你的爱心(可选项)：   

<div align=center><img src="https://github.com/2anchao/HS_resnet18/blob/main/pictures/pay.jpeg" width="250" height="400" /></div>.
