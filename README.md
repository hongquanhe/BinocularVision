# 一、项目简介
##项目名称：弱光环境下的小目标缺陷定位与监测
###作品编号：2022012455
###2022年中国大学生计算机设计大赛参赛作品
###人工智能赛道（普通赛道）

###Risk Small Target Defect Location and Monitoring in Low-light Environment
@author: 何洪权
@Email: sigermenz_he@henu.edu.cn

1）采用双目RGBD相机对管道缺陷的类别进行预测与定位，
相比现有的方法提升了检测的速度与精度，
并提出了一种管道缺陷数据集的分类方法，
将深度学习应用到双目视觉定位实现目标检测，
参考YOLOX网络搭建深度学习卷积神经网络，
并做适当调整，简化了特征提取部分并精简网络结构，
保证信息传播的同时，对像素进行适当的定位提高预测率。

2）减小了检测的成本，
能够对各类管道进行多类别多数量的预测与定位，
具有监测范围广，实时性好，精确度高，
并能够准确定位等优点；
同时，克服了漏磁检测法等传统管道检测方法只能检测铁磁性材料管道的局限性，
且不会产生虚假信号，对漏报、误报的情况有一定程度的改善。

##如何演示我们的作品？

1）解释何为双目视觉、解释何为相对坐标

![img_left](gui_img/0left.png)

![img_right](gui_img/0right.png)



2）演示实时GUI界面

3）演示小目标检测与计数等功能

4）演示测距与定位功能

##Step1.目标检测
(参见21.07旷世发表的一篇论文DOI:10.16383/J.ASS.2021.c120909)
##Step2.目标跟踪(DeepSort)
##Step3.定位
###3.1使用双目相机，进行相机标定
###3.2立体矫正
###3.3立体匹配及深度计算
###3.4点云图信息监测
###3.5得到实际空间中的三维坐标
##Step4.Python GUI搭建，采用Pyside2
- 本文件为大赛参赛作品的说明文件

# 二、安装与使用

### 1、部署环境
Terminal or cmd下执行以下脚本命令
```python
pip install -r requirements.txt
```
特别说明：
- 1.推荐使用国内镜像部署环境：如清华镜像https://pypi.tuna.tsinghua.edu.cn/simple
- 2.默认的requirements.txt文件中
torch==1.6.0 ，与之对应的torchvision==0.7.0
- 3.如果运行报错，可能是由于国内镜像不包含对应的torch版本，解决方法如下：
  >a.将txt中的torch与torchvision版本升级，如：torch==1.8.0&&torchvision==0.9.0。注意二者版本需一一对应
  > 
  > torch与torchvision对应版本请参照https://github.com/pytorch/vision
  > 
  >b.找到路径：python安装目录/torch/nn/modules/activation.py，在438行左右，删除“,self.inplace”并保存文件

- 4.解决有关DeepSort的OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized. #1715
>https://github.com/dmlc/xgboost/issues/1715
>
>（you can run this comment conda install nomkl）
- 5.可安装CUDA&CUDNN加速推理

### 2、Terminal/cmd运行非GUI
>python name.py --source path

- name.py :指"一"中的py文件

- path :指视频路径或摄像头IP地址：如./inference/video/test.mp4('绝对路径’)或者test.mp4('相对路径‘)
- 若使用手机端的IP摄像头：运行
  >python name.py --source http://admin:admin@192.168.xx.xx:xxxx
  
- IP登录的账号、密码与地址请根据实际更换，示例中的账号与密码均为admin
### 3、Terminal/cmd运行GUI
- 使用test.onnx对缺陷进行监测，缺陷类别共六类
  ```python
  ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
  ```
  - 若在浏览器使用摄像头:from camera import Camera
- 运行如下代码
  ```python
  python3 main.py
  ```
- 浏览器访问:http://返回的具体IP地址/index即可（务必加上index，详见app.py文件）

# 三、训练自己的数据集
本项目基于YOLOv5算法实现，其数据集训练过程如下：
### 1.Kaggle中谢韦尔钢铁公司提供的带钢缺陷数据集+自建数据集（12568张图片和4段1分钟左右的视频）
该数据集中提供了四种类型的带钢表面缺陷。训练集共有12568张，测试集5506张。图像尺寸为1600×256:
#### (a).负样本：数据集中引入了351幅负样本图像，即无人缺陷。收集一些在纹理上相似的图像，如PCB版、硬件电路等相似特征的图像，期望帮助模型提高泛化性能。
#### (b).高分辨率图像。对于较小缺陷目标场景做优化处理，数据集搜集了高分辨率的图像，其中最大尺寸达到了640*1194。数据图像的平均分辨率为291×924。
#### (c).更加明显的表观差异。整个数据集的缺陷数量范围在0到18.
### 2.创建数据集的配置文件dataset.yaml
可以基于已存在的[coco.yaml]修改自己数据集的
yaml文件，如下所示：
```python
#train and val datasets (image directory or *.txt file with image paths)

train: ../datasets/images/train/

val: ../datasets/images/val/
#number of classes
nc: 1
#class names
names: ['person']
```
### 3.创建标注文件
可以使用LabelImg, Labme, Labelbox, CVAT来标注数据，对于目标检测而言，标注bounding box即可。
此时需要将标注转换为和darknet format相同的标注形式，每一个图像生成一个*.txt的标注文件（如果
该图像没有标注目标，则不用创建*.txt文件）。创建的*.txt文件遵循如下规则：
- 每一行存放一个标注类别，以上述已创建的yaml中names的序列记，如person记0
- 每一行的内容包括:
```
  <class> x_center y_center width height
```
- Bounding box的坐标信息是归一化之后的(0-1)
- class label转化为index时计数从0开始
```python
def convert(size, box):
    '''
    将标注的xml文件标注转换为darknet形的坐标
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
```
- 每一个标注`*.txt`文件存放在和图像相似的文件目录下，只需要将`/images/*.jpg`替换为`/lables/*.txt`即可（在加载数据时代码内部的处理如此，可以自行修改为VOC的数据格式进行加载）

例如：

```
datasets/score/images/train/002021041091.jpg  # image
datasets/score/labels/train/002021041091.txt  # label
```

### 4.组织训练集的目录

#### 将训练集train和验证集val的images和labels文件夹按类进行存放

至此数据准备阶段已经完成，过程中我们假设数据清洗和数据集的划分过程已经自行完成。

### 5.选择模型backbone进行模型配置文件的修改

在项目的`./models`文件夹下选择一个需要训练的模型，这里我们选择`yolov5x.yaml`亦即最大的一个模型进行训练
(https://github.com/ultralytics/yolov5/blob/master/models/yolov5x.yaml)
- 参考官方README中的`table`
(https://github.com/ultralytics/yolov5#pretrained-checkpoints),
了解不同模型的大小和推断速度。如果选定一个模型，需要修改模型对应的`yaml`文件

```yaml

# parameters
nc: 1  # number of classes   <------------------  UPDATE to match your dataset
depth_multiple: 1.33  # model depth multiple
width_multiple: 1.25  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 8-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]],  # 10
  ]

# yolov5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 11
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 12 (P5/32-large)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 17 (P4/16-medium)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 22 (P3/8-small)

   [[], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

### 6.训练

```bash
# Train yolov5x on person for 300 epochs

$ python3 train.py --img-size 640 --batch-size 16 --epochs 300 --data ./data/score.yaml --cfg ./models/score/yolov5x.yaml --weights weights/yolov5x.pt

```


### 7.可视化

- 开始训练后，查看`train*.jpg`图片查看训练数据，标签和数据增强，如果你的图像显示标签或数据增强不正确，你应该查看你的数据集的构建过程是否有问题

- 一个训练epoch完成后，查看`test_batch0_gt.jpg`查看batch 0 ground truth的labels

- 查看`test_batch0_pred.jpg`
  
- 查看test batch 0的预测

- 训练的losses和评价指标被保存在Tensorboard和`results.txt`log文件。`results.txt`在训练结束后会被可视化为`results.png`

```python
>> rom utils.utils import plot_results
>> plot_results()
```
#### 社交距离
![img_4](runs/exp/res.png)

### 8.推断

```python
>> python detect.py --source file.jpg  # image 
                            file.mp4  # video
                            ./dir  # directory
                            0  # webcam
````


```python
# inference  /home/myuser/xujing/EfficientDet-Pytorch/dataset/test/ 文件夹下的图像
>> python detect.py --source path --weights weights/file.pt --conf 0.1
>> python detect.py --source path --weights weights/file.pt --conf 0.5

# inference  视频
>> python detect.py --source test.mp4 --weights weights/file.pt --conf 0.4
```

### 9.YOLOv5的模型评价
#### 该模型大小为42931KB，大小介于官方的yolov5s.pt 与yolov5m.pt之间，但推理速度与精度均高于YOLOv5s模型。
[Here](https://github.com/ultralytics/yolov3)
- 该链接为YOLOv3的TensorRT加速，详见其README.md文件。同样适用于YOLOv5

- 各评价指标
![img](runs/exp/results.png)
- 验证集结果

![img_1](runs/exp/labels.jpg)
- 模型优化

![img_2.png](runs/exp/better.png)

### 11.GUI架构的前后端搭建
####整体思路：模块化实现功能如下：双目摄像头调用、参数设置、目标检测




**参考**

[1].https://github.com/ultralytics/yolov5

[2].https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data




