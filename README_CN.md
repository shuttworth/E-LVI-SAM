# E-LVI-SAM

## 更简洁好用的LVI-SAM

E-LVI-SAM是[LVI-SAM](https://github.com/TixiaoShan/LVI-SAM)的更简单、更好用的实现。主要修改如下：

* 优化了代码结构，删除了不必要的代码，遵循[LVI-SAM-Easyused](https://github.com/Cc19245/LVI-SAM-Easyused)
* 代码中添加了详细的中文注释，遵循[知乎](https://www.zhihu.com/people/gao-li-dong-62/posts)和[LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)
* 使用之前在[LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)中创建的docker镜像，可以节省本地环境配置时间

## 前置条件
### 使用Docker

我们非常建议使用docker来帮助环境配置


```docker pull liangjinli/slam-docker:v1.2```

更多细节可以在[LVI-SAM Docker使用图文简介](https://github.com/shuttworth/E-LVI-SAM/blob/master/LVI-SAM%20Docker%E4%BD%BF%E7%94%A8%E5%9B%BE%E6%96%87%E7%AE%80%E4%BB%8B.pdf)文档中看到

### 不使用Docker
详见[dependency](https://github.com/TixiaoShan/LVI-SAM#dependency)


## 建图结果
<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/result.png" width="70%">
</center>

<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/gate_01_v1.gif" width="70%">
</center>

<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/street_08_v1.gif" width="70%">
</center>


## 在不同的数据集上运行
### 在官方采集数据集上运行
```roslaunch lvi_sam run_official.launch```
```rosbag play handheld.bag```

### 在M2DGR数据集上运行
```roslaunch lvi_sam M2DGR.launch```
```rosbag play gate_01.bag```

### 在UrbanNavDataset数据集上运行 
```roslaunch lvi_sam UrbanNavDataset.launch```
```rosbag play 2020-03-14-16-45-35.bag```

### 在KITTI raw数据集上运行
```roslaunch lvi_sam KITTI.launch```
```rosbag play kitti_2011_09_26_drive_0084_synced.bag```

## 致谢和参考
* [Original LVI-SAM](https://github.com/TixiaoShan/LVI-SAM)
* [Cc19245 LVI-SAM-Easyused](https://github.com/Cc19245/LVI-SAM-Easyused)
* [CVlife LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)
* [Alvin Zhihu](https://www.zhihu.com/people/gao-li-dong-62/posts)

## 收藏历史

[![Star History Chart](https://api.star-history.com/svg?repos=shuttworth/E-LVI-SAM&type=Date)](https://star-history.com/#shuttworth/E-LVI-SAM&Date)