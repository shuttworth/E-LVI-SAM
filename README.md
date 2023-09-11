# E-LVI-SAM

## Easier and Simplified Implementation of LVI-SAM

E-LVI-SAM is a easier and simplified implementation of [LVI-SAM](https://github.com/TixiaoShan/LVI-SAM).The main modifications are as follows:

* The code structure has been optimized, and the unnecessary codes have been deleted, followed [LVI-SAM-Easyused](https://github.com/Cc19245/LVI-SAM-Easyused)
* Detailed Chinese notes are added to the code, follewed [Zhihu](https://www.zhihu.com/people/gao-li-dong-62/posts) and [LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)
* Using a docker image created before in [LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments), which can save local environment configuration time

## Prerequisites
### with Docker
We strongly recommend the docker


```docker pull liangjinli/slam-docker:v1.2```


more details can be seen in [LVI-SAM Docker使用图文简介](https://github.com/shuttworth/E-LVI-SAM/blob/master/LVI-SAM%20Docker%E4%BD%BF%E7%94%A8%E5%9B%BE%E6%96%87%E7%AE%80%E4%BB%8B.pdf)

### without Docker
See in [dependency](https://github.com/TixiaoShan/LVI-SAM#dependency)


## Map Result
<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/result.png" width="70%">
</center>

<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/gate_01_v1.gif" width="70%">
</center>

<center>
<img src="https://github.com/shuttworth/Record_Issues_For_LVI-SAM_detailed_comments/blob/main/img/street_08_v1.gif" width="70%">
</center>


## Run the package on different datasets
### in Official dataset
```roslaunch lvi_sam run_official.launch```
```rosbag play handheld.bag```

### in M2DGR dataset
```roslaunch lvi_sam M2DGR.launch```
```rosbag play gate_01.bag```

### in UrbanNavDataset
```roslaunch lvi_sam UrbanNavDataset.launch```
```rosbag play 2020-03-14-16-45-35.bag```

### in KITTI raw dataset
```roslaunch lvi_sam KITTI.launch```
```rosbag play kitti_2011_09_26_drive_0084_synced.bag```

## Acknowledgement
* [Original LVI-SAM](https://github.com/TixiaoShan/LVI-SAM)
* [Cc19245 LVI-SAM-Easyused](https://github.com/Cc19245/LVI-SAM-Easyused)
* [CVlife LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)
* [Alvin Zhihu](https://www.zhihu.com/people/gao-li-dong-62/posts)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=shuttworth/E-LVI-SAM&type=Date)](https://star-history.com/#shuttworth/E-LVI-SAM&Date)
