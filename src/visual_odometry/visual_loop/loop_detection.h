#pragma once

#include "parameters.h"
#include "keyframe.h"

using namespace DVision;
using namespace DBoW2;

class LoopDetector
{
public:

	BriefDatabase db; // 描述子数据库
	BriefVocabulary* voc; // 字典

	map<int, cv::Mat> image_pool; // 用于可视化和Debug

	list<KeyFrame*> keyframelist;

	LoopDetector(); // 默认构造函数
	void loadVocabulary(std::string voc_path); // 按照路径加载字典，初始化描述子数据库db
	
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
	void addKeyFrameIntoVoc(KeyFrame* keyframe);
	KeyFrame* getKeyFrame(int index);

	void visualizeKeyPoses(double time_cur);

	int detectLoop(KeyFrame* keyframe, int frame_index);
};
