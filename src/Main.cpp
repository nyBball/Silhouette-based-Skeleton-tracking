#include "stdafx.h"
#include "random_fern_train.h"
#include "random_fern_test.h"
#include "math.h"

#define fHFOV 1.0144686707507438
#define fVFOV 0.78980943449644714
#define pi 3.1415926

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

float near_ = 0.3;
float far_ = 10.0;

#define TRAIN 1
#define TEST 2


Eigen::Matrix3d Identity;
Eigen::Vector3d zero;
Eigen::Matrix3d colourCameraProj;

vector<Eigen::Vector3d> iniJoints;
vector<Eigen::Vector3d> outputJoints;
vector<Eigen::Vector3d> gtJoints;


//快速排序
void quick(ohday::Delta *a, int i, int j)
{
	int m, n;
	ohday::Delta temp;
	double k;
	m = i;
	n = j;
	k = a[(i + j) / 2].value;
	do
	{
		while (a[m].value <k&&m<j) m++;
		while (a[n].value >k&&n>i) n--;
		if (m <= n)
		{
			temp = a[m];
			a[m] = a[n];
			a[n] = temp;
			m++;
			n--;
		}
	} while (m <= n);
	if (m < j) quick(a, m, j);
	if (n > i) quick(a, i, n);
}

vector <int> similarSearch(char * filename, const std::vector<std::vector<vector<double>>> &trainData)
{

	vector <int> similarIndex;
	ohday::Delta deltaParam[49482];
	random_ferns::RFSample_BodyJoints testSample;
	testSample.initial_params_.Read(filename);
	//testSample.current_params_ = testSample.initial_params_;

	for (int i = 0; i < 49481; i++)
	{
		double delta = 0;
		for (int j = 0; j < 14; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				delta += fabs(testSample.initial_params_.joints_[j](k) - trainData[i][j][k]);
			}
		}
		deltaParam[i].value = delta;
		deltaParam[i].index = i + 1;
	}
	//排序 选出最近的五个
	quick(deltaParam, 0, 49480);
	for (int k = 0; k < 5; k++)
	{
		similarIndex.push_back(deltaParam[k].index);
	}
	return similarIndex;
}


//database path
char DataPath[300] = "E:/SkeletonTrackingData/mineAllData/";

int main()
{
	Identity(0, 0) = 1.0; Identity(0, 1) = 0.0; Identity(0, 2) = 0.0;
	Identity(1, 0) = 0.0; Identity(1, 1) = 1.0; Identity(1, 2) = 0.0;
	Identity(2, 0) = 0.0; Identity(2, 1) = 0.0; Identity(2, 2) = 1.0;

	zero(0) = 0.0; zero(1) = 0.0; zero(2) = 0.0;

	colourCameraProj(0, 0) = -1058.848004; colourCameraProj(0, 1) = 0.0; colourCameraProj(0, 2) = 985.152484;
	colourCameraProj(1, 0) = 0.0; colourCameraProj(1, 1) = 1062.905217; colourCameraProj(1, 2) = 561.13151;
	colourCameraProj(2, 0) = 0.0; colourCameraProj(2, 1) = 0.0; colourCameraProj(2, 2) = 1.0;


	int TRIGGER = TEST;
	switch (TRIGGER)
	{
	case TRAIN:
	{
		random_ferns::RFTrainBodyJoints trainer;
		trainer.SetSamples();

		char pathTraingImgs[100];
		strcpy_s(pathTraingImgs, DataPath);
		// modify this!!!
		strcat_s(pathTraingImgs, "Train");
		time_t tStart = clock();
		trainer.Train(pathTraingImgs, ".png");//modify
		time_t tEnd = clock();
		float deltaT = float(tEnd - tStart) / CLOCKS_PER_SEC;
		printf("Estimate Body Joints took:%f s,\n", deltaT);
		trainer.SaveResult("../data/train_result.txt");
		printf("Saving result done\n");
	}
	break;

	case TEST:
	{
		random_ferns::RFTest<random_ferns::RFSample_BodyJoints> tester;
		cout << "start reading training result" << endl;
		tester.ReadTrainResult("../data/train_result.txt");
		printf("Reading result done\n");

		cv::Mat test_img;

		char pathImg[300];
		char ini_pose_name[300];
		char out_pose_name[300];
		char tmp1[100];
		char tmp2[100];
		char tmp3[100];
		random_ferns::RFSample_BodyJoints testSample;
		//1th
		test_img = cv::imread("F:/毕设资料2/手臂动作/Train/6.jpg", CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
		testSample.initial_params_.Read("F:/毕设资料2/手臂动作/实测/6.txt");
		testSample.current_params_ = testSample.initial_params_;
		tester.Test(test_img, testSample);
		testSample.current_params_.Param2Joints();
		outputJoints = testSample.current_params_.joints_;
		ofstream ofs1("F:/毕设资料2/手臂动作/实测/7_1.txt");
		for (int i = 0; i<21; i++)
		{
			ofs1 << outputJoints[i](0) << " " << outputJoints[i](1) << " " << outputJoints[i](2) << endl;
		}
		ofs1.close();
		//2...
		for (int idx = 7; idx<4352; idx++)//modify
		{
			strcpy_s(pathImg, DataPath);
			sprintf_s(tmp1, "Train/%d.jpg", idx);//载入测试图像
			strcat_s(pathImg, tmp1);
			test_img = cv::imread(pathImg, CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);

			//load initial
			strcpy_s(ini_pose_name, DataPath);
			sprintf_s(tmp2, "实测/%d_1.txt", idx);//载入初始化骨架
			strcat_s(ini_pose_name, tmp2);
			testSample.initial_params_.Read(ini_pose_name);
			testSample.current_params_ = testSample.initial_params_;

			tester.Test(test_img, testSample);
			testSample.current_params_.Param2Joints();
			outputJoints = testSample.current_params_.joints_;

			strcpy_s(out_pose_name, DataPath);
			sprintf_s(tmp3, "实测/%d_1.txt", idx + 1);//输出文件夹
			strcat_s(out_pose_name, tmp3);
			ofstream ofs(out_pose_name);
			for (int j = 0; j<21; j++)
			{
				ofs << outputJoints[j](0) << " " << outputJoints[j](1) << " " << outputJoints[j](2) << endl;
			}
			ofs.close();
			cout << idx << "th completed" << endl;
		}
		break;
	}

	}
	return 0;
}
