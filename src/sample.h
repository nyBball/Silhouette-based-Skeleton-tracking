//sample for random fern training
#ifndef RF_SAMPLE
#define RF_SAMPLE

#include"stdafx.h"
#include "utils.h"

extern Eigen::Matrix3d Identity;
extern Eigen::Vector3d zero;
extern Eigen::Matrix3d colourCameraProj;
using namespace cv;

namespace random_ferns
{
	const int random_fern_Q = 5;

	class RFParam
	{

	public:
		ohday::VectorNf params_;

	public:
		RFParam()
		{
		}

		RFParam(const RFParam & p)
		{
			params_ = p.params_;
		}

		~RFParam()
		{
		}

		//retrive landmark position with respect to parameters
		virtual ohday::VectorNf Param2Landmarks()
		{
			return params_;
		}

		//load parameters from file
		virtual void Read(const char* file_name)
		{
		}

		//draw landmarks
		virtual void DrawLandmarks(cv::Mat img)
		{
			ohday::VectorNf Landmarks = Param2Landmarks();
			int numLandmarks = Landmarks.dims / 2;

			for (int i = 0; i<numLandmarks; i++)
			{
				cv::Point point;

				point.x = Landmarks[2 * i];
				point.y = Landmarks[2 * i + 1];

				cv::circle(img, point, 2, cv::Scalar(0, 255, 0), -1);
			}
		}

	};

	struct RFSampleVecNode
	{
		double angle;
		int index;
	};

	template<typename Param>
	class RFSample
	{

	public:
		int initial_index_;
		int destination_index_;
		vector<int> status_;

		Param initial_params_;
		Param current_params_;
		Param destination_params_;
		vector<float> features_;

	public:
		RFSample()
		{
			initial_index_ = destination_index_ = -1;

			status_.resize(random_fern_Q);
		}

		RFSample(const RFSample & s)
		{
			initial_index_ = s.initial_index_;
			initial_params_ = s.initial_params_;

			current_params_ = s.current_params_;

			destination_index_ = s.destination_index_;
			destination_params_ = s.destination_params_;

			features_ = s.features_;
			status_ = s.status_;
		}

		~RFSample()
		{
		}

		virtual void Sampling(cv::Mat& img, vector<RFSampleVecNode> &sample_vector)
		{
		}

		ohday::VectorNf GetParamDelta()
		{
			int numParam = initial_params_.params_.dims;
			ohday::VectorNf res(numParam);

			res = destination_params_.params_ - current_params_.params_;

			return res;
		}

		void SetStatus(vector<ohday::Vector2f>& v_index, vector<float>& v_threshold)
		{
			for (int i = 0; i < random_fern_Q; i++)
			{
				int index_a = int(v_index[i].x);
				int index_b = int(v_index[i].y);
				float delta_feature = features_[index_a] - features_[index_b];

				if (delta_feature > v_threshold[i])
					status_[i] = 1;
				else
					status_[i] = 0;
			}
		}

		int GetStatus()
		{
			int res = 0;
			for (int i = 0; i < random_fern_Q; i++)
			{
				res += int(pow(float(2), random_fern_Q - 1 - i)) * status_[i];
			}

			return res;
		}

		void UpdateParam(ohday::VectorNf v)
		{
			int numParams = current_params_.params_.dims;
			for (int i_d = 0; i_d < numParams; i_d++)
			{
				current_params_.params_[i_d] = current_params_.params_[i_d] + v[i_d];
			}
		}

	};


	/*-----------------Specified Task-------------------*/

	//3D joints regression
	class RFBodyJoints :public RFParam
	{
	public:
		RFBodyJoints()
		{
			numJoints_ = 14;

			// modify this!!!(computed)
			rot2ColourCamera_ = &Identity;
			translate2ColourCamera_ = &zero;
			projectCamera_ = &colourCameraProj;
		}

		~RFBodyJoints()
		{
		}

	public:
		int numJoints_;
		vector<Eigen::Vector3d> joints_;

		Eigen::Matrix3d* rot2ColourCamera_;
		Eigen::Vector3d* translate2ColourCamera_;
		Eigen::Matrix3d* projectCamera_;

	public:
		virtual void Read(const char* file_name)
		{
			ifstream ifs(file_name);

			joints_.resize(numJoints_);

			for (int i = 0; i<numJoints_; i++)
			{
				Eigen::Vector3d tmpJoint;

				ifs >> tmpJoint(0);
				ifs >> tmpJoint(1);
				ifs >> tmpJoint(2);

				//translate to colour camera coordinate system
				tmpJoint = (*rot2ColourCamera_)*tmpJoint + (*translate2ColourCamera_);

				joints_[i] = tmpJoint;
			}

			Joints2Param();
		}

		void Joints2Param()//joint是3d的节点，用二维数组表示。param是3d joint的三维坐标，用一维数组表示。landmark是2d的
		{
			params_.resize(3 * numJoints_);
			for (int i = 0; i<numJoints_; i++)
			{
				params_[3 * i] = joints_[i](0);
				params_[3 * i + 1] = joints_[i](1);
				params_[3 * i + 2] = joints_[i](2);
			}
		}

		void Param2Joints()
		{
			for (int i = 0; i<params_.dims; i++)
			{
				joints_[i / 3](i % 3) = params_[i];
			}
		}

		virtual ohday::VectorNf Param2Landmarks()
		{
			Param2Joints();

			//project 3D joints to 2D silouette
			ohday::VectorNf landmarks;
			landmarks.resize(2 * numJoints_);
			for (int i = 0; i<numJoints_; i++)
			{
				Eigen::Vector3d homo = (*projectCamera_) * joints_[i];

				landmarks[2 * i] = homo(0) / homo(2);
				landmarks[2 * i + 1] = homo(1) / homo(2);
			}

			return landmarks;
		}
	};

	class RFSample_BodyJoints :public RFSample<RFBodyJoints>
	{
	public:

		virtual void Sampling(cv::Mat& siluhouette, vector<RFSampleVecNode> &sample_vector)
		{
			int numVec = sample_vector.size();
			features_.clear();
			features_.resize(numVec);
			ohday::VectorNf landmarks = current_params_.Param2Landmarks();
			current_params_.Param2Joints();

			//get features
			for (int i_p = 0; i_p < numVec; i_p++)
			{
				int i_key = sample_vector[i_p].index;
				double x = landmarks[i_key * 2];
				double y = landmarks[i_key * 2 + 1];
				cv::Point pt((int)x, (int)y);
				int intensity = siluhouette.at<uchar>(pt);
				int counter = 0;
				//在剪影内
				if (intensity > 125)
				{
					//
					if ((sample_vector[i_p].angle >= 0 && sample_vector[i_p].angle <= M_PI / 4) ||
						(sample_vector[i_p].angle >= 7 * M_PI / 4 && sample_vector[i_p].angle <= 2 * M_PI))
					{
						while (intensity > 125)
						{
							x = x + 1;
							y = y + tan(sample_vector[i_p].angle);
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = fabs(counter / cos(sample_vector[i_p].angle));
					}

					//
					else if (sample_vector[i_p].angle > 3 * M_PI / 4 && sample_vector[i_p].angle <= 5 * M_PI / 4)
					{
						while (intensity > 125)
						{
							x = x - 1;
							y = y - tan(sample_vector[i_p].angle);
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = fabs(counter / cos(sample_vector[i_p].angle));
					}

					//
					else if (sample_vector[i_p].angle >M_PI / 4 && sample_vector[i_p].angle <= 3 * M_PI / 4 &&
						sample_vector[i_p].angle != M_PI / 2)
					{
						while (intensity > 125)
						{
							y = y + 1;
							x = x + 1 / tan(sample_vector[i_p].angle);
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = fabs(counter / sin(sample_vector[i_p].angle));
					}
					//
					else if (sample_vector[i_p].angle >5 * M_PI / 4 && sample_vector[i_p].angle < 7 * M_PI / 4 &&
						sample_vector[i_p].angle != 3 * M_PI / 2)
					{
						while (intensity > 125)
						{
							y = y - 1;
							x = x - 1 / tan(sample_vector[i_p].angle);
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = fabs(counter / sin(sample_vector[i_p].angle));
					}
					//在剪影90度情况
					else if (sample_vector[i_p].angle == M_PI / 2)
					{
						while (intensity > 125)
						{
							y = y + 1;
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = counter;
					}
					//在剪影270度情况
					else
					{
						while (intensity > 125)
						{
							y = y - 1;
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
						}
						features_[i_p] = counter;
					}
				}

				//点在剪影外部情况
				else
				{
					//1-1
					if (sample_vector[i_p].angle >= 0 && sample_vector[i_p].angle <= M_PI / 4)
					{
						double colLimit = siluhouette.cols - x - 10;
						double rowLimit = siluhouette.rows - y - 10;
						while (intensity <= 125)
						{
							x = x + 1;
							y = y + tan(sample_vector[i_p].angle);//y增大
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= colLimit || fabs(counter*tan(sample_vector[i_p].angle)) >= rowLimit)
								break;
						}
						features_[i_p] = -fabs(counter / cos(sample_vector[i_p].angle));
					}

					//1-2
					else if (sample_vector[i_p].angle >= 7 * M_PI / 4 && sample_vector[i_p].angle <= 2 * M_PI)
					{
						double y0 = y;
						double colLimit = siluhouette.cols - x - 10;
						while (intensity <= 125)
						{
							x = x + 1;
							y = y + tan(sample_vector[i_p].angle);//y减小
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= colLimit || fabs(counter*tan(sample_vector[i_p].angle)) >= y0 - 10)
								break;
						}
						features_[i_p] = -fabs(counter / cos(sample_vector[i_p].angle));
					}

					//2-1
					else if (sample_vector[i_p].angle > 3 * M_PI / 4 && sample_vector[i_p].angle <= M_PI)
					{
						double x0 = x;
						double rowLimit = siluhouette.rows - y - 10;
						while (intensity <= 125)
						{
							x = x - 1;
							y = y - tan(sample_vector[i_p].angle);//y增大
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= x0 - 10 || fabs(counter * tan(sample_vector[i_p].angle)) >= rowLimit)
								break;
						}
						features_[i_p] = -fabs(counter / cos(sample_vector[i_p].angle));
					}
					//2-2
					else if (sample_vector[i_p].angle > M_PI  && sample_vector[i_p].angle <= 5 * M_PI / 4)
					{
						double x0 = x;
						double y0 = y;
						while (intensity <= 125)
						{
							x = x - 1;
							y = y - tan(sample_vector[i_p].angle);//y减小
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= x0 - 10 || fabs(counter *tan(sample_vector[i_p].angle)) >= y0 - 10)
								break;
						}
						features_[i_p] = -fabs(counter / cos(sample_vector[i_p].angle));
					}
					//3-1
					else if (sample_vector[i_p].angle > M_PI / 4 && sample_vector[i_p].angle < M_PI / 2)
					{
						double colLimit = siluhouette.cols - x - 10;
						double rowLimit = siluhouette.rows - y - 10;
						while (intensity <= 125)
						{
							y = y + 1;
							x = x + 1 / tan(sample_vector[i_p].angle);//x增大
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= rowLimit || fabs(counter / tan(sample_vector[i_p].angle)) >= colLimit)
								break;
						}

						features_[i_p] = -fabs(counter / sin(sample_vector[i_p].angle));
					}
					//3-2
					else if (sample_vector[i_p].angle > M_PI / 2 && sample_vector[i_p].angle <= 3 * M_PI / 4)
					{
						double x0 = x;
						double rowLimit = siluhouette.rows - y - 10;
						while (intensity <= 125)
						{
							y = y + 1;
							x = x + 1 / tan(sample_vector[i_p].angle);//x减小
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= rowLimit || fabs(counter / tan(sample_vector[i_p].angle)) >= x0 - 10)
								break;
						}

						features_[i_p] = -fabs(counter / sin(sample_vector[i_p].angle));
					}

					//4-1
					else if (sample_vector[i_p].angle > 5 * M_PI / 4 && sample_vector[i_p].angle < 3 * M_PI / 2)
					{
						double x0 = x;
						double y0 = y;
						while (intensity <= 125)
						{
							y = y - 1;
							x = x - 1 / tan(sample_vector[i_p].angle);//x减小
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= y0 - 10 || fabs(counter / tan(sample_vector[i_p].angle)) >= x0 - 10)
								break;
						}
						features_[i_p] = -fabs(counter / sin(sample_vector[i_p].angle));
					}
					//4-2
					else if (sample_vector[i_p].angle > 3 * M_PI / 2 && sample_vector[i_p].angle < 7 * M_PI / 4)
					{
						double y0 = y;
						double colLimit = siluhouette.cols - x - 10;
						while (intensity <= 125)
						{
							y = y - 1;
							x = x - 1 / tan(sample_vector[i_p].angle);//x增大
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= y0 - 10 || fabs(counter / tan(sample_vector[i_p].angle)) >= colLimit)
								break;
						}
						features_[i_p] = -fabs(counter / sin(sample_vector[i_p].angle));
					}
					//不在剪影90度情况
					else if (sample_vector[i_p].angle == M_PI / 2)
					{
						double rowLimit = siluhouette.rows - y - 10;
						while (intensity <= 125)
						{
							y = y + 1;
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= rowLimit)
								break;
						}
						features_[i_p] = -counter;
					}
					//不在剪影270度情况
					else
					{
						double y0 = y;
						while (intensity <= 125)
						{
							y = y - 1;
							cv::Point pt((int)x, (int)y);
							intensity = siluhouette.at<uchar>(pt);
							counter++;
							if (counter >= y0 - 10)
								break;
						}
						features_[i_p] = -counter;
					}
				}
			}


			/*---just for test---*/
			/*int height = siluhouette.rows; int width = siluhouette.cols;

			IplImage* image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
			for (int row = 0; row<height; row++)
			{
				unsigned char* ptr = (unsigned char*)(image->imageData + row*image->widthStep);

				for (int col = 0; col<width; col++)
				{
					cv::Point pt((int)col, (int)row);
					double val = siluhouette.at<uchar>(pt);

					ptr[3 * col] = val;
					ptr[3 * col + 1] = val;
					ptr[3 * col + 2] = val;
				}
			}

			Mat cpy = cvarrToMat(image);
			for (int i_p = 0; i_p < numVec; i_p++)
			{
				int i_key = sample_vector[i_p].index;
				double x = landmarks[i_key * 2] + fabs(features_[i_p])* cos(sample_vector[i_p].angle);
				double y = landmarks[i_key * 2 + 1] + fabs(features_[i_p]) * sin(sample_vector[i_p].angle);
				double intensity = siluhouette.at<uchar>(cvPoint(landmarks[i_key * 2], landmarks[i_key * 2 + 1]));
				if (intensity>125)
					cv::line(cpy, cvPoint(landmarks[i_key * 2], landmarks[i_key * 2 + 1]), cvPoint(x, y), cvScalar(0, 255, 0));
				else
					cv::line(cpy, cvPoint(landmarks[i_key * 2], landmarks[i_key * 2 + 1]), cvPoint(x, y), cvScalar(255, 0, 0));
			}

			ohday::VectorNf landmarks_dest = destination_params_.Param2Landmarks();
			for (int i = 0; i<landmarks_dest.dims / 2; i++)
			{
				cv::circle(cpy, cvPoint(landmarks_dest[2 * i], landmarks_dest[2 * i + 1]), 3, cvScalar(0, 0, 255));
			}

			cv::imshow("tst_", cpy);

			cvReleaseImage(&image);
			cvWaitKey();*/
		}
	};
}

#endif //RF_SAMPLES