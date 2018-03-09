//random fern training algorithm
#ifndef RF_TRAIN
#define RF_TRAIN

#include "sample.h"
extern char DataPath[300];
namespace random_ferns
{
	const int random_fern_T = 10;
	const int random_fern_K = 200;
	const float random_fern_beta = 1000.0f;

	template<typename Sample>
	class RFTrain
	{
	public:

		vector<Sample> samples_;

		vector<vector<RFSampleVecNode>> sample_vectors_;

		vector<vector<vector<ohday::Vector2f>>> fern_feature_indices_;

		vector<vector<vector<float>>> fern_feature_threshold_;

		vector<vector<vector<ohday::VectorNf>>> forward_function_;

	public:

		RFTrain()
		{
			sample_vectors_.resize(random_fern_T);

			fern_feature_indices_.resize(random_fern_T);
			fern_feature_threshold_.resize(random_fern_T);
			forward_function_.resize(random_fern_T);

			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				fern_feature_indices_[i_t].resize(random_fern_K);
				fern_feature_threshold_[i_t].resize(random_fern_K);
				forward_function_[i_t].resize(random_fern_K);
				for (int i_k = 0; i_k < random_fern_K; i_k++)
				{
					fern_feature_indices_[i_t][i_k].resize(random_fern_Q);
					fern_feature_threshold_[i_t][i_k].resize(random_fern_Q);
					int num_status = pow(float(2), random_fern_Q);
					forward_function_[i_t][i_k].resize(num_status);
				}
			}
		}

		~RFTrain()
		{

		}

		virtual void SetSamples()
		{
		}

		virtual vector<RFSampleVecNode>GenerateRandomVector()
		{
			vector<RFSampleVecNode> t;
			return t;
		}

		void Train(char* ImgFolder, char* ImgType)
		{
			int num_status = pow(float(2), random_fern_Q);

			vector<int> status_omega;
			status_omega.resize(num_status);

			vector<ohday::VectorNf> status_param_delta;
			status_param_delta.resize(num_status);
			int numParams = samples_[0].initial_params_.params_.dims;//length of regression target parameters
			for (int i = 0; i < num_status; i++)
			{
				status_param_delta[i].resize(numParams);//32*42
			}

			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				for (int i_k = 0; i_k < random_fern_K; i_k++)
				{
					int num_status = pow(float(2), random_fern_Q);
					forward_function_[i_t][i_k].resize(num_status);
					for (int i_status = 0; i_status < num_status; i_status++)
					{
						forward_function_[i_t][i_k][i_status].resize(numParams);
					}
				}
			}

			cv::Mat img;

			//outer layer
			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				printf("T %d begins...\n", i_t);
				// 1. Generate random 3D vectors
				sample_vectors_[i_t] = GenerateRandomVector();//随机产生了每个joint的偏移量

				// 2. Sampling
				int numSample = samples_.size();//sample指train文件夹里的样本

				int current_index = -1;
				for (int i_s = 0; i_s < numSample; i_s++)
				{
					if (samples_[i_s].destination_index_ != current_index)
					{
						current_index = samples_[i_s].destination_index_;//train里的下标
						char image_name[300];
						strcpy_s(image_name, ImgFolder);
						char idx_name[100];
						sprintf_s(idx_name, "/%d", current_index);
						strcat_s(image_name, idx_name);
						strcat_s(image_name, ImgType);
						img.release();
						img = cv::imread(image_name, CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
					}
					samples_[i_s].Sampling(img, sample_vectors_[i_t]);//对每个剪影及随机产生的joint偏移量进行采样，numSample个sample共用一个偏移

					SET_FONT_YELLOW;
					printf("Sampling of NO.%d out of %d samples. t = %d .\n", i_s + 1, numSample, i_t);
					SET_FONT_WHITE;
				}
				printf("Sampling done\n");

				// 3. Calculate cov paras
				//a. get feature_map
				int numFeature = samples_[0].features_.size();

				vector<ohday::VectorNf> feature_map;
				feature_map.resize(numFeature);
				for (int i_p = 0; i_p < numFeature; i_p++)
				{
					feature_map[i_p] = ohday::VectorNf(numSample);
					for (int i_s = 0; i_s < numSample; i_s++)
					{
						feature_map[i_p][i_s] = samples_[i_s].features_[i_p];
					}
				}
				printf("Got feature map\n");

				// b. count feature cov and para(a.variance() + b.variance() - 2*cov(a,b)) 
				vector<float> feature_variance;
				vector<vector<float>> feature_para;

				feature_variance.resize(numFeature);
				feature_para.resize(numFeature);

				//#pragma omp parallel for
				for (int i_p = 0; i_p < numFeature; i_p++)
				{
					feature_variance[i_p] = feature_map[i_p].variance();
				}
				printf("Got feature variance\n");

				//#pragma omp parallel for
				for (int i_p = 0; i_p < numFeature - 1; i_p++)
				{
					feature_para[i_p].resize(numFeature);
					for (int j_p = i_p + 1; j_p < numFeature; j_p++)
					{
						feature_para[i_p][j_p] = feature_variance[i_p] + feature_variance[j_p] - 2 * feature_map[i_p].cov(feature_map[j_p]);
					}
				}
				printf("Got feature cov\n");

				// 4. inner loop
				ohday::RandomDevice random_device;
				for (int i_k = 0; i_k < random_fern_K; i_k++)
				{
					printf("\tK %d begins\n", i_k);

					for (int i_f = 0; i_f < random_fern_Q; i_f++)
					{
						// a. Generate random vector and project , get vector_Y
						ohday::VectorNf random_Y = random_device.GetVectorN(numParams, -1, 1);
						float y_length = random_Y.length();

						ohday::VectorNf vector_Y(numSample);

						//#pragma omp parallel for
						for (int i_s = 0; i_s < numSample; i_s++)
						{
							ohday::VectorNf delta_param = samples_[i_s].GetParamDelta();

							vector_Y[i_s] = delta_param.cross(random_Y) / y_length;//根据随机投影将42维降至一维
						}
						float variance_Y = vector_Y.variance();

						// b. Calculate Y_cov 
						vector<float> Y_cov;
						Y_cov.resize(numFeature);
						//#pragma omp parallel for
						for (int i_p = 0; i_p < numFeature; i_p++)
						{
							Y_cov[i_p] = feature_map[i_p].cov(vector_Y);//feature_map某一个特征的所有样本与随机投影后所有样本的cov
						}

						// c. Get fern feature_delta and threshold
						int i_max = 0, j_max = 0;
						float coff_max = -1000000.0f;

						for (int i_p = 0; i_p < numFeature - 1; i_p++)
						{
							for (int j_p = i_p + 1; j_p < numFeature; j_p++)
							{
								bool is_calced = false;
								for (int i_ff = 0; i_ff < random_fern_Q; i_ff++)
								{
									if (int(fern_feature_indices_[i_t][i_k][i_ff].x) == i_p && int(fern_feature_indices_[i_t][i_k][i_ff].y) == j_p)
									{
										is_calced = true;
										break;
									}
								}
								if (is_calced)
									continue;

								float f_numerator = Y_cov[i_p] - Y_cov[j_p];
								float f_denominator = sqrt(feature_para[i_p][j_p] * variance_Y);

								float f_fraction = 0.0f;
								if (f_denominator != 0.0f)
									f_fraction = f_numerator / f_denominator;

								if (f_fraction > coff_max)
								{
									coff_max = f_fraction;
									i_max = i_p;
									j_max = j_p;
								}
							}
						}//至此选出了两个维度特征表示所有的特征，最大化与delta_param的关系
						float min_delta_feature = 1000000.0f;
						float max_delta_feature = -1000000.0f;
						for (int i_s = 0; i_s < samples_.size(); i_s++)
						{
							float current_delta_feature = samples_[i_s].features_[i_max] - samples_[i_s].features_[j_max];

							if (current_delta_feature > max_delta_feature)
								max_delta_feature = current_delta_feature;
							if (current_delta_feature < min_delta_feature)
								min_delta_feature = current_delta_feature;//选出了所有样本这两个维度特征的最大值和最小值
						}

						fern_feature_indices_[i_t][i_k][i_f] = ohday::Vector2f(i_max, j_max);// i_max, j_max是选出来的代表特征的系数

						//CITE BY CHENG: Considering entropy decrease would be appropriate
						fern_feature_threshold_[i_t][i_k][i_f] = random_device.GetFloatLine(min_delta_feature, max_delta_feature);
					}


					// d. Classify every samples using these ferns
					for (int i_s = 0; i_s < samples_.size(); i_s++)
					{
						samples_[i_s].SetStatus(fern_feature_indices_[i_t][i_k], fern_feature_threshold_[i_t][i_k]);//将一个sample分类成5个0或1
					}

					// e. make statistics for each status and get forward function
					for (int i_status = 0; i_status < num_status; i_status++)
					{
						status_omega[i_status] = 0;
						for (int i_d = 0; i_d < numParams; i_d++)
							status_param_delta[i_status][i_d] = 0.0f;
					}

					for (int i_s = 0; i_s < samples_.size(); i_s++)
					{
						int current_status = samples_[i_s].GetStatus();

						status_omega[current_status] ++;
						ohday::VectorNf current_param_delta = samples_[i_s].GetParamDelta();

						for (int i_d = 0; i_d < numParams; i_d++)
						{
							status_param_delta[current_status][i_d] = status_param_delta[current_status][i_d] + current_param_delta[i_d];
						}
					}//所有样本_param_delta相加并归类到32个状态中的current_status中

					for (int i_status = 0; i_status < num_status; i_status++)
					{
						if (status_omega[i_status] != 0)
						{
							float para = 1.0f / ((1 + random_fern_beta / status_omega[i_status]) * status_omega[i_status]);

							for (int i_d = 0; i_d < numParams; i_d++)
							{
								forward_function_[i_t][i_k][i_status][i_d] = status_param_delta[i_status][i_d] * para;//Cao 2013 公式10
							}
						}
					}

					// f. samples update
					for (int i_s = 0; i_s < samples_.size(); i_s++)
					{
						int current_status = samples_[i_s].GetStatus();
						samples_[i_s].UpdateParam(forward_function_[i_t][i_k][current_status]);
					}
				}
			}
		}

		void SaveResult(const char* file_name)
		{
			ofstream save_file(file_name);

			//num outer layer
			save_file << random_fern_T << endl;

			//num inner layer
			save_file << random_fern_K << endl;

			//num of random fern bits
			save_file << random_fern_Q << endl;

			//num of sampling vectors
			int numVec = sample_vectors_[0].size();
			save_file << numVec << endl;

			//num of params
			int numParams = samples_[0].initial_params_.params_.dims;
			save_file << numParams << endl;

			// save sampling vector2f
			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				for (int i_v = 0; i_v < sample_vectors_[i_t].size(); i_v++)
				{
					save_file << sample_vectors_[i_t][i_v].index << ' ' << sample_vectors_[i_t][i_v].angle << ' ';
				}
				save_file << endl;
			}
			save_file << endl;

			// save useful delta_feature and fern thread int each loop
			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				for (int i_k = 0; i_k < random_fern_K; i_k++)
				{
					for (int i_f = 0; i_f < random_fern_Q; i_f++)
					{
						save_file << int(fern_feature_indices_[i_t][i_k][i_f].x) << ' ' << int(fern_feature_indices_[i_t][i_k][i_f].y) << ' ' << fern_feature_threshold_[i_t][i_k][i_f] << ' ';
					}
				}
				save_file << endl;
			}
			save_file << endl;

			// save useful forward_fuction for each status int each loop
			for (int i_t = 0; i_t < random_fern_T; i_t++)
			{
				for (int i_k = 0; i_k < random_fern_K; i_k++)
				{
					for (int i_status = 0; i_status < int(pow(2.0, random_fern_Q)); i_status++)
					{
						for (int i_d = 0; i_d < numParams; i_d++)
						{
							save_file << forward_function_[i_t][i_k][i_status][i_d] << ' ';
						}
					}
				}
				save_file << endl;
			}
			save_file << endl;

			save_file.close();
		}

	};


	/*-----------------Specified Task-------------------*/

	class RFTrainBodyJoints :public RFTrain<RFSample_BodyJoints>
	{

	public:
		RFTrainBodyJoints()
		{
			numIni_ = 6; //modify this!!!
		}

		~RFTrainBodyJoints()
		{
		}

	public:
		int numIni_;

		virtual void SetSamples()
		{
			//load samples'indices
			char samplePath[300];
			strcpy_s(samplePath, DataPath);
			strcat_s(samplePath, "record.txt");

			ifstream ifs(samplePath);
			//number of training data
			int num_data;
			ifs >> num_data;
			//indices for training data
			vector<int> indices_data;
			for (int i = 0; i<num_data; i++)
			{
				int tmp_index;
				ifs >> tmp_index;
				indices_data.push_back(tmp_index);
			}
			ifs.close();

			//for every training data, load its initial joints for constrcting training pair
			for (int i = 0; i < num_data; i++)//num_data,测试画图时更改
			{
				//load destination parameters
				RFSample_BodyJoints new_sample;
				char pose_name[300];
				strcpy_s(pose_name, DataPath);
				char tmp[100];
				sprintf_s(tmp, "Train/%d.txt", indices_data[i]);
				strcat_s(pose_name, tmp);
				new_sample.destination_index_ = indices_data[i];
				new_sample.destination_params_.Read(pose_name);
				new_sample.destination_params_.Joints2Param();

				ohday::RandomDevice rd;
				for (int j = 0; j<numIni_; j++)
				{
					char ini_pose_name[300];
					strcpy_s(ini_pose_name, DataPath);
					char tmp[100];
					sprintf_s(tmp, "Train/%d_%d.txt", indices_data[i], j + 1);
					strcat_s(ini_pose_name, tmp);


					new_sample.initial_index_ = 0; //no use here
					new_sample.initial_params_.Read(ini_pose_name);
					new_sample.initial_params_.Joints2Param();

					new_sample.current_params_ = new_sample.initial_params_;

					samples_.push_back(new_sample);
				}

				SET_FONT_YELLOW;
				printf("Get %d samples\n", samples_.size());
				SET_FONT_WHITE;

			}
		}

		virtual vector<RFSampleVecNode>GenerateRandomVector()
		{
			//modified
			char pathStandardSkeleton[300];
			strcpy_s(pathStandardSkeleton, DataPath);
			char tmp[100] = "Train/1.txt";//,标准骨架格式
			strcat_s(pathStandardSkeleton, tmp);

			ohday::RandomDevice random_device;
			RFBodyJoints standard_joints;
			standard_joints.Read(pathStandardSkeleton);
			standard_joints.Joints2Param();
			ohday::VectorNf standard_landmarks = standard_joints.Param2Landmarks();

			//角度随机生成
			int numVec = 200;
			int numLandmarks = (standard_landmarks.dims) / 2;
			ohday::VectorNf dir = random_device.GetVectorN(numVec, 0, 2 * ohday::ohday_pi);
			vector<RFSampleVecNode>res;
			res.resize(numVec);
			for (int i = 0; i < numVec; i++)
			{
				res[i].angle = dir[i];
				res[i].index = i%numLandmarks;
			}
			return res;
		}
	};
}

#endif //RF_TRAIN