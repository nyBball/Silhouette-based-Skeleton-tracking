#ifndef RF_TEST
#define RF_TEST

#include "sample.h"

namespace random_ferns
{
	template<typename Sample>
	class RFTest
	{

	public:
		int num_outer_layer_;
		int num_inner_layer_;
		int num_q_;
		int num_vec_;
		int num_params_;

	public:
		vector<vector<RFSampleVecNode>> sample_vectors_;

		vector<vector<vector<ohday::Vector2f>>> fern_feature_indices_;

		vector<vector<vector<float>>> fern_feature_threshold_;

		vector<vector<vector<ohday::VectorNf>>> forward_function_;

	public:
		RFTest()
		{
		}
		~RFTest()
		{
		}

		void ReadTrainResult(const char* file_name)
		{
			ifstream ifs(file_name);

			//num outer layer
			ifs >> num_outer_layer_;

			//num inner layer
			ifs >> num_inner_layer_;

			//num of random fern bits
			ifs >> num_q_;

			//num of sampling vectors
			ifs >> num_vec_;

			//num of params
			ifs >> num_params_;

			//allocate memory for random fern testing
			sample_vectors_.resize(num_outer_layer_);

			fern_feature_indices_.resize(num_outer_layer_);
			fern_feature_threshold_.resize(num_outer_layer_);
			forward_function_.resize(num_outer_layer_);

			for (int i_t = 0; i_t < num_outer_layer_; i_t++)
			{
				fern_feature_indices_[i_t].resize(num_inner_layer_);
				fern_feature_threshold_[i_t].resize(num_inner_layer_);
				forward_function_[i_t].resize(num_inner_layer_);

				sample_vectors_[i_t].resize(num_vec_);

				for (int i_k = 0; i_k < num_inner_layer_; i_k++)
				{
					fern_feature_indices_[i_t][i_k].resize(num_q_);
					fern_feature_threshold_[i_t][i_k].resize(num_q_);

					int num_status = pow(float(2), num_q_);
					forward_function_[i_t][i_k].resize(num_status);
					for (int i_status = 0; i_status < num_status; i_status++)
					{
						forward_function_[i_t][i_k][i_status].resize(num_params_);
					}
				}
			}

			//load train result

			for (int i_t = 0; i_t < num_outer_layer_; i_t++)
			{
				for (int i_v = 0; i_v < sample_vectors_[i_t].size(); i_v++)
				{
					ifs >> sample_vectors_[i_t][i_v].index >> sample_vectors_[i_t][i_v].angle;
				}
			}

			// load delta_features and thresholds
			for (int i_t = 0; i_t < num_outer_layer_; i_t++)
			{
				for (int i_k = 0; i_k < num_inner_layer_; i_k++)
				{
					for (int i_f = 0; i_f < num_q_; i_f++)
					{
						ifs >> fern_feature_indices_[i_t][i_k][i_f].x >> fern_feature_indices_[i_t][i_k][i_f].y >> fern_feature_threshold_[i_t][i_k][i_f];
					}
				}
			}

			// load forward_fuctions
			for (int i_t = 0; i_t < num_outer_layer_; i_t++)
			{
				for (int i_k = 0; i_k < num_inner_layer_; i_k++)
				{
					for (int i_status = 0; i_status < int(pow(2.0, num_q_)); i_status++)
					{
						for (int i_d = 0; i_d < num_params_; i_d++)
						{
							ifs >> forward_function_[i_t][i_k][i_status][i_d];
						}
					}
				}
			}

			ifs.close();
		}

		void Test(cv::Mat & img, Sample & testSample)
		{
			for (int i_t = 0; i_t < num_outer_layer_; i_t++)
			{
				testSample.Sampling(img, sample_vectors_[i_t]);

				ohday::VectorNf total_forward_fuction(num_params_);

				for (int i_k = 0; i_k < num_inner_layer_; i_k++)
				{
					testSample.SetStatus(fern_feature_indices_[i_t][i_k], fern_feature_threshold_[i_t][i_k]);

					int current_status = testSample.GetStatus();

					total_forward_fuction = total_forward_fuction + forward_function_[i_t][i_k][current_status];

				}
				testSample.UpdateParam(total_forward_fuction);
			}
		}
	};
}

#endif //RF_TEST