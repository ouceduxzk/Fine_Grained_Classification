#include <vector>
#include <opencv2/core/core.hpp>
#include "caffe/layers/attention_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "opencv2/opencv.hpp"
#include "caffe/util/rng.hpp"
#define ek 0.05
#define core 3
namespace caffe {

template<typename Dtype>
void AttentionCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	Dtype mean_value[3] = { mean_value1, mean_value2, mean_value3 };
    Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* top_data_cpu = top[0]->mutable_cpu_data();
    const Dtype* bottom_data_0 = bottom[0]->cpu_data();
	const Dtype* bottom_data_1 = bottom[1]->cpu_data();
	
	cv::Mat cv_img = cv::Mat(bottom[0]->shape(2), bottom[0]->shape(2), CV_8UC3);
	cv::Mat out_cv_img = cv::Mat(out_size, out_size, CV_8UC3);
	int bottom_index;
	int in_size = bottom[0]->shape(2);
	for (int n = 0; n<bottom[0]->shape(0); n++)
	{
		Dtype a = bottom_data_1[n * 3];
		Dtype b = bottom_data_1[n * 3 + 1];
		Dtype c = bottom_data_1[n * 3 + 2];
		c = c>0.01 * in_size ? c : 0.01 * in_size;
		int w_off = int(a - c > 0 ? a - c : 0);
		int h_off = int(b - c > 0 ? b - c : 0);
		int w_end = int((a + c) < in_size ? (a + c) : in_size);
		int h_end = int((b + c) < in_size ? (b + c) : in_size);
		for (int i = 0; i < bottom[0]->shape(2); i++)
		{
			uchar* ptr = cv_img.ptr<uchar>(i);
			int img_index = 0;
			for (int j = 0; j < bottom[0]->shape(2); j++)
			{
				for (int k = 0; k < 3; k++)
				{
					bottom_index = n * bottom[0]->count(1) + (k * bottom[0]->shape(2) + i) * bottom[0]->shape(2) + j;
					ptr[img_index++] = bottom_data_0[bottom_index] + mean_value[k];
				}
			}
		}

		cv::Rect roi(w_off, h_off, w_end - w_off, h_end - h_off);
		cv::Mat cv_cropped_img = cv_img(roi);

		cv::resize(cv_cropped_img, out_cv_img, out_cv_img.size(), 0, 0, cv::INTER_LINEAR);

		int top_index;
		for (int i = 0; i < out_size; i++)
		{
			const uchar* ptr = out_cv_img.ptr<uchar>(i);
			int img_index = 0;
			for (int j = 0; j < out_size; j++)
			{

				for (int k = 0; k < 3; k++)
				{	
					Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					top_index = n * top[0]->count(1) + (k * out_size + i) * out_size + j;
					top_data_cpu[top_index] = pixel - mean_value[k];
				}


			}
		}
	}

	caffe_gpu_memcpy(top[0]->count() * sizeof(Dtype), top_data_cpu, top_data);
}


template<typename Dtype>
Dtype H(Dtype x){
	return 1 / (1 + exp(-ek*x));
}

template<typename Dtype>
Dtype diff_H(Dtype x){
	return ek * exp(-ek*x) / ((1 + exp(-ek*x))*(1 + exp(-ek*x)));
}

template<typename Dtype>
Dtype F(Dtype a, Dtype b, Dtype c, Dtype x, Dtype y) {
	return (H(x - (a - c)) - H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)));
}

template<typename Dtype>
Dtype diff_F_a(Dtype a, Dtype b, Dtype c, Dtype x, Dtype y) {
	return (diff_H(x - (a - c)) - diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)));
}

template<typename Dtype>
Dtype diff_F_b(Dtype a, Dtype b, Dtype c, Dtype x, Dtype y) {
	return (diff_H(y - (b - c)) - diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c)));
}

template<typename Dtype>
Dtype diff_F_c(Dtype a, Dtype b, Dtype c, Dtype x, Dtype y) {
	return -((diff_H(y - (b - c)) + diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c))) + (diff_H(x - (a - c)) + diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c))))+0.005;
}



template<typename Dtype>
void AttentionCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
	Dtype* bottom_diff_cpu = bottom[1]->mutable_cpu_diff();
	const Dtype* bottom_data = bottom[1]->cpu_data();
	int in_size = bottom[0]->shape(2);
	caffe_set(bottom[1]->count(), Dtype(0.0), bottom_diff_cpu);
	int count_1 = top[0]->count(1);
	int count_2 = top[0]->count(2);
	int count_3 = top[0]->count(3);
	Dtype a;
	Dtype b;
	Dtype c;
	for (int i = 0; i < top[0]->shape(0); i++)
	{
		a = bottom_data[i * 3];
		b = bottom_data[i * 3 + 1];
		c = bottom_data[i * 3 + 2];
		for (int j = 0; j < top[0]->shape(1); j++)
		{
			Dtype max_diff = 0;
			for (int k = 0; k < top[0]->shape(2); k++)
			{
				for (int l = 0; l < top[0]->shape(3); l++)
				{
					int top_index = i * count_1 + j * count_2 + k * count_3 + l;
					Dtype top = top_diff[top_index]>0 ? top_diff[top_index] : -top_diff[top_index];
					if (top > max_diff)
					{
						max_diff = top;
					}
				}
			}
			for (int k = 0; k < top[0]->shape(2); k++)
			{
				for (int l = 0; l < top[0]->shape(3); l++)
				{
					int top_index = i * count_1 + j * count_2 + k * count_3 + l;
					Dtype top = top_diff[top_index]>0 ? top_diff[top_index] : -top_diff[top_index];
					if (max_diff > 0)
					{
						top = top / max_diff * 0.0000001;
					}
					Dtype x = a - c + 2 * l * c / out_size;
					Dtype y = b  - c + 2 * k * c / out_size;
					bottom_diff_cpu[3 * i + 0] += top * diff_F_a(a, b, c, x, y);
					bottom_diff_cpu[3 * i + 1] += top * diff_F_b(a, b, c, x, y);
					bottom_diff_cpu[3 * i + 2] += top * diff_F_c(a, b, c, x, y);
					
				}
			}
		}
	}

	caffe_gpu_memcpy(bottom[1]->count() * sizeof(Dtype), bottom_diff_cpu, bottom_diff);


}


INSTANTIATE_LAYER_GPU_FUNCS(AttentionCropLayer);

}  // namespace caffe
