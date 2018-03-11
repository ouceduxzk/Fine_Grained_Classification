#include <vector>
#include <algorithm>
#include <iostream>

#include "caffe/layers/rank_loss2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RankLoss2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
    const Dtype* pred = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num(); //number of samples
    int count = bottom[0]->count(); //length of data
    int dim = count / num; // dim of classes

    Dtype margin=this->layer_param_.rank_loss2_param().margin();
    Dtype scale_num = this->layer_param_.rank_loss2_param().scale_num();
    Dtype loss = Dtype(0.0);
    for(int i=0;i<num;i++)
    {
        for(int j=0;j<scale_num-1;j++)
        {
            int scale1_index = i*dim+dim/3*j+(int)label[i];
            int scale2_index = i*dim+dim/3*(j+1)+(int)label[i];
            loss += std::max(Dtype(0), pred[scale1_index]-pred[scale2_index]+margin);
        }
    }
    top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void RankLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    //const Dtype top_cpu = top[0]->cpu_data()[0];
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    const Dtype* pred = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num; // dim of classes
    //std::cout << "num: " << num << "dim: " << dim << ", loss_weight: " << loss_weight << ", top_cpu: " << top[0]->cpu_data()[0] << std::endl;

    Dtype margin=this->layer_param_.rank_loss2_param().margin();
    Dtype scale_num = this->layer_param_.rank_loss2_param().scale_num();
    memset(bottom_diff, Dtype(0), count*sizeof(Dtype));
    for(int i=0;i<num;i++)
    {
        for(int j=0;j<scale_num-1;j++)
        {
            int scale1_index = i*dim+dim/3*j+(int)label[i];
            int scale2_index = i*dim+dim/3*(j+1)+(int)label[i];
            if(pred[scale1_index]-pred[scale2_index]+margin>0)
            {
                bottom_diff[scale1_index] += loss_weight;
                bottom_diff[scale2_index] -= loss_weight;
            }           
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(RankLoss2Layer);
#endif

INSTANTIATE_CLASS(RankLoss2Layer);
REGISTER_LAYER_CLASS(RankLoss2);
}   // namespace caffe