#include <vector>
#include <algorithm>
#include <iostream>

#include "caffe/layers/rank_loss2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void RankLoss2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{

}

template<typename Dtype>
void RankLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

}
INSTANTIATE_LAYER_GPU_FORWARD(RecurrentLayer);


}