#ifndef CAFFE_RANK_LOSS2_LAYER_HPP_
#define CAFFE_RANK_LOSS2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RankLoss2Layer:public LossLayer<Dtype>
{
public:
    explicit RankLoss2Layer(const LayerParameter& param):LossLayer<Dtype>(param){}
    //virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const
    {
        return "RankLoss2";
    }

protected:
   // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //                        const vector<Blob<Dtype>*>& top);
   //
   // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
   //                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}   //namespace caffe

#endif // CAFFE_RANK_LOSS2_LAYER_HPP_