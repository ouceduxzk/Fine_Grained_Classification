#ifndef CAFFE_ATTENTION_CROP_LAYER_HPP_
#define CAFFE_ATTENTION_CROP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template<typename Dtype>
class AttentionCropLayer: public Layer<Dtype> {
 public:
    explicit AttentionCropLayer(const LayerParameter& param) :
            Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "AttentionCrop";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }
 protected:
    /// @copydoc AttentionCropLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
	shared_ptr<Caffe::RNG> rng_;
private:
	int out_size = 224;
	Dtype mean_value1 = 109.961;
	Dtype mean_value2 = 127.21;
	Dtype mean_value3 = 123.645;
};

}  // namespace caffe

#endif  // CAFFE_BILINEAR_LAYER_HPP_
