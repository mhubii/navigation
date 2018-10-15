#include "model_base.h"

namespace caffe2 {

ModelBase::ModelBase(NetDef& init_net, NetDef& predict_net)
	: init_(init_net), predict_(predict_net) {

};
} // caffe2