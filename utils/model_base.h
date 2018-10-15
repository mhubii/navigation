#ifndef MODEL_BASE_H_
#define MODEL_BASE_H_

#include <caffe2/core/net.h>

#include "net_util.h"

namespace caffe2 {

class ModelBase {

public:

	ModelBase(NetDef& init_net, NetDef& predict_net);

public:

	NetUtil init_;
	NetUtil predict_;
};
} // caffe2

#endif // MODEL_BASE_H_
