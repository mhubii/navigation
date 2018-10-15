#ifndef DQN_H_
#define DQN_H_

#include <string>

#include "model_base.h"

namespace caffe2 {

class DQNModel : public ModelBase {

public:
    
    DQNModel(NetDef& init_net, NetDef& predict_net)
        : ModelBase(init_net, predict_net) {    };

    OperatorDef* AddConvOps(const std::string &input, const std::string &output,
                            int in_size, int out_size, int stride, int padding,
                            int kernel) {

        init_.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
        predict_.AddInput(output + "_w");
        init_.AddConstantFillOp({out_size}, output + "_b");
        predict_.AddInput(output + "_b");
        predict_.AddConvOp(input, output + "_w", output + "_b", output,
                           stride, padding, kernel);

        return predict_.AddReluOp(output, output);
    };

    OperatorDef* AddFc(const std::string &input, const std::string &output,
                       int in_size, int out_size, bool relu) {

        init_.AddXavierFillOp({out_size, in_size}, output + "_w");
        predict_.AddInput(output + "_w");
        init_.AddConstantFillOp({out_size}, output + "_b");
        predict_.AddInput(output + "_b");
        
        return predict_.AddFcOp(input, output + "_w", output + "_b", output);
    };

    OperatorDef* AddTrain();

    void Add(int out_size, bool train = false) {

        // TODO
    };

};

} // caffe2.

#endif // DQN_H_
