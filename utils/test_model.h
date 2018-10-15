#ifndef TEST_MODEL_H_
#define TEST_MODEL_H_

#include <caffe2/core/net.h>
#include <string>

#include "model_base.h"

namespace caffe2 {

class TestModel : public ModelBase {

public:

    TestModel(NetDef& init, NetDef& predict)
        : ModelBase(init, predict) {    };

    OperatorDef* AddFc(const std::string &input, const std::string &output,
                       int in_size, int out_size) {

        init_.AddXavierFillOp({out_size, in_size}, output + "_w");
        predict_.AddInput(output + "_w");
        init_.AddConstantFillOp({out_size}, output + "_b");
        predict_.AddInput(output + "_b");
        
        return predict_.AddFcOp(input, output + "_w", output + "_b", output);
    };

    //OperatorDef* AddTrain();

    OperatorDef* Add(int out_size, bool train = false) {

        predict_.SetName("TestModel");
        std::string input = "data";

        std::string layer = input;
        predict_.AddInput(layer);
        //layer = AddFc(layer, "fc1", 10, 10)->output(0);

        predict_.AddOutput(layer);
    };

};

} // caffe2

#endif // TEST_MODLE_H_