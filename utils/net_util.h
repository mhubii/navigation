#ifndef NET_H_
#define NET_H_

#include <caffe2/core/net.h>
#include <caffe2/core/operator.h>

namespace caffe2 {

class NetUtil {

public:

    NetUtil(NetDef& net, const std::string& name = "") 
        : net_(net) {

        if (name.size()) {

            SetName(name);
        }
    };

    OperatorDef* AddOp(const std::string& name,
                       const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs);

    OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
                                   const std::string& param);

    OperatorDef* AddConvOp(const std::string& input, const std::string& w,
                           const std::string& b, const std::string& output,
                           int stride, int padding, int kernel, int group = 0,
                           const std::string& order = "NCHW");

    OperatorDef* AddFcOp(const std::string& input, const std::string& w,
                         const std::string& b, const std::string& output,
                         int axis = 1);

    OperatorDef* AddReluOp(const std::string& input, const std::string& output);

    // OperatorDef* AddSmoothL1Op(const std::string& input, const std::string& )

    OperatorDef* AddXavierFillOp(const std::vector<int>& shape,
                                 const std::string& param);

    void AddInput(const std::string input);

    void AddOutput(const std::string output);

    const std::string& Input(int i);

    const std::string& Output(int i);

    void SetName(const std::string name);

public:

    NetDef& net_;

};
} // caffe2

#endif // NET_H_