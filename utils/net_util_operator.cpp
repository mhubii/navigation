#include "net_util.h"

namespace caffe2 {

OperatorDef* NetUtil::AddOp(const std::string& name,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs) {
    OperatorDef* op = net_.add_op();
    op->set_type(name);
    
    for (auto input : inputs) {
        op->add_input(input);
    }

    for (auto output : outputs) {
        op->add_output(output);
    }

    return op;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name) {
  
    Argument* arg = op.add_arg();
    arg->set_name(name);

    return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, int value) {

    Argument* arg = net_add_arg(op, name);
    arg->set_i(value);

    return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<int> values) {
    
    auto arg = net_add_arg(op, name);
    
    for (auto value : values) {
        arg->add_ints(value);
    }
    
    return arg;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        const std::string& param) {

    auto op = AddOp("ConstantFill", {}, {param});
    net_add_arg(*op, "shape", shape);

    return op;
}

OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,
                              const std::string& b, const std::string& output,
                              int axis) {

    OperatorDef* op = AddOp("FC", {input, w, b}, {output});
    
    if (axis != 1) {
        
        net_add_arg(*op, "axis", axis);
    }

    return op;
}

OperatorDef* NetUtil::AddXavierFillOp(const std::vector<int>& shape,
                                      const std::string& param) {
    
    auto op = AddOp("XavierFill", {}, {param});
    net_add_arg(*op, "shape", shape);

    return op;
}

} // caffe2