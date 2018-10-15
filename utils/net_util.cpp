#include "net_util.h"

namespace caffe2 {

void NetUtil::AddInput(const std::string input) {
    
    net_.add_external_input(input);
}

void NetUtil::AddOutput(const std::string output) {
  
    net_.add_external_output(output);
}

const std::string& NetUtil::Input(int i) {

    CAFFE_ENFORCE(net_.external_input_size() != 0, net_.name(),
                  " doesn't have any exteral inputs");
    CAFFE_ENFORCE(net_.external_input_size() > i, net_.name(),
                  " is missing exteral input ", i);

    return net_.external_input(i);
}

const std::string& NetUtil::Output(int i) {

    CAFFE_ENFORCE(net_.external_output_size() != 0, net_.name(),
                  " doesn't have any exteral outputs");
    CAFFE_ENFORCE(net_.external_output_size() > i, net_.name(),
                  " is missing exteral output ", i);

    return net_.external_output(i);
}

void NetUtil::SetName(const std::string name) { 
    
    net_.set_name(name); 
}

} // caffe2