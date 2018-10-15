#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/workspace.h>
//#include <caffe2/core/blob.h>
//#include <caffe2/core/context.h>
//#include <caffe2/core/common.h>
#include <vector>
//#include <stdio.h>
/*
namespace caffe2 {

OperatorDef* AddOp(NetDef& net,
		   const std::string& name,
		   const std::vector<std::string>& inputs,
		   const std::vector<std::string>& outputs) {

	OperatorDef* op = net.add_op();
	op->set_type(name);

	for (auto input : inputs) {
		
		op->add_input(input);
	}

	for (auto output : outputs) {
		
		op->add_output(output);
	}

	return op;
};

Argument* net_add_arg(OperatorDef& op, 
		      const std::string& name) {
  
	Argument* arg = op.add_arg();
	arg->set_name(name);

	return arg;
};

Argument* net_add_arg(OperatorDef& op, const std::string& name, int value) {

	Argument* arg = net_add_arg(op, name);
	arg->set_i(value);

	return arg;
}


Argument* net_add_arg(OperatorDef& op, 
		      const std::string& name,
                      std::vector<int> values) {
    
	auto arg = net_add_arg(op, name);

	for (auto value : values) {
		
		arg->add_ints(value);
	}

	return arg;
};

OperatorDef* AddXavierFillOp(NetDef& net,
			     const std::vector<int>& shape,
			     const std::string& param) {
    
	OperatorDef* op = AddOp(net, "XavierFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
};

OperatorDef* AddConstantFillOp(NetDef& net,
			       const std::vector<int>& shape,
			       const std::string& param) {

	auto op = AddOp(net, "ConstantFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
}

OperatorDef* AddFcOp(NetDef& net,
		     const std::string& input, const std::string& w,
		     const std::string& b, const std::string& output,
		     int axis = 1) {

	OperatorDef* op = AddOp(net, "FC", {input, w, b}, {output});

	if (axis != 1) {

		net_add_arg(*op, "axis", axis);
	}

	return op;
}

void AddInput(NetDef& net, 
	      const std::string input) {

	net.add_external_input(input);
};

void AddOutput(NetDef& net, 
	       const std::string output) {

	net.add_external_output(output);
};

void run() {

	// DeviceOption option;
	// CPUContext* cpu_context;

	// Tensor.
	vector<int64_t> dims({10});
	vector<float> values(10, 1);

	Tensor tensor1(dims, CPU);
	tensor1.ShareExternalPointer(values.data());

	switch (tensor1.GetDeviceType()) {

		case CPU:
	
			printf("CPU\n");
			printf("Tensor dim: %i\n", (int)tensor1.dim(0));
			printf("1st element: %f\n", (float)tensor1.data<float>()[0]);
			break;

		case CUDA:
		
			printf("GPU\n");
			printf("Tensor dim: %i\n", (int)tensor1.dim(0));
			printf("1st element: %f\n", (float)tensor1.data<float>()[0]);
			break;	

		default:

			printf("No device\n”");
			break;
	}

	// Workspace.
	Workspace workspace;

	Tensor* data = workspace.CreateBlob("data")->GetMutable<Tensor>();
	data->ResizeLike(tensor1);
	//data->ShareData(tensor1);
	


	// Net.
	NetDef init, predict;

	predict.set_name("TestModel");

	std::string input = "data";
	std::string layer = input;
	
	// Input layer.
	AddInput(predict, layer);

	// Fully connected layer.
	int out_size = 1;
	int in_size = tensor1.dim(0);

	std::string output = "fc";

	AddXavierFillOp(init, {out_size, in_size}, output + "_w");
        AddInput(predict, output + "_w");
        AddConstantFillOp(init, {out_size}, output + "_b");
        AddInput(predict, output + "_b");
        
        AddFcOp(predict, input, output + "_w", output + "_b", output);

	// Output layer.
	AddOutput(predict, layer);

	// Initialize nets.
	CAFFE_ENFORCE(workspace.RunNetOnce(init));
	CAFFE_ENFORCE(workspace.CreateNet(predict));

	// Run net.
	//Tensor* data = workspace.GetBlob("data")->GetMutable<Tensor>();
	//data->ShareData(tensor1);
	//printf("Data shared.\n");

	// Forward.
	//workspace.RunNet(predict.name());
	//printf("Net was run.\n");


}
}
*/

using namespace caffe2;
using namespace std;

int main(int argc, char** argv) {

	caffe2::GlobalInit(&argc, &argv);
	//caffe2::run();

	// Tensor.
	vector<float> values(10, 1);

	Tensor tensor(10, CPU);
	tensor.ShareExternalPointer(values.data());

	// Workspace.
	Workspace workspace;

	workspace.CreateBlob("data");
	//Tensor* test = workspace.GetBlob("data")->GetMutable<Tensor>();
	Tensor* data = BlobGetMutableTensor(workspace.GetBlob("data"), CPU);
	data->ResizeLike(tensor);
	data->ShareData(tensor);

	//printf("size: %i", test->nbytes());

	google::protobuf::ShutdownProtobufLibrary();
	
	return 0;
}
