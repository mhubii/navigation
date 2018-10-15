#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/blob.h>
#include <stdio.h>
#include <vector>

#include "test_model.h"
#include "blob_util.h"

namespace caffe2 {

void run() {

	NetDef init, predict;

	TestModel test_model(init, predict);
	test_model.Add(10, false);

	std::vector<TIndex> dims({10});
	std::vector<int> data(10, 1);

	TensorCPU input(dims, data, NULL);

	std::printf("Input dim: %d\n", (int)input.dim(0));

	Workspace workspace;
	
	const std::string& input_name = test_model.predict_.Input(0);
	const std::string& output_name = test_model.predict_.Output(0);

	CAFFE_ENFORCE(workspace.RunNetOnce(test_model.init_.net_));
	CAFFE_ENFORCE(workspace.CreateNet(test_model.predict_.net_));

	std::printf("Input name: %s\n", input_name.c_str());
	std::printf("Output name: %s\n", output_name.c_str());
	
	BlobUtil(*workspace.GetBlob(input_name)).Set(input, false);

	std::printf("done\n");
}
} // caffe2

int main(int argc, char **argv) {

	caffe2::GlobalInit(&argc, &argv);
	caffe2::run();
	google::protobuf::ShutdownProtobufLibrary();
	return 0;
}

