import torch
import torch.nn as nn


class StereoCNN(nn.Module):
    """
        Stereo convolutional neural net.
    """

    def __init__(self, input_shape, dof, batch_size):
        """
            Initialize the CNN.
        """

        super(StereoCNN, self).__init__()

        self.batch_size = batch_size
        self.left_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.Relu(),
            nn.Conv2d(8, 16, 5, 2),
            nn.Relu(),
            nn.Conv2d(16, 32, 3, 2),
            nn.Relu(),
            nn.Conv2d(32, 64, 3, 2),
            nn.Relu(),
            nn.Conv2d(64, 64, 3, 2),
            nn.Relu(),
        )

	n_left = self._get_conv_output(self.left_cnn, input_shape)

        self.left_classification = nn.Sequential(
            nn.Linear(n_left, 32),
            nn.Relu(),
            nn.Linear(32, 16),
            nn.Relu(),
            nn.Linear(16, 8),
            nn.Relu(),
            nn.Linear(8, dof)
        )

	self.right_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.Relu(),
            nn.Conv2d(8, 16, 5, 2),
            nn.Relu(),
            nn.Conv2d(16, 32, 3, 2),
            nn.Relu(),
            nn.Conv2d(32, 64, 3, 2),
            nn.Relu(),
            nn.Conv2d(64, 64, 3, 2),
            nn.Relu(),
        )

        n_right = self._get_conv_output(self.right_cnn, input_shape)

        self.right_classification = nn.Sequential(
            nn.Linear(n_right, 32),
            nn.Relu(),
            nn.Linear(32, 16),
            nn.Relu(),
            nn.Linear(16, 8),
            nn.Relu(),
            nn.Linear(8, dof)
        )

    def _get_conv_output(self, net, shape):
        """
            Determine the dimension of the feature space.
        """

        # Unsqueeze to obtain 1x(shape) as dimensions.
        input = torch.rand(shape).unsqueeze(0)
        input = torch.autograd.Variable(input)
        output = net(input)
        n = output.numel()
        return n

    def forward(self, left_in, right_in):
        """
            Forward through the left and the right CNN.
        """

        # Convolutional layers for feature extraction.
	left_out = self.left_cnn(left_in)
	right_out = self.right_cnn(right_in)

        # Flatten.
        left_out = left_out.view(self.batch_size, int(left_out.numel()/self.batch_size))
        right_out = right_out.view(self.batch_size, int(right_out.numel()/self.batch_size))

        # Linear layers for classification.
        left_out = self.left_classification(left_out)
	right_out = self.right_classification(right_out)

	# Addition for comprehensive decission.
	output = left_out - right_out

	return output
