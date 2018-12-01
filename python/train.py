import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import StereoCNN
import utils


def train(args):
    """
        Load data, build  and train model.
    """
    # Load, pre-process and augment data.
    data_set = utils.DataSetGenerator(data_dir=args.data_dir,
                                      transform=transforms.Compose([
                                          utils.PreProcessData(),
                                          utils.ToTensor()
                                      ]))

    # Data loader for batch generation.
    data_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=True)

    # Build model.
    model = StereoCNN(utils.INPUT_SHAPE, 3, args.batch_size)

    # Loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model.
    best_loss = float('inf')

    for epoch in range(args.epochs):
        for idx, sample in enumerate(data_loader):
            img_left = Variable(sample['img_left'])
            img_right = Variable(sample['img_right'])
            vel = Variable(sample['vel'])
            optimizer.zero_grad()
            vel_out = model(img_left, img_right)
            loss = criterion(vel_out, vel)
            loss.backward()
            optimizer.step()

            # Save weights.
            if loss.data.item() < best_loss:
                best_loss = loss.data.item()
                torch.save(model.state_dict(), 'trained.pt')

            if idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, idx * len(img_left), len(data_loader.dataset),
                      100. * idx / len(data_loader), loss.data.item()))    


if __name__ == '__main__':

	# Get command line arguments.
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-d', type=str,   help='data directory',   default='data', dest='data_dir')
	parser.add_argument('-l', type=float, help='learning rate',    default=0.001,  dest='learning_rate')
	parser.add_argument('-b', type=int,   help='batch size',       default=40,  dest='batch_size')
	parser.add_argument('-e', type=int,   help='number of epochs', default=10,     dest='epochs')

	args = parser.parse_args()
	
	train(args)

	# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
	trained_model = StereoCNN(utils.INPUT_SHAPE, 3, 1)
	
	trained_model.load_state_dict(torch.load('trained.pt'))

	example = torch.rand(1, utils.IMAGE_CHANNELS, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH)

	traced_script_module = torch.jit.trace(trained_model, (example, example))
	traced_script_module.save('trained_script_module.pt')
