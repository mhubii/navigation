import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import CNN
import utils


def train():
    """
        Load data, build  and train model.
    """

    # Get command line arguments.
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', type=str,   help='data directory',   default='data', dest='data_dir')
    parser.add_argument('-l', type=float, help='learning rate',    default=0.001,  dest='learning_rate')
    parser.add_argument('-b', type=int,   help='batch size',       default=40,  dest='batch_size')
    parser.add_argument('-e', type=int,   help='number of epochs', default=10,     dest='epochs')

    args = parser.parse_args()

    # Load, pre-process and augment data.
    data_set = utils.DataSetGenerator(data_dir=args.data_dir,
                                      transform=transforms.Compose([
                                          utils.AugmentData(),
                                          utils.PreProcessData(),
                                          utils.ToTensor()
                                      ]))

    # Data loader for batch generation.
    data_loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=True)

    # Build model.
    model = CNN(utils.INPUT_SHAPE, args.batch_size)

    # Loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model.
    best_loss = float('inf')

    for epoch in range(args.epochs):
        for idx, sample in enumerate(data_loader):
            img = Variable(sample['img'])
            steering_angle = Variable(sample['steering_angle'])
            optimizer.zero_grad()
            steering_angle_out = model(img)
            loss = criterion(steering_angle_out, steering_angle)
            loss.backward()
            optimizer.step()

            # Save weights.
            if loss.data[0] < best_loss:
                best_loss = loss.data[0]
                torch.save(model.state_dict(), 'train.pth')

            if idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(img), len(data_loader.dataset),
                    100. * idx / len(data_loader), loss.data[0]))


if __name__ == '__main__':
	
	train()
