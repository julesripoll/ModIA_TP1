import tensorflow as tf  
import tensorboard as tb  
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import argparse # to parse script arguments
from statistics import mean # to compute the mean of a list
from tqdm import tqdm #used to generate progress bar during training

import torch
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter
from  torchvision.utils import make_grid #to generate image grids, will be used in tensorboard 
from torch import nn

from data_utils import get_colorized_dataset_loader # dataloarder
from unet import UNet

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, epochs=5, writer=None):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t: # x: black and white image, y: colored image 
            x,y=x.to(device),y.to(device)
            y_hat=net(x)
            loss=criterion(y_hat,y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        if writer is not None:
            #Logging loss in tensorboard
            writer.add_scalar('training loss', mean(running_loss), epoch)
            # Logging a sample of inputs in tensorboard
            input_grid = make_grid(x[:16].detach().cpu())
            writer.add_image('Input', input_grid, epoch)
            # Logging a sample of predicted outputs in tensorboard
            colorized_grid = make_grid(y_hat[:16].detach().cpu())
            writer.add_image('Predicted', colorized_grid, epoch)
            # Logging a sample of ground truth in tensorboard
            original_grid = make_grid(y[:16].detach().cpu())
            writer.add_image('Ground truth', original_grid, epoch)
    return mean(running_loss)
        


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str,default='/Users/julesripoll/Developer/Cours/Techno IA/ModIA_TP1/data/landscapes', help='data path')
    parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
    parser.add_argument('--epochs', type=int,default=3,help='nombre epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    data_path = args.data_path

    unet = UNet().to(device)
    loader = get_colorized_dataset_loader(path=data_path, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=0)


    optimizer = optim.Adam(unet.parameters(), lr=lr)
    writer = SummaryWriter(f'runs/{exp_name}')
    train(unet, optimizer, loader, epochs=epochs, writer=writer)
    writer.add_graph(unet,x.to(device))

    # Save model weights
    torch.save(unet.state_dict(), 'unet.pth')