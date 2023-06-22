import argparse
import os
import numpy as np
from model import StaticSINRModel
from data_loader import StaticSinrDataset 
import torch
from rich.pretty import pprint
import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torchinfo import summary

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", help="training data path", required=True, type=str)
    parser.add_argument("--val_folder", help="validation data path", required=True, type=str)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    # parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--load_from_main_checkpoint", type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--checkpoint_dir", help="checkpoint dir", required=True, type=str)
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    parser.add_argument("--lr_gamma", help='learning rate decay rate per step', required=True, type=float)
    parser.add_argument("--cuda_device", help="cude device index", required=True, type=str)
    args, unknown = parser.parse_known_args()
    return args

args = set_parser()

def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)

def val(model, val_loader, device, criterion, epoch, batch_size):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            x = data['x'].to(device).float()
            y = data['y'].to(device).float()
            yhat = model(x)
            loss = criterion(yhat, y.data)
            val_loss = val_loss + loss

    print('\nVal: epoch:', epoch, ' Loss: ', val_loss / len(val_loader))

    return val_loss

def train(model, train_loader, device, optimizer, criterion, epoch):
    model.to(device)
    model.train()

    for batch_idx, data in enumerate((train_loader)):
        optimizer.zero_grad()
        xs = data['x'].to(device).float()
        ys = data['y'].to(device).float()

        yhats = model(xs)

        loss = criterion(yhats, ys.data)

        loss.backward()

        optimizer.step()

def split_data(train_folder, val_folder):
    train_dataset = StaticSinrDataset(train_folder)
    val_dataset = StaticSinrDataset(val_folder)
    train_dataloader  = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    num_workers=5)
    val_dataloader  = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=None,
                                                    batch_sampler=None)
    return train_dataloader, val_dataloader, train_dataset, val_dataset 

def main():
    time_stamp = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    args.checkpoint_dir = args.checkpoint_dir + time_stamp
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    device = torch.device('cuda:' + args.cuda_device)

    print('=' * 50 + "loading data and instantiating data loader" + '=' * 50)
    train_dataloader, val_dataloader, train_dataset, val_dataset = split_data(args.train_folder, args.val_folder)
    print('=' * 50 + "ending loading data and instantiating data loader" + '=' * 50)

    model = StaticSINRModel()

    with open('./model_arch.txt', 'w+') as f:
        f.write(str(summary(model, input_size=(10, 5), device='cpu')))

    if args.load_from_main_checkpoint:
        print('=' * 50 + "load check point" + '=' * 50)
        chkpt_mainmodel_path = args.load_from_main_checkpoint
        print("Loading check point", chkpt_mainmodel_path)
        model.load_state_dict(torch.load(chkpt_mainmodel_path, map_location=device))
        print('=' * 50 + "end loading check point" + '=' * 50)

    print('=' * 50 + "setting loss criterion, optimizer, lr decay" + '=' * 50)
    criterion = torch.nn.MSELoss()
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.epochs/10.0, gamma=0.5)
    print('=' * 50 + "ending setting loss criterion, optimizer, lr decay" + '=' * 50)

    if args.eval_only:
        print("evaluate only")
        loss = val(model, val_dataloader, device, criterion, 0, args.batch_size) 
        print('loss = ' + str(loss))
        return True

    best_loss = np.inf
    print('=' * 50 + "train and validating" + '=' * 50)
    for epoch in tqdm.tqdm(range(args.epochs)):
        train(model, train_dataloader, device, optimizer_ft, criterion, epoch)
        exp_lr_scheduler.step()

        loss = val(model, val_dataloader, device, criterion, epoch, args.batch_size) 

        if (loss < best_loss) or (epoch % 50 == 0):
            save_model(checkpoint_dir=args.checkpoint_dir,
                       model_checkpoint_name='epoch_' + str(epoch) + '_' + str(loss.item()),
                       model=model)
        best_loss = min(best_loss, loss) 

    print('=' * 50 + "ending train" + '=' * 50)


if __name__ == '__main__':
    main()
