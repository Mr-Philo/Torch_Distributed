import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
import time
import utils


def get_args_parser():
    parser = argparse.ArgumentParser('Distributed training')
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--sync-bn", action="store_true")
    
    return parser.parse_args()


def main():
    args = get_args_parser()
    utils.init_distributed_mode(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = utils.ConvNet().to(device)
    #! ---------------------------------------------------------------
    if args.distributed:
        if args.sync_bn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], output_device=args.gpu)
    #! ----------------------------------------------------------------

    trainset, testset = utils.get_dataset()
    #! ---------------------------------------------------------------
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, sampler=train_sampler)
    #! ----------------------------------------------------------------
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):
        #! ---------------------------------------------------------------
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #! ----------------------------------------------------------------
        t0 = time.time()
        
        loss_sum,acc_sum = 0,0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criteria(outputs, labels)
            
            loss_sum += loss.item()
            predict = torch.argmax(outputs, dim=1)
            acc_sum += torch.sum(predict == labels).item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Epoch: {}, Loss: {:.2f}, acc: {:.2f}, time cost: {:.2f}s".format(epoch, loss_sum/len(train_loader), acc_sum/len(trainset), time.time()-t0))
        
        
if __name__ == "__main__":
    main()
