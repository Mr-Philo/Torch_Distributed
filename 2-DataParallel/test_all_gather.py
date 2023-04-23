import torch
import torch.distributed as dist
import argparse
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
    utils.init_distributed_mode(args, disable_print=False)
    
    rank = utils.get_rank()
    torch.manual_seed(rank + 123)
    t = torch.randint(0, 10, (2,3)).to("cuda")
    print(f"Rank {rank} has tensor {t}")
    print("---------------------------------------------------")
    
    tensor_list = [torch.ones_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, t)
    print(f"Rank {rank} has gathered tensor {tensor_list}")
    
    
if __name__ == "__main__":
    main()
