import os
import torch
import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.data import broadcast_data

class Utils:

    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
        torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend='nccl', world_size=Utils.world_size, rank=Utils.rank, init_method=init_method)
        
    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1, virtual_pipeline_model_parallel_size = None, pipeline_model_parallel_split_rank = None):
        ps.destroy_model_parallel()
        if not torch.distributed.is_initialized():
            Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)
        
def test_broadcast_data():
    Utils.initialize_model_parallel(2,4)
    input_data = {
        0 : torch.ones((8,8)).cuda() * 0.0,
        1 : torch.ones((8,8)).cuda() * 1.0,
        2 : torch.ones((8,8)).cuda() * 2.0,
        3 : torch.ones((8,8)).cuda() * 3.0,
        4 : torch.ones((8,8)).cuda() * 4.0,
        5 : torch.ones((8,8)).cuda() * 5.0,
        6 : torch.ones((8,8)).cuda() * 6.0,
        7 : torch.ones((8,8)).cuda() * 7.0
        }
    dtype = torch.float32
    actual_output = broadcast_data([0,1],input_data, dtype)
    assert(torch.equal(actual_output[0], input_data[0]))
    assert(torch.equal(actual_output[1], input_data[1]))
    Utils.destroy_model_parallel()
    
if __name__ == '__main__':
    test_broadcast_data()