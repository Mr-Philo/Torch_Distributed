import os
import torch
import megatron.core.parallel_state as ps


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
        

def print_in_rank_zero(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)
        
       
from megatron.core.tensor_parallel.data import broadcast_data
  
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
    
    print_in_rank_zero("Broadcast assertion test passed")
    Utils.destroy_model_parallel()
    

from megatron.core.tensor_parallel import mappings

def test_CopyToModelParallelRegion():
    Utils.initialize_model_parallel(4,2)
    input_data = torch.ones((1)).cuda()*Utils.rank
    output_data = mappings._CopyToModelParallelRegion.backward(None, input_data)
    print(f'rank: {Utils.rank}, input_data: {input_data}, output_data: {output_data}')
    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert(torch.equal(output_data, result))
    assert(torch.equal(input_data, mappings.copy_to_tensor_model_parallel_region(input_data)))
    assert(torch.equal(input_data, mappings._CopyToModelParallelRegion.symbolic(None, input_data)))
    
    
    print_in_rank_zero("Copy to model parallel region test passed")
    Utils.destroy_model_parallel()
    
   
def test_ReduceFromModelParallelRegion():
    Utils.initialize_model_parallel(4,2)
    input_data = torch.ones((1)).cuda()*Utils.rank
    output_data = mappings._ReduceFromModelParallelRegion.symbolic(None, input_data)
    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert(torch.equal(output_data, result))
    input_data = torch.ones((1)).cuda()*Utils.rank
    assert(torch.equal(mappings.reduce_from_tensor_model_parallel_region(input_data), result))
    assert(torch.equal(input_data, mappings._ReduceFromModelParallelRegion.backward(None, input_data)))
    
    print_in_rank_zero("Reduce from model parallel region test passed")
    Utils.destroy_model_parallel()
    
    
def test_ScatterToModelParallelRegion():
    Utils.initialize_model_parallel(4,2)
    input_data = torch.rand((8,4)).cuda()
    output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)
    req_dim = int(Utils.rank%(Utils.world_size/2))
    assert(torch.equal(output_data, input_data[:,req_dim].reshape((8,1))))
    output_data = mappings._ScatterToModelParallelRegion.symbolic(None, input_data)
    assert(torch.equal(output_data, input_data[:, req_dim].reshape((8,1))))

    input_data = torch.ones(8).cuda() * Utils.rank
    actual_output_data = mappings._ScatterToModelParallelRegion.backward(None, input_data)
    expected_output = torch.cat((
        torch.ones(8)*0,
        torch.ones(8)*1,
        torch.ones(8)*2,
        torch.ones(8)*3)).cuda()
    if (Utils.rank >= 4):
        expected_output = expected_output + 4
    assert(torch.equal(actual_output_data, expected_output))
    
    print_in_rank_zero("Scatter to model parallel region test passed")
    Utils.destroy_model_parallel()
    
    
import megatron.core.tensor_parallel.utils as util

def test_gather_split_1d_tensor():
    rank = Utils.rank
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    input_tensor = torch.ones((2,4)).cuda() * rank
    actual_output_tensor = util.gather_split_1d_tensor(input_tensor)
    if rank %2 == 0:
        expected_output_tensor = torch.concat((input_tensor.flatten(), input_tensor.flatten() + 1))
    else : 
        expected_output_tensor = torch.concat((input_tensor.flatten() - 1, input_tensor.flatten()))
    print(f'rank: {rank}, input_tensor: {input_tensor}; output_tensor: {actual_output_tensor}')    
        
    assert(torch.equal(actual_output_tensor, expected_output_tensor))
    print_in_rank_zero("Gather split 1d tensor test passed")
    Utils.destroy_model_parallel()
    

if __name__ == '__main__':
    
    TEST_BROAD = bool(int(os.getenv('TEST_BROAD', 0)))
    TEST_COPY = bool(int(os.getenv('TEST_COPY', 0)))
    TEST_REDUCE = bool(int(os.getenv('TEST_REDUCE', 0)))
    TEST_SCATTER = bool(int(os.getenv('TEST_SCATTER', 0)))
    TEST_GATHER_SPLIT = bool(int(os.getenv('TEST_GATHER_SPLIT', 0)))
    
    if TEST_BROAD:
        test_broadcast_data()
    if TEST_COPY:
        test_CopyToModelParallelRegion()
    if TEST_REDUCE:
        test_ReduceFromModelParallelRegion()
    if TEST_SCATTER:
        test_ScatterToModelParallelRegion()
    if TEST_GATHER_SPLIT:
        test_gather_split_1d_tensor()
