import torch
import torch.nn.functional as F
from typing import TypeVar

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)
    
block_shape = (128, 128)

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for Blockwise W8A8 Matmul scaled to FP32.
    
    Returns:
        Tuple of (
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k

    # Generate random inputs with FP8 quantization
    a = (torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)/10).to(torch.float8_e4m3fnuz)
    b = (torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=gen)/10).to(torch.float8_e4m3fnuz)

    # Generate scaling factors with FP32
    a_scale = torch.randn([m, scale_k], dtype=torch.float32, device="cuda", generator=gen)
    b_scale = torch.randn([scale_n, scale_k], dtype=torch.float32, device="cuda", generator=gen)


    c = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    return (a, b, a_scale, b_scale, c)


def ref_kernel(data: input_t) -> output_t:
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_shape_n = 128
    block_shape_k = 128
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    # Apply scaling to input 'a'
    a_scale = a_scale.unsqueeze(-1).repeat(1, 1, block_shape_k)  # Shape: [m, scale_k, block_shape_k]
    a_scale = a_scale.reshape(m, scale_k * block_shape_k) 
    a_scale = a_scale[:, :k]
    a = a.to(a_scale.dtype) * a_scale 

    # Apply scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]
    b = b.to(b_scale.dtype) * b_scale 

    c = F.linear(a, b, bias=None).to(torch.bfloat16)
    c = (a @ b.T).to(torch.bfloat16)
    return c

def main():
    data = generate_input(64, 1536, 7168, 42)
    o = ref_kernel(data)
    print(o)
    
if __name__ == '__main__':
    main()