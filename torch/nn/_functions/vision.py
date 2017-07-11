import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn


class GridSampler(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("GridSampler needs CuDNN for processing CUDA inputs,"
                               " but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, input, grid):
        ctx.save_for_backward(input, grid)
        if input.is_cuda:
            GridSampler._enforce_cudnn(input)
            output = input.new(input.size())
            grid = grid.contiguous()
            torch._C._cudnn_grid_sampler_forward(input, grid, output)
        else:
            backend = type2backend[type(input)]
            output = input.new(input.size())
            backend.SpatialGridSamplerBilinear_updateOutput(backend.library_state, input, grid, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grid = grid.contiguous()
        if input.is_cuda:
            GridSampler._enforce_cudnn(input)
            grad_input = input.new(input.size())
            grad_grid = grid.new(grid.size())
            torch._C._cudnn_grid_sampler_backward(input, grad_input,
                                                  grid, grad_grid,
                                                  grad_output)
        else:
            backend = type2backend[type(input)]
            grad_input = input.new(input.size())
            grad_grid = grid.new(grid.size())
            backend.SpatialGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output)
        return grad_input, grad_grid
