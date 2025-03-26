import torch

class GradientOperator:
    def clamp_grad(grad, min, max):
        return tuple(torch.clamp(g, min, max) for g in grad)

    def normalize_grad(grad):
        n = GradientOperator.grad_norm(grad)
        return tuple(g / (n + 1e-5) for g in grad)

    def grad_norm(grad):
        flattened_tensors = [t.flatten() for t in grad]
        combined = torch.cat(flattened_tensors) 
        return torch.norm(combined)

    def set_grad(parameters, grad):
        # Assign computed gradients to policy parameters
        for param, grad in zip(parameters, grad):
            param.grad = grad  # Manually set gradients
        
    def grad_const_mul(grad, scalar):
        return tuple(scalar * t for t in grad)

    def grad_add(grad1, grad2):
        return tuple(a + b for a, b in zip(grad1, grad2))