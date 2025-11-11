import torch
from typing import Optional
import pdb

class DiagonalGaussianDistributionFromGetMesh(object):
    def __init__(self, parameters, deterministic=False, max_logvar=20, min_logvar=-30):
        self.parameters = parameters
        # parameters is of shape B, 2*C, N
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar
        # this clamp avoid to have extreme values for std and var 
        logvar_clamp = torch.clamp(self.logvar, self.min_logvar, self.max_logvar)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * logvar_clamp)
        self.var = torch.exp(logvar_clamp)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None, sequence_length=None, add_extreme_var_loss=False, simple_loss=False):
        # sequence_length is of shape B
        if self.deterministic:
            return torch.Tensor([0.])
        elif simple_loss:
            sum_dims = list(range(1,len(self.mean.shape)))
            return self.mean.abs().sum(dim=sum_dims) + self.logvar.abs().sum(dim=sum_dims)
        else:
            sum_dims = list(range(1,len(self.mean.shape)))
            if sequence_length is None:
                mask = 1
            else:
                N = self.mean.shape[2]
                device = self.mean.device
                mask = (torch.arange(N).unsqueeze(0).to(device) < sequence_length.unsqueeze(1).to(device)).float() # B,N
                mask = mask.unsqueeze(1) # B,1,N
            if add_extreme_var_loss:
                extreme_var_mask = ((self.logvar< (self.min_logvar+0.05*abs(self.min_logvar)) ).float() * 
                    (self.logvar> (self.max_logvar-0.05*abs(self.max_logvar)) ).float() * mask)
                var_loss = torch.sum((self.logvar * extreme_var_mask)**2, dim=sum_dims)
            else:
                var_loss = 0
            if other is None:
                kl_div = 0.5 * torch.sum((torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)*mask,
                                       dim=sum_dims)
                return kl_div + var_loss
            else:
                kl_div = 0.5 * torch.sum((torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar)*mask, dim=sum_dims)
                return kl_div + var_loss

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        # parameters is of shape B,N,2C
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1) # B,N,C
        # we assume the last dimension is the channel dimension
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, correlate_over_seq_len=False) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        # sample = randn_tensor(
        #     self.mean.shape,
        #     generator=generator,
        #     device=self.parameters.device,
        #     dtype=self.parameters.dtype,
        # )
        # x = self.mean + self.std * sample
        if correlate_over_seq_len:
            B,N,C = self.mean.shape
            noise =torch.randn(B,1,C, dtype=self.mean.dtype, device=self.mean.device)
            # in this case, different tokens in the same sequence are added with the same noise
            # this is used when different tokens in the same sequence are correlated
            x = self.mean + self.std * noise
        else:
            x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other = None, valid_token_idx = None, return_mean = True) -> torch.Tensor:
        # valid_token_idx could be a tensor of shape B,N that contain 1 and 0
        # it indicates whether the tokens are valid ones or padded ones
        # we only compute kl loss for valid tokens
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            B,N,C = self.mean.shape
            dimension_count = 1
            if valid_token_idx is None:
                valid_token_idx = 1
                if return_mean:
                    dimension_count = N*C*torch.ones(B, dtype=self.mean.dtype, device=self.mean.device) # B
            else:
                valid_token_idx = valid_token_idx.unsqueeze(-1).float() # B,N,1
                if return_mean:
                    dimension_count = (valid_token_idx * C).sum(dim=[1,2]) # B
                

            sum_dims = list(range(1,len(self.mean.shape)))
            # sum all dimensions in mean except for the first dimension
            if other is None:
                return 0.5 / dimension_count * torch.sum(
                    (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)*valid_token_idx,
                    dim=sum_dims,
                )
            else:
                return 0.5 / dimension_count * torch.sum(
                    (torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar)*valid_token_idx,
                    dim=sum_dims,
                )

    # def nll(self, sample: torch.Tensor, dims: tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
    #     if self.deterministic:
    #         return torch.Tensor([0.0])
    #     logtwopi = np.log(2.0 * np.pi)
    #     return 0.5 * torch.sum(
    #         logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
    #         dim=dims,
    #     )

    def mode(self) -> torch.Tensor:
        return self.mean

if __name__ == '__main__':
    B = 4
    C = 1024
    N = 16
    # H = 32
    # W = 32
    # shape = (B,C,H,W)
    shape = (B,N,C)
    parameters = (torch.rand(*shape)-0.5) * 2 * 3
    # parameters[:,512:,:] = -1e5
    parameters.requires_grad = True
    posterior = DiagonalGaussianDistribution(parameters)
    valid_token_idx = torch.randint(0,2,size=(B,N))
    loss = posterior.kl(valid_token_idx=valid_token_idx)
    print(loss)
    loss.mean().backward()
    sample = posterior.sample(correlate_over_seq_len=True)
    pdb.set_trace()