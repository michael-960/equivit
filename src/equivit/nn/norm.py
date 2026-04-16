import torch
import torch.nn as nn
from typing import Union, List



# adapted from octic-vit
class ListLayerScale(nn.Module):
    """
    This is the same as :class:`ListAffine` but without the bias term. 

    Args:
        dims: list of integers :math:`C_0, C_1, \dotsb, C_{M-1}`, the dimensions of the tensors in the input list
        init_values: a float or a tensor specifying the initial value of the
        scaling parameters. 
    """
    def __init__(
        self,
        dims: List[int],
        init_values: Union[float, torch.Tensor] = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = nn.ParameterList([init_values*torch.ones((d,1)) for d in dims])
        self.n_tensors = len(dims)

    def forward(self, x):
        """
        Args:
            x: list of tesnors, each of shape :math:`(*, C_i, d_i)`

        Returns:
            list of tensors, each of shape :math:`(*, C_i, d_i)`
        """
        return [self.alpha[i]*x[i] for i in range(self.n_tensors)]


# adapted from octic-vit
class ListAffine(nn.Module):
    r"""
    Given an input list of tensors, each of shape :math:`(*, C_i, d_i)`, 
    this layer applies a learnable channel-wise affine transformation to each tensor in the list independently, i.e. it computes

    .. math::
            (y_i)_{kl} = (\alpha_i)_k (x_i)_{kl} + (\beta_i)_k,

    where :math:`\alpha_i` and :math:`\beta_i` are learnable parameters of shape :math:`(C_i, 1)`. 
    Note that the bias term :math:`\beta_i` is only applied to the trivial representation (:math:`i=0`) if :attr:`bias` is True.

    The parameters :math:`\alpha_i` and :math:`\beta_i` are initialized to 1 and 0 respectively, so that the layer initially performs the identity transformation.

    Args:
        dims: list of integers :math:`C_0, C_1, \dotsb, C_{M-1}`, the dimensions of the tensors in the input list
        bias: whether to use bias **for the zeroth tensor** in the affine transformation
    """
    def __init__(self, 
        dims: List[int], bias: bool=True
    ):
        super().__init__()
        self.alpha = nn.ParameterList([torch.ones((d,1)) for d in dims])

        if bias:
            self.beta = nn.Parameter(torch.zeros(dims[0], 1))
        else:
            self.register_parameter('beta', None)

        self.n_tensors = len(dims)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        r"""
        Args:
            x: list of tesnors, each of shape :math:`(*, C_i, d_i)`
        Returns:
            list of tensors, each of shape :math:`(*, C_i, d_i)`
        """
        y = [self.alpha[i]*x[i] for i in range(self.n_tensors)]

        if self.beta is not None:
            y[0] = y[0] + self.beta
        
        return y


# adapted from octic-vit
class EquivariantLayerNorm(nn.Module):
    r"""
    We implement an irrep-wise layer norm, which is more precisely a group normalization. 

    The input is a list of tensors :math:`x_0, x_1, \dotsb, x_{M-1}`,
    where :math:`x_i` has shape :math:`(*, C_i, d_i)`.


    This layer does the following to each :math:`x_i` independently:

    For notational simplicity assume :math:`x_i` has shape :math:`(C_i, d_i)`.
    We will index it as :math:`(x_i)_{kl}`, where :math:`k`
    indexes the channels and :math:`l` indexes the dimensions of
    the irrep. 
    We compute the means and standard deviation as follows:
    
    .. math::
        \begin{aligned}
        &(m_i)_l = \frac{1}{C_i} \sum_{k=0}^{C_i-1} (x_i)_{kl} \\
        &\sigma_i = \sqrt{\frac{1}{d_i}\sum_{l=0}^{d_i-1}\frac{1}{C_i} \sum_{k=0}^{C_i-1} ((x_i)_{kl} - (m_i)_l)^2 + \epsilon}
        \end{aligned}

    Then we normalize :math:`x_i` as follows:

    .. math::
        (y_i)_{kl} = ((x_i)_{kl} - (m_i)_l) / \sigma_i.

    If :attr:`elementwise_affine` is True, we apply a learnable channel-wise affine transformation to the output, i.e. we compute

    .. math::
        (z_i)_{kl} = (\alpha_i)_k (y_i)_{kl} + (\beta_i)_k,

    where :math:`\alpha_i` and :math:`\beta_i` are learnable parameters of shape
    :math:`(C_i, 1)`. Note that the bias term :math:`\beta_i` is only applied to
    the trivial representation (:math:`i=0`) if :attr:`bias` is True. 
    See :class:`ListAffine` for more details on the affine transformation.

    Args:
        dims: list of integers :math:`C_0, C_1, \dotsb, C_{M-1}`, the dimensions of each tensor
        eps: small value to avoid division by zero
        elementwise_affine: whether to apply a layer of :class:`ListAffine` to the output
        bias: whether to use bias in the affine transformation
    """
    def __init__(self, 
        dims: List[int], 
        eps=1e-05, 
        elementwise_affine=True, 
        bias=True
    ):
        super().__init__()

        if elementwise_affine:
            self.scaling = ListAffine(dims, bias=bias) 
        else: 
           self.scaling = nn.Identity()
        self.eps = eps

        self.dims = dims

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of tensors, each of shape :math:`(*, C_i, d_i)`

        Returns:
            list of tensors, each of shape :math:`(*, C_i, d_i)`
        """
        # Note that the following two are the same for a complex64 tensor z:
        # (1) z.var(dim=-2, correction=0, keepdim=True).sum(dim=-1, keepdim=True)
        # (2) z.view(torch.float32).var(dim=-2, correction=0, keepdim=True).sum(dim=-1, keepdim=True)
        stds = [
            torch.sqrt(z.var(dim=-2, correction=0, keepdim=True).sum(dim=-1, keepdim=True) + self.eps)
            if self.dims[i] > 0 else 1.
            for i, z in enumerate(x)
        ]        
        y = [(z - z.mean(dim=-2, keepdim=True)) / stds[i] for i, z in enumerate(x)]

        return self.scaling(y)

