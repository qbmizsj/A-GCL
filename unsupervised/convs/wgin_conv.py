from typing import Callable, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch.nn.functional as F

from unsupervised.convs.inits import reset


class WGINConv(MessagePassing):
	def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
				 **kwargs):
		kwargs.setdefault('aggr', 'add')
		super(WGINConv, self).__init__(**kwargs)
		#self.nn = nn
		self.lin = nn
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))
		self.reset_parameters()

	def reset_parameters(self):
		#reset(self.nn)
		reset(self.lin)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight = None,
				size: Size = None) -> Tensor:
		""""""
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)

		# propagate_type: (x: OptPairTensor)
		out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

		x_r = x[1]
		if x_r is not None:
			out += (1 + self.eps) * x_r

		#return self.nn(out)
		return self.lin(out)

	def message(self, x_j: Tensor, edge_weight) -> Tensor:
		return F.relu(x_j) if edge_weight is None else  F.relu(x_j)  * edge_weight.view(-1, 1)


	def __repr__(self):
		#return '{}(nn={})'.format(self.__class__.__name__, self.nn)
		return '{}(nn={})'.format(self.__class__.__name__, self.lin)