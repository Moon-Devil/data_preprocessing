import torch
import numpy as np
import torch.autograd


class Restricted_Boltzmann_Machine:
    def __init__(self, input_dimension, reduction_dimension):
        super(Restricted_Boltzmann_Machine, self).__init__()
        self.weights = None
        self.h_bias = None
        self.v_bias = None
        self.persistent = None
        self.n_visible = input_dimension
        self.n_hidden = reduction_dimension

    def build(self, weights=None, h_bias=None, v_bias=None, persistent = None) -> object:
        if weights is None:
            persistent = None
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visible + self.n_hidden))
            weights = np.random.uniform(-bounds, bounds, (self.n_visible, self.n_hidden))
        if h_bias is None:
            h_bias = np.random.rand(self.n_hidden)
        if v_bias is None:
            v_bias = np.random.rand(self.n_visible)

        return weights, h_bias, v_bias, persistent

    def energy_function(self, v_sample) -> object:
        h_prediction = torch.matmul(v_sample, self.weights) + self.h_bias
        v_bias_term = torch.matmul(v_sample, torch.unsqueeze(self.v_bias, dim=1))
        v_bias_term = v_bias_term.squeeze(dim=1)
        hidden_term = torch.mean(torch.log(1.0 + torch.exp(h_prediction)), dim=1)
        result = -hidden_term - v_bias_term

        return result

    def gibbs_hvh(self, h_sample) -> object:
        v1_value = torch.sigmoid(torch.matmul(h_sample, torch.t(self.weights)) + self.v_bias)
        v1_simple_ReLU = torch.nn.ReLU(inplace=True)
        v1_sample = v1_simple_ReLU(torch.sign(v1_value - torch.rand(v1_value.shape)))

        h1_value = torch.sigmoid(torch.matmul(v1_sample, self.weights) + self.h_bias)
        h1_simple_ReLU = torch.nn.ReLU(inplace=True)
        h1_sample = h1_simple_ReLU(torch.sign(h1_value - torch.rand(h1_value.shape)))

        return v1_sample, h1_sample

    def cd_k_function(self, input_values, k):
        learning_rate = 0.01
        input_values.to("cuda:0")
        h1_value = torch.sigmoid(torch.mm(input_values, self.weights, out=None) + self.h_bias)
        h1_sample_ReLU = torch.nn.ReLU(inplace=True)
        h1_sample = h1_sample_ReLU(torch.sign(h1_value - torch.rand(h1_value.shape)))

        if self.persistent is None:
            chain_start = h1_sample
        else:
            chain_start = self.persistent

        hk_sample = 0
        vk_sample = 0
        for _ in np.arange(k):
            vk_sample, hk_sample = self.gibbs_hvh(chain_start)
            chain_start = hk_sample

        chain_end = vk_sample.detach()

        cost = self.energy_function(input_values) - self.energy_function(chain_end)
        cost.data.zero_()
        cost.backward(torch.ones_like(cost))
        g_weights = self.weights.grad
        g_h_bias = self.h_bias.grad
        g_v_bias = self.v_bias.grad

        weight = self.weights - learning_rate * g_weights
        h_bias = self.h_bias - learning_rate * g_h_bias
        v_bias = self.v_bias - learning_rate * g_v_bias

        self.weights = weight.clone().detach().requires_grad_(True)
        self.h_bias = h_bias.clone().detach().requires_grad_(True)
        self.v_bias = v_bias.clone().detach().requires_grad_(True)
        if self.persistent is not None:
            self.persistent = hk_sample
        else:
            self.persistent = None

    def fit(self, input_values, epochs, k):
        input_values = torch.tensor(input_values, dtype=torch.float64)
        weight, h_bias, v_bias, persistent = self.build()
        self.weights = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
        self.h_bias = torch.tensor(h_bias, dtype=torch.float64, requires_grad=True)
        self.v_bias = torch.tensor(v_bias, dtype=torch.float64, requires_grad=True)
        if persistent is not None:
            self.persistent = torch.tensor(persistent, dtype=torch.float64, requires_grad=False)
        else:
            self.persistent = None

        self.weights.to("cuda:0")
        self.h_bias.to("cuda:0")
        self.v_bias.to("cuda:0")
        input_values.to("cuda:0")

        for epoch in range(epochs):
            self.cd_k_function(input_values, k)

    def data_reduction_function(self, input_values) -> object:
        input_values = torch.tensor(input_values, dtype=torch.float64)
        activation_h = torch.sigmoid(torch.matmul(input_values, self.weights) + self.h_bias)
        activation_h = activation_h.detach().numpy()

        return activation_h

