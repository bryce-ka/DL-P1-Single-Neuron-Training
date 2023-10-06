import Base: copy

mutable struct Neuron
    weights::Vector{Float64}
    bias::Float64
    activation::Function
    derivative::Function
end

function copy(other::Neuron)
    Neuron(copy(other.weights), other.bias, other.activation, other.derivative)
end