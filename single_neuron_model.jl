#peer revieed by Hala Abu Altayeb
#Updated some functions to use less code lines but still same implemetation
#updated some variables so that they were clearer
#updated our predict function to use a dot product instead of looping through weights

using Distributions: Normal
using LinearAlgebra

include("neuron.jl") # Revise.includet doesn't play nice with structs, so Neuron is in its own file.

# y=x activation for a linear neuron
function linear_activation(input)
    input
end

# derivative of a linear neuron's activation function
function linear_derivative(activation)
    1
end

# σ(x) activation for a sigmoid neuron
function sigmoid_activation(input)
    1/(1+exp(-input))
end

# derivative of a sigmoid neuron's activation function
function sigmoid_derivative(activation)
    activation* (1 - activation)
end

# create a neuron with linear activation and random initial weights
function LinearNeuron(input_dimension::Integer)
    weights = rand(Normal(0,1), input_dimension)
    bias = rand(Normal(0,1))
    Neuron(weights, bias, linear_activation, linear_derivative)
end

# create a neuron with sigmoid activation and random initial weights
function SigmoidNeuron(input_dimension::Integer)
    weights = rand(Normal(0,1), input_dimension)
    bias = rand(Normal(0,1))
    Neuron(weights, bias, sigmoid_activation, sigmoid_derivative)
end

# finds the model's output on a single data point
# input: neuron, point (represented as a vector)
# output: number
function predict(model::Neuron, data_point::AbstractVector{<:Real})
    pred_val = dot(data_point,model.weights) + model.bias #finds the sum of weighted inputs + bias
    model.activation(pred_val) #calls the model's activation     
end

# finds the model's output on a collection of data points
# input: neuron, data_set (array where each row represents a point)
# output: vector of predictions for each point
function predict(model::Neuron, data_set::AbstractMatrix{<:Real})
    result = Vector{Real}() #initialize a vector
    rows = size(data_set)[1]
    for i in range(1,rows)
        data_point= data_set[i, :]
        push!(result,predict(model,data_point))
    end 
    result
end
