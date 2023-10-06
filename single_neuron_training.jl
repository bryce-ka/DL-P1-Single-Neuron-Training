#peer revieed by Hala Abu Altayeb
#updated some variables so that they were clearer
#updated accuracy function to use round()


# finds the mean-squared-error loss between predictions and targets
# input: vector of predictions, vector of targets
# output: number


include("single_neuron_model.jl")

function MSE(predictions::AbstractVector{<:Real}, targets::AbstractVector{<:Real})
    #num_preds = length(predictions[1,:])
    num_preds = length(predictions)
    errors = Real[1:num_preds;]
    sum_error = 0
    for pred in [1:num_preds;]
        this_error = predictions[pred] - targets[pred]
        squared_error = this_error^2
        sum_error += squared_error
    end
    MSE = sum_error/num_preds
end

# finds the fraction of points classified correctly if predictions are rounded to 0/1
# only applicable for classification models
# input: vector of predictions, vector of targets
# output: number
function accuracy(predictions::AbstractVector{<:Real}, targets::AbstractVector{<:Real})
    num_preds = length(targets)
    num_errorless = 0
    for pred in [1:num_preds;]
        if round(predictions[pred]) == targets[pred]
            num_errorless+=1
        end
    end
    accuracy = num_errorless/num_preds
end

# finds the gradient of loss w.r.t. model parameters on a data set
# input: model, input array (one data point per row), target vector (length = #points)
# output: a vector of length data_dim + 1, where entries 1:d are weight
# partial derivatives, and entry d+1 is the bias partial derivative

function gradient(model::Neuron, inputs::AbstractMatrix{<:Real},targets::AbstractVector{<:Real})
    len = length(model.weights)
    gradient_vec = zeros(len+1)
    bias_der = 0
    
    for i in [1:length(targets);]
        data_point = inputs[i, :]
        for w in [1:len;]
            bias_der = 0
            pred = predict(model, data_point)
            error = targets[i] - pred
            gradient_vec[w]  += (error * data_point[w]* model.derivative(pred))
            gradient_vec[len+1] += error* model.derivative(pred)
        end
    end
    
    #multiply by -2        
    for i in [1:(len+1);]           
        gradient_vec[i] *= -2
    end
    
    gradient_vec
                
end




# changes the model's weights and bias to take a step in the -gradient direction
# input: gradient is a vector of length #weights + 1, where entries 1:d are weight
#        partial derivatives, and entry d+1 is the bias partial derivative
function update!(model::Neuron, gradient::AbstractVector{<:Real}, step_size::Real)
    len = length(gradient)
    model.bias = model.bias - step_size*gradient[len]
    for i in [1:length(model.weights);] 
        model.weights[i] = model.weights[i] - step_size*gradient[i]
    end    
end

# gradient descent: repeatedly updates the model by small steps in the -gradient direction
# inputs: inputs/targets are a data set to train on
#         losses/gradients/parameters specify vectors for logging itermediate values
#         if these are nothing, we just update the model
#         if they are vectors, then at each iteration, we should push the current values
function train!(model::Neuron, inputs::AbstractMatrix{<:Real},
                targets::AbstractVector{<:Real}, iterations::Integer,
                step_size::Real; losses::Union{Vector,Nothing}=nothing,
                gradients::Union{Vector,Nothing}=nothing,
                parameters::Union{Vector,Nothing}=nothing)

    predictions = predict(model, inputs)
    grad = gradient(model, inputs,targets)
    gradients =[]
    parameters = []
    losses =[]
    
    for i in [1:iterations;]
        predictions = predict(model, inputs)
        grad = gradient(model, inputs,targets)
        weights = model.weights
        
        if gradients isa Vector
            push!(gradients, grad)
        end
        if losses isa Vector
            push!(losses, MSE(predictions, targets))
        end
        if parameters isa Vector
            push!(parameters, weights)
        end
        update!(model, grad,step_size)
    end
    losses,gradients,parameters
end
