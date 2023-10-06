using Plots

# plots a ONE DIMENSIONAL regression data set and model
# inputs: model should be a Neuron with linear activation
#         inputs/targets specify a data set
#         resolution is optional and specifies how many
#         points the line should be plotted from
# outputs: a scatter-plot of the data set and a line representing model predictions

function plot_regressor(model::Neuron, inputs::AbstractMatrix{<:Real}, targets::AbstractVector{<:Real}; resolution::Integer=100)
    @assert length(model.weights) == 1
    lb = minimum(inputs)
    ub = maximum(inputs)
    x_range = collect(lb:(ub - lb)/resolution:ub)
    x_range = reshape(x_range, length(x_range), 1)
    plot(x_range, predict(model, x_range), leg=false)
    scatter!(inputs, targets)
end

# plots a TWO DIMENSIONAL classification data set and model
# inputs: model should be a Neuron with sigmoid activation
#         inputs/targets specify a data set
#         resolution is optional and specifies how many points
#         (along each dimension) the heatmap is generated from.
# outputs: a heatmap of model predictions, overlayed with a scatter-plot of the data set
function plot_classifier(model::Neuron, inputs::AbstractMatrix{<:Real}, targets::AbstractVector{<:Real}; resolution::Integer=100)
    @assert length(model.weights) == 2
    lb = minimum(inputs)
    ub = maximum(inputs)
    x_range = lb:(ub - lb)/resolution:ub
    y_range = lb:(ub - lb)/resolution:ub
    data = zeros(length(x_range),length(y_range))
    for (r,x) in enumerate(x_range)
        for (c,y) in enumerate(y_range)
            data[r,c] = predict(model,[y,x]) # why???
        end
    end
    heatmap(x_range, y_range, data, c=cgrad([:blue, :white, :red]), clim=(0,1))
    scatter!(inputs[targets .== 0, 1], inputs[targets .== 0, 2], color=:blue, label="")
    scatter!(inputs[targets .== 1, 1], inputs[targets .== 1, 2], color=:red, label="")
end