using Distributions: Uniform, MvNormal
using Random: shuffle!
using LinearAlgebra: I

# generates a random data set with a linear trend plus noise
# returns two arrays: (x,y)
function generate_regression_data(num_points, input_dimension; lb=-1, ub=1, var=.1)
    coefficients = rand(Uniform(lb/input_dimension, ub/input_dimension), input_dimension)
    constant = rand(Uniform(lb,ub))
    inputs = rand(Uniform(lb,ub), num_points, input_dimension)
    targets = inputs * coefficients .+ constant .+ rand(Normal(0,var), num_points)
    return inputs, targets
end

# generates a random data set with two or more distinct clusters, labeled 0 & 1
# returns two arrays: (x,y)
function generate_classification_data(num_points, input_dimension; num_clusters=2, lb=-1, ub=1, var=.1, covar=0)
    covar_matrix = zeros(input_dimension, input_dimension)
    covar_matrix[I(input_dimension)] .= var
    covar_matrix[.!I(input_dimension)] .= covar
    points_per_cluster = num_points ÷ num_clusters
    num_points = points_per_cluster * num_clusters
    centroids = [rand(Uniform(lb,ub), input_dimension) for c in 1:num_clusters]
    inputs = Matrix{Float64}(undef, num_points, input_dimension)
    targets = zeros(num_points)
    for c in 1:num_clusters
        targets[points_per_cluster*(c-1)+1 : points_per_cluster*c] .= c
    end
    shuffle!(targets)
    for c in 1:num_clusters
        inputs[targets .== c, :] = rand(MvNormal(centroids[c], covar_matrix), points_per_cluster)'
    end
    return inputs, targets .% 2
end