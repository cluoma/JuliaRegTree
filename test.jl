# Example of basic regression tree
#
# Colin Luoma

using DataFrames, RDatasets, Gadfly

# Include regression tree
include("regTree.jl")

# Import cars dataset and remove NA values
cars = dataset("rpart", "cu.summary")
cars = cars[!isna(cars[:, :Mileage]), :]
cars = cars[!isna(cars[:, :Reliability]), :]

# Split data and perform regression
dep = cars[:Mileage]
reg_data = cars[:, [:Price, :Country, :Reliability, :Type]]
tree_model = regression_tree(dep, reg_data, 10, 4, 6)

# Add predicted values to cars dataframe
pred_data = predict_tree(tree_model, reg_data)
cars[:pred_mileage] = pred_data

# Show tree and leaf nodes
print_tree(tree_model)
print_tree_leaves(tree_model)
