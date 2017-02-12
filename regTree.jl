# Basic regression tree implementation
#
# Colin Luoma

using DataFrames, Combinatorics, RDatasets

type Node
    depth::Int

    left_child::Node
    right_child::Node

    value::Float64
    mse::Float64
    elements::Int
    split_col::Int
    split_val

    node_type::String

    Node() = new()
end

function mse(y1, y2)
    mse1 = sum((y1 - mean(y1)).^2)
    mse2 = sum((y2 - mean(y2)).^2)
    return (mse1 + mse2) / (length(y1) + length(y2))
end

function mse(y)
    mse1 = sum((y - mean(y)).^2)
    return (mse1) / length(y)
end

function regression_tree(y, x, minsplit=20, minbucket=round(minsplit/3), maxsplits=30; curdepth=0)
    root = Node()
    root.mse = mse(y)
    root.elements = length(y)
    root.value = mean(y)
    root.depth = curdepth+1;
    min_mse = mse(y)

    success = 0

    if length(y) < minsplit || root.depth > maxsplits
        return root
    end

    for i in 1:ncol(x)
        if typeof(x[:, i][1]) <: Number
            # Split by percentile when variable is a number
            for split in quantile(x[:, i], linspace(0.01, .99, 99))
                y1 = y[x[:, i] .< split]
                y2 = y[x[:, i] .>= split]

                new_mse = mse(y1, y2)

                if 1 - (new_mse/root.mse) > 0.002 && new_mse < min_mse && length(y1) >= minbucket && length(y2) >= minbucket
                    min_mse = new_mse
                    root.split_col = i
                    root.split_val = split
                    root.node_type = "numeric"
                    success = 1
                end
            end
        elseif typeof(x[:, i][1]) <: String
            # Split by combination when variable is string
            for split in combinations(unique(x[:, i]))
                y1 = y[Bool[elm in split for elm in x[:, i]]]
                y2 = y[!Bool[elm in split for elm in x[:, i]]]

                new_mse = mse(y1, y2)

                if 1 - (new_mse/root.mse) > 0.002 && new_mse < min_mse && length(y1) >= minbucket && length(y2) >= minbucket
                    min_mse = new_mse
                    root.split_col = i
                    root.split_val = split
                    root.node_type = "factor"
                    success = 1
                end
            end
        end
    end

    if success == 1
        if root.node_type == "numeric"
            root.left_child = regression_tree(
                y[x[:, root.split_col] .< root.split_val],
                x[x[:, root.split_col] .< root.split_val, :],
                minsplit,
                minbucket,
                maxsplits,
                curdepth=root.depth
            )
            root.right_child = regression_tree(
                y[x[:, root.split_col] .>= root.split_val],
                x[x[:, root.split_col] .>= root.split_val, :],
                minsplit,
                minbucket,
                maxsplits,
                curdepth=root.depth
            )
        else
            root.left_child = regression_tree(
                y[Bool[elm in root.split_val for elm in x[:, root.split_col]]],
                x[Bool[elm in root.split_val for elm in x[:, root.split_col]], :],
                minsplit,
                minbucket,
                maxsplits,
                curdepth=root.depth
            )
            root.right_child = regression_tree(
                y[!Bool[elm in root.split_val for elm in x[:, root.split_col]]],
                x[!Bool[elm in root.split_val for elm in x[:, root.split_col]], :],
                minsplit,
                minbucket,
                maxsplits,
                curdepth=root.depth
            )
        end
    end
    return root
end

function predict_tree(tree, x)
    ret = []
    for i in 1:nrow(x)
        cur_node = tree
        while isdefined(cur_node, :right_child)
            if cur_node.node_type == "numeric"
                if x[i, cur_node.split_col] < cur_node.split_val
                    cur_node = cur_node.left_child
                else
                    cur_node = cur_node.right_child
                end
            elseif cur_node.node_type == "factor"
                if x[i, cur_node.split_col] in cur_node.split_val
                    cur_node = cur_node.left_child
                else
                    cur_node = cur_node.right_child
                end
            end
        end
        ret = [ret; cur_node.value]
    end
    return ret
end

function print_tree_leaves(tree)
    if !isdefined(tree, :left_child)
        println(tree.elements, " : ", tree.value, " : ", tree.depth)
    else
        print_tree_leaves(tree.left_child)
        print_tree_leaves(tree.right_child)
    end
end

function print_tree(tree)
    if !isdefined(tree, :left_child)
        println(tree.elements, " : ", tree.value, " : ", tree.depth)
    else
        if tree.node_type == "numeric"
            println(names(reg_data)[tree.split_col], " < ", tree.split_val, " : ", tree.depth)
        else
            println(names(reg_data)[tree.split_col], " IN ", tree.split_val, " : ", tree.depth)
        end
        print_tree(tree.left_child)
        print_tree(tree.right_child)
    end
end
