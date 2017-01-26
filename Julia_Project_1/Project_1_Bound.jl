using Distances
using Plots
pyplot()

# setup data
D = readcsv("iris.csv")
X_train = D[:, 1:2]
y_train = D[:, end]
X_transpose = X_train'

# setup grid
x1 = collect(2:0.01:5)
x2 = collect(0:0.01:3)

# compute 1NN decision
decision = zeros(length(x1), length(x2))
for i = 1:length(x1)
  for j = 1:length(x2)
    point = [x1[i], x2[j]]

    # compute euclidan distance from the point to all training data
    dist = colwise(Euclidean(), X_transpose, point);

    # sort the distance, get the index
    idx_sorted = sortperm(dist)

    # find the class of the nearest neighbour
    pred = y_train[idx_sorted[1]]

    decision[i,j] = pred
  end
end

# color gradient for the classes
# class 1 = light red, 2 = light green, 3 = light blue
color = cgrad([RGBA(1,0.8,0.8), RGBA(0.8,1,0.8), RGBA(0.8, 0.8, 1)])

# plot decision & data
heatmap(x1, x2, decision', fillcolor = color, legend = false)
scatter!(X_train[y_train .== 1, 1], X_train[y_train .== 1, 2], color = :red, markerstrokewidth = 0)
scatter!(X_train[y_train .== 2, 1], X_train[y_train .== 2, 2], color = :green, markerstrokewidth = 0)
scatter!(X_train[y_train .== 3, 1], X_train[y_train .== 3, 2], color = :blue, markerstrokewidth = 0)
Plots.gui()
