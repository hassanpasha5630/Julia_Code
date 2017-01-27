using MAT

# function for calculating squared distance
function sqdistance(p, q, pSOS, qSOS)
  return broadcast(+, pSOS, qSOS') - 2*p*q'
end

# open file
file = matopen("mnistData.mat")
mnist = read(file, "mnist")

# load data
train_images = mnist["train_images"]
train_labels = mnist["train_labels"]
test_images = mnist["test_images"]
test_labels = mnist["test_labels"]

# set training & testing
#for trainData in trainndx:


testndx =  1:10000


counter = 0  # this is the counter for the main loop
Error_Counter = 0 # this is the Error Counter for counting eror index
Information = Vector(7)
accuracyArray = Array{Float64}(0)

#while counter <= 300
while true

  if ( counter ==  0)
    counter = 100
  end
  trainndx = 1: counter
  tracker = counter

  ntrain = length(trainndx)
  ntest = length(testndx)
  Xtrain = float(reshape(train_images[:,:,trainndx], 28*28, ntrain)')
  Xtest  = float(reshape(test_images[:,:,testndx], 28*28, ntest)')

  ytrain = train_labels[trainndx]
  ytest  = test_labels[testndx]

  # Precompute sum of squares term
  XtrainSOS = sum(Xtrain.^2, 2)
  XtestSOS  = sum(Xtest.^2, 2)

  # fully solution takes too much memory so we will classify in batches
  # nbatches must be an even divisor of ntest, increase if you run out of memory
  if ntest > 1000
    nbatches = 50
    else
    nbatches = 5
  end

  nel = ntest ÷ nbatches
  batches = [nel*(i-1)+1:nel*i for i = 1:nbatches]

  ypred = zeros(UInt8, ntest)
  # classify
  for i=1:nbatches
    dist = sqdistance(Xtest[batches[i],:], Xtrain, XtestSOS[batches[i]], XtrainSOS)
    for j = 1:length(batches[i])
      closest = indmin(dist[j,:])
      ypred[batches[i][j]] = ytrain[closest]
    end
  end

  error_rate = mean(ypred .!= ytest)

  for i in eachindex(ypred)
    if(ypred[i] .!= ytest[i])
      using Plots
      pyplot()
      println("ERROR PRINT",ypred[i])
      img = flipdim(train_images[:,:,i], 1)  # ploting error images
      heatmap(1:28, 1:28, img, fillcolor = :grays, legend = false)
      Plots.gui()
      break ;
    end
  end

  print("Value was :", counter)
  print(" ")
  println("Error Rate : ", 100 * error_rate)


  acc = float(100 - (100 * error_rate ))

  print(" ")
  println("Accuracy Rate :" ,acc)
  push!(accuracyArray,acc)


  if ( tracker == 100)
      counter = 200
  elseif tracker == 200
      counter = 500
  elseif tracker == 500
      counter =1000
  elseif tracker == 1000
      counter = 2000
  elseif tracker==2000
      counter = 5000
  elseif tracker == 5000
      counter = 10000
  elseif tracker == 10000
      break ;
  end

end

# Checking to see if the values have safely entered into the array

for i in eachindex(accuracyArray)
  @show accuracyArray[i]
end

println("Plotting")

# Ploting the Information
using Plots
pyplot()
plot(accuracyArray,label = " accurcy")
Plots.gui()
