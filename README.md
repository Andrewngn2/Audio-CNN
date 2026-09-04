Audio CNN project learning from Andreas Trolle

Built with 
Pytorch, Next.js, React, Tailwind, Python, Modal

This entire project is something I used to learn many fundamentals of working with neural networks and getting exposure to building front-end dashboard + back-end for the first time.
I'll first go over concepts I learned and then go into its implementation.

Concepts 

  Neural Networks- 
  
    Neurons-
      are the building blocks that have multiple inputs and a single output which is determined by inputs -> weights -> bias-> activation function. Each input is multiplied by a specific weight, then a bias is added on 
    and put into an activation function which gives us the output. y= f((w1*x1 =w2*x2 +b)). Weights and bias are randomly initialized but then tuned during training. In this project we use ReLU which only allows positive values
    to pass through so [12,-3,2] becomes [12,0,2] which introduces non linearity.

    Layers-
      Each layer has as many inputs as neurons from the last layer. Layers allow more weights to be tuned so patterns can be detected.
      
    Forward Pass-
       we pass inputs through hidden layers, which is what we call layers of neurons in which we dont really care about their outputs only the weights and biases so it doesnt make sense to track as we only care about final 
       ouput.

    Example of a NN
      lets say an image is a 28*28 pixel image. They are broken down into a single list of pixels so in this pictures case we will have 28*28=784 pixels. So we need an input layer of 784 neurons in which we will pass through hidden 
      layers with a bunch of neurons. Then there is a final layer of output neurons for each guess the NN can guess in which it will produce probabilities of each guess.

    Training-
      we call inputs the feature and the output the label. a sample is a single set of a feature and corresponding label. in training we  use 80% for training in the dataset and 20% for validation.
      1. We first run samples through the network with the forward pass and we compare the output with the true output to calculate loss(number representing how good or bas the network is at predicting) higher loss means more wrong.

      2. An optimizer determines how weight and bias values should be changes to improve loss. 
        this works through back propagation (calculating partial derivatives for all weights and biases) which gives directions to change the values to decrease loss.
        Learning rate affects how much each weight and bias is changed, which is important for making sure the changes in weights and biases are not undershot or overshot. This affects the speed of training.
      3. Run inference again(like step 1), and monitor loss over time to see if network improves. After a certain time the model converges and doesn't really get better.
      4. When network is trained, it should know samples well, and be able to predict sample labels not seen before. This is what the validation is for.

      pitfalls of training.

      underfitting- the model does not regress to the trend at all and training and validation loss doesn't significantly decrease. This is caused because the model is too simple (too little layers or too little neurons in each layer)
      or it is not trained enough and needs more epochs.
      overfitting - the model hyper focuses on points and doesn't generalize and memorizes the dataset. training loss is decreases significantly but validation loss doesn't go down that much after many epochs. This is likely because the 
      model too complex(too many neurons) and there is not enough data compared to nodes or model is trained too long.
      good fit- both validation and training loss go down.

  Convolutional Neural Networks

    This is network architecture that's good for classification ex: classifying a picture of a dog.
    lets say a picture is 1000*1000 pixels which would result in way too many input neurons which could result in overfitting. CNN can help with this.

    An image will have pixels and if the image has color it will have multiple channels for ex: RGB: 99, 136,30. For example if the image is greyscale it will only have one channel as it color will be between 0 and 1 in brightness.
    
    CNNs scan over images using a kernel to learn features from an image, instead of looking at each pixel one by one.

    Ex: this is an image and each pixel has value for specifically the red channel of rgb.
    [ 1,2,3,0]
    [4,5,6,1]
    [0,1,2,3]
    [1,3,2,4]
    This would be the input matrix for the CNN
    then a kernel(edge detector for vertical edges) will slide over the image above
    [1,0,-1]
    [1,0,-1]
    [1,0,-1]
     As it slides across the image math is done between the image and kernel (element wise multiplcation and summation is calculated every time the kernal slides aka convolution operation)
     if the kernel is in the top left
     (1x1) +(2x0)+ (3x-1)+
     (4x1)+(5x0)+(6x-1)+
     (0x1)+(1x0)+(2x-1) = -6
      [-6,_]
      [_,_]  -->> this is called a feature Map
      since the kernel can only cover 4 patches it will produce a  2x2
      The feature map will only have the most important features the kernel was looking for extracted. This would allow the CNN to learn how simple pieces combine to form complex objects

      Flow in neural network 
      raw image(1000x1000) --> Conv2d#0 (500x500) --> Conv2d #1 (250x250) --> Conv2d #2 (25x25) pixels --> flatten into 625 numbers and pass into 625 neurons in NN and recieve probability.
      an advantage instead of flattening the raw image is that we wont have so many weights and biases to tune as we can tune the kernel's weights and biases to reuse them. Another advantage of CNNs is solving translation
      invariance where linear layers have to recognize features like an eye at separate locations so it needs different weights to detect features at different places where as the kernel can identify features no matter the position
      because the kernels is tuned and slides across the image. CNNs retains spatial information.

      CNN depth /Layers

      A convolutional layer in a neural network doesn't just use one kernel it uses many kernels at once. 
      We define in an dout channels when we use a CNN(depth)
      depth it recieves
      Color channels- RGB would have 3 in channels
      greyscale- brightness would have 1 in channel
      Depth it ouputs
      Setting out 64 out channels would tell it to create 64 kernels and generate 64 feature maps
      Subsequent convolutional layers need to have amount of in channels as the previous out channels
      You would have one feature map looking for an eye while another looks for a tail so on and so forth.

      After we get 64 feature maps we append them and put them in a linear layer. The reasoning the flattening of information isnt a problem here is because the convolutional layer has done the job of finding the eye
      After flattening the feature maps  just act as a signal to the linear layer that the eye exists and shifts to reasoning with those features and not finding them.

      Training CNN
      When you pick a number of kernels each kernel is initialized to a set of random values. Kernel values are tuned during back propagation 

      Width vs. Depth of CNN

      width would refer to how many out channels a conv layer produces. The width controls the variety of features. Different kernels become specialized in finding different features. A layer at a low level of abstraction(earlier layer) 
      with 64 kernels can find 64 different types of simple edges and colors. 
      Depth would refer to how many conv layers are after each other. Each layer will combine features from last layer to create something more complex. EX:  conv layer 1: finds edges --> conv layer 2: finds combinations of edges
      --> conv layer 3: finds parts of objects like a face. 
      You need to have an appropriate amount of out channels in a preceding convolutional layer so the next layer has data to work with. So the first layer would need to find enough edges so the second layer can use all feature maps to 
      identify combinations of edges.

      Pooling, activation functions, and Batch normalization 
      

      

      
      
     
    
    
      
        
        
        
    
      
      
    
    
  

