# Image Classifier using PyTorch

This repository contains code for an image classification neural network using PyTorch. The model is trained on the MNIST dataset to classify handwritten digits from 0 to 9.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)

You can install the necessary dependencies using:

```zsh
pip install -r requirements.txt
```

## Dataset

The code uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. The dataset will be automatically downloaded when you run the code.

## Model Architecture


The neural network model (ImageClassifier) is a convolutional neural network (CNN) with the following layers:

1.  Convolutional layer with 32 filters of size 3x3
    
2.  ReLU activation
    
3.  Convolutional layer with 64 filters of size 3x3
    
4.  ReLU activation
    
5.  Convolutional layer with 64 filters of size 3x3
    
6.  ReLU activation
    
7.  Flatten layer
    
8.  Fully connected layer with output size 10 (number of classes)
    

## Training the Model

The training code is currently commented out. To train the model, uncomment the training loop in the `if __name__ == "__main__":` block.

```py
# Training flow
if __name__ == "__main__":
    for epoch in range(10): # train for 10 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")

    with open('model.pt', 'wb') as f:
        save(clf.state_dict(), f)
```

The training process includes:

*   Loading the data in batches
    
*   Forward pass through the network
    
*   Calculating the loss using cross-entropy loss
    
*   Backward pass and optimization using Adam optimizer
    
*   Saving the trained model to model.pt
    

## Loading a Pre-trained Model

To use a pre-trained model, ensure that `model.pt` exists in the directory. The code will load this model for inference.

```py
with open('model.pt', 'rb') as f:
    clf.load_state_dict(load(f))
```

## Making Predictions

To make predictions on a new image, place the image (`img_3.jpg`) in the directory and run the code. The image will be converted to a tensor and passed through the model to get the predicted class.

```py
img = Image.open('img_3.jpg')
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

print(torch.argmax(clf(img_tensor)))
```

If you use a Nvidia GPU Use `CUDA` In Place of `cpu`.
The output will be the predicted digit (0-9).

## Usage

1.  **Train the model:**
    
    - Uncomment the training code and run the script.
        
    - The model will be saved as model.pt after training.
        
2.  **Load and use the pre-trained model:**
    
    - Ensure model.pt is present in the directory.
        
    - Run the script to load the model and make predictions on img\_3.jpg.
        
## License

Feel free to modify the code and experiment with different architectures and hyperparameters. Contributions are welcome!
