# SinePredict
This uses a Neural network with hidden Dense layers to predict what the sin of a number would be. 

In SinWaveEstimate.py a Sequential model is created with 4 dense hidden layers with gelu activation.
This model is trained on np.sin(). 

# Installation
pip install tensorflow 2.5

pip install matplotlib

create a logs folder under the main directory or change the tensorboard_callback dir

# Usage
To use you can either make the prediction right after training the model or you can use loadTest.py. This will pull the saved model, and make a prediction based on what you give it
