# Image Sequence LSTM+CNN
# Objective
To develop a  Recurrent Neural Network (RNN) architecture shown in the figure below. 
![LSTM-CNN](https://user-images.githubusercontent.com/3444740/55726172-cb74c880-5a2c-11e9-8e42-9ab605816b32.jpg)
CNNs are pre-trained and used as feature generators.

# Simple CNN
![MNIST](https://user-images.githubusercontent.com/3444740/55726114-b39d4480-5a2c-11e9-94f8-d1b1f24580c9.jpg)
Trained on MNIST data to recognize hand written digits.

# Sample Application
![Pos](https://user-images.githubusercontent.com/3444740/55726122-b5670800-5a2c-11e9-9805-06da1e6eb7f9.jpg)

![Neg](https://user-images.githubusercontent.com/3444740/55726134-b7c96200-5a2c-11e9-8fe6-80312a07a48d.jpg)




# Dependencies
          1. Keras 2.2.4 or >
          2. Tensorflow 1.13 or >

# Commands:
          python main.py train cnn 
          python main.py train lstm 
          python main.py test cnn 
          python main.py test lstm -l n      # n-optinal length of the image sequence
          
