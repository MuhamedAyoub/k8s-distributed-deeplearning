# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import errno
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import numpy as np
import argparse

# Configure TensorFlow logging
tf.get_logger().setLevel('INFO')

# Enable mixed precision (optional)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Training settings
parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--lr', default=0.001, type=float, help='Adam learning rate')
parser.add_argument('--num-steps', default=20000, type=int, help='Number of training steps')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size')
args = parser.parse_args()


class MNISTModel(tf.keras.Model):
    """2-layer convolution model for MNIST classification."""
    
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First conv layer will compute 32 features for each 5x5 patch
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), 
                                           activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                                              padding='same')
        
        # Second conv layer will compute 64 features for each 5x5 patch
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), 
                                           activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                                              padding='same')
        
        # Flatten the output for the dense layer
        self.flatten = tf.keras.layers.Flatten()
        
        # Densely connected layer with 1024 neurons
        self.dense = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        # Reshape inputs to 4D tensor [batch_size, height, width, channels]
        x = tf.reshape(inputs, [-1, 28, 28, 1])
        
        # Apply convolutional layers and pooling
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Flatten and apply fully connected layers
        x = self.flatten(x)
        x = self.dense(x)
        
        # Apply dropout only during training
        if training:
            x = self.dropout(x)
        
        # Output logits
        return self.output_layer(x)


def main():
    # Initialize Horovod
    hvd.init()
    
    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        # Memory growth needs to be set before GPUs have been initialized
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Create directory for Keras datasets if it doesn't exist
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        f'MNIST-data-{hvd.rank()}')

    # Normalize the features between 0 and 1
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Reshape for the model input
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    # Create the model
    model = MNISTModel()
    
    # Adjust learning rate based on number of GPUs
    lr_scaler = hvd.size()
    if args.use_adasum:
        lr_scaler = hvd.local_size() if hvd.nccl_built() else 1
    
    opt = tf.keras.optimizers.Adam(args.lr * lr_scaler)
    
    # Horovod: Add Horovod DistributedOptimizer
    opt = hvd.DistributedOptimizer(
        opt, op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Set up callbacks
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        
        # Horovod: average metrics among workers at the end of every epoch.
        hvd.callbacks.MetricAverageCallback(),
    ]

    # Add TensorBoard callback for worker 0 only
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='./logs'))
        
        # Save checkpoints only on worker 0 to prevent conflicts
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            './checkpoints/mnist-{epoch}.h5',
            save_best_only=True))

    # Calculate steps per epoch and validation steps
    steps_per_epoch = max(1, len(x_train) // (args.batch_size * hvd.size()))
    validation_steps = max(1, len(x_test) // args.batch_size)
    
    # Calculate the total number of epochs based on num_steps and steps_per_epoch
    epochs = max(1, args.num_steps // steps_per_epoch)

    # Train the model
    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0
    )
    
    # Evaluate the model on the test data (only on worker 0)
    if hvd.rank() == 0:
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        # Save the final model
        model.save('./final_model')


if __name__ == "__main__":
    main()