# Model: nn_per
# ==========================================

# Biblioteca para operações de Deep Learning com TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError


class nn_per(tf.keras.Model):
    def __init__(self, num_actions=9, input_shape=(7, 7, 3), discount_factor=0.99):
        super(nn_per, self).__init__()
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        lr = 0.0001
        l2_regularization = 0.01
        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = MeanSquaredError()

        # Entrada e camada de combinação de cores
        self.input_layer = InputLayer(input_shape=input_shape, name="input_layer")

        # Camadas convolucionais
        self.conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation=None, padding='same', name="conv1_layer")
        self.bn1 = BatchNormalization(name="bn1_layer")
        self.dropout_conv1 = Dropout(0.3, name="dropout_conv1_layer")

        self.conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding='same', name="conv2_layer")
        self.bn2 = BatchNormalization(name="bn2_layer")
        self.dropout_conv2 = Dropout(0.3, name="dropout_conv2_layer")

        # Flatten e camadas densas
        self.flatten = Flatten(name="flatten_layer")
        self.dropout_flatten = Dropout(0.3, name="dropout_flatten_layer")

        self.dense1 = Dense(64, activation='relu', name="dense1_layer", kernel_regularizer=l2(l2_regularization))
        self.dropout_dense1 = Dropout(0.4, name="dropout_dense1_layer")

        self.dense2 = Dense(32, activation='relu', name="dense2_layer", kernel_regularizer=l2(l2_regularization))
        self.dropout_dense2 = Dropout(0.4, name="dropout_dense2_layer")

        self.dense_output = Dense(num_actions, activation='linear', name="dense_output_layer", kernel_regularizer=l2(l2_regularization))

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x, name="relu1")
        x = self.dropout_conv1(x, training=training)  # Aplicação do Dropout após conv1

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x, name="relu2")
        x = self.dropout_conv2(x, training=training)  # Aplicação do Dropout após conv2

        x = self.flatten(x)
        x = self.dropout_flatten(x, training=training)  # Aplicação do Dropout após Flatten

        x = self.dense1(x)
        x = self.dropout_dense1(x, training=training)  # Aplicação do Dropout após dense1
        x = self.dense2(x)
        x = self.dropout_dense2(x, training=training)  # Aplicação do Dropout após dense2

        Q_values = self.dense_output(x)
        return Q_values
    

    def training_step(self, batch_data):
        states, actions, targetQ, weights = batch_data
        with tf.GradientTape() as tape:
            Q_values = self(states, training=True)
            actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            Q = tf.reduce_sum(Q_values * actions_onehot, axis=1)
            td_errors = targetQ - Q
            weighted_loss = tf.reduce_mean(weights * tf.square(td_errors))

        grads = tape.gradient(weighted_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return weighted_loss


    def predict_action(self, state):
        Q_values = self(state)
        return tf.argmax(Q_values, axis=1)[0].numpy()  # Retorna a ação com o maior valor Q como um número Python

    def save_model(self, file_path):
        self.save(file_path)
        print(f"Modelo salvo em: {file_path}")

    def load_model(self, file_path):
        model_loaded = tf.keras.models.load_model(file_path)
        return model_loaded


