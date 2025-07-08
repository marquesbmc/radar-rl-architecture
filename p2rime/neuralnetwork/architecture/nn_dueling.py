# Model: nn_dueling
# ==========================================

# Biblioteca para operações de Deep Learning com TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError


class nn_dueling(tf.keras.Model):
    def __init__(self, num_actions=9, input_shape=(7, 7, 3)):
        super(nn_dueling, self).__init__()
        self.num_actions = num_actions
        lr = 0.0001
        l2_regularization = 0.01
        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = MeanSquaredError()
        
        # Entrada e camada de combinação de cores
        self.input_layer = InputLayer(input_shape=input_shape, name="input_layer")

        # Camadas convolucionais
        self.conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation=None, padding='same', name="conv1_layer")
        self.bn1 = BatchNormalization(name="bn1_layer")
        self.dropout_conv1 = Dropout(0.3, name="dropout_conv1_layer")  # Dropout na primeira camada convolucional

        self.conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding='same', name="conv2_layer")
        self.bn2 = BatchNormalization(name="bn2_layer")
        self.dropout_conv2 = Dropout(0.3, name="dropout_conv2_layer")  # Dropout na segunda camada convolucional


        # Flatten
        self.flatten = Flatten(name="flatten_layer")
        self.dropout_flatten = Dropout(0.4, name="dropout_flatten_layer")  # Dropout após Flatten

        # Camadas densas compartilhadas
        self.dense_shared1 = Dense(64, activation='relu', name="shared_dense1", kernel_regularizer=l2(l2_regularization))
        self.dropout_shared = Dropout(0.4, name="dropout_shared_layer")  # Dropout na camada compartilhada

        # Rede para Valor (V)
        self.value_dense = Dense(32, activation='relu', name="value_dense", kernel_regularizer=l2(l2_regularization))
        self.dropout_value = Dropout(0.4, name="dropout_value_layer")  # Dropout na rede de valor
        self.value_output = Dense(1, activation='linear', name="value_output", kernel_regularizer=l2(l2_regularization))

        # Rede para Vantagem (A)
        self.advantage_dense = Dense(32, activation='relu', name="advantage_dense", kernel_regularizer=l2(l2_regularization))
        self.dropout_advantage = Dropout(0.4, name="dropout_advantage_layer")  # Dropout na rede de vantagem
        self.advantage_output = Dense(num_actions, activation='linear', name="advantage_output", kernel_regularizer=l2(l2_regularization))

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)

        # Camada convolucional 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x, name="relu1")
        x = self.dropout_conv1(x, training=training)  # Dropout aplicado

        # Camada convolucional 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x, name="relu2")
        x = self.dropout_conv2(x, training=training)  # Dropout aplicado
        

        x = self.flatten(x)
        x = self.dropout_flatten(x, training=training)  # Dropout aplicado após Flatten

        # Camadas compartilhadas
        x = self.dense_shared1(x)
        x = self.dropout_shared(x, training=training)  # Dropout aplicado na camada compartilhada

        # Valor (V)
        v = self.value_dense(x)
        v = self.dropout_value(v, training=training)  # Dropout aplicado na rede de valor
        v = self.value_output(v)

        # Vantagem (A)
        a = self.advantage_dense(x)
        a = self.dropout_advantage(a, training=training)  # Dropout aplicado na rede de vantagem
        a = self.advantage_output(a)

        # Combina V e A para calcular Q
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q
    
    def training_step(self, batch_data):
        states, actions, targetQ = batch_data
        with tf.GradientTape() as tape:
            Q_values = self(states, training=True)
            actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            Q = tf.reduce_sum(Q_values * actions_onehot, axis=1)
            loss = self.loss_fn(targetQ, Q)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def predict_action(self, state):
        Q_values = self(state)
        return tf.argmax(Q_values, axis=1)[0].numpy()  # Retorna a ação com o maior valor Q como um número Python

    def save_model(self, file_path):
        self.save(file_path)
        print(f"Modelo salvo em: {file_path}")

    def load_model(self, file_path):
        model_loaded = tf.keras.models.load_model(file_path)
        return model_loaded
    

