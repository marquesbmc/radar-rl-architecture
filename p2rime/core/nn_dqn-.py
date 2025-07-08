
# Biblioteca para operações de Deep Learning com TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, BatchNormalization
from tensorflow.keras.optimizers import Adam  # Otimizador para ajuste dos pesos da rede neural
from tensorflow.keras.losses import MeanSquaredError  # Função de perda para treinamento da rede neural


class nn_dqn(tf.keras.Model):
    def __init__(self, num_actions=8, input_shape=(7, 7, 3)):
        super(nn_dqn, self).__init__()
        self.num_actions = num_actions
        self.optimizer = Adam(learning_rate=0.0001)
        self.loss_fn = MeanSquaredError()

        self.input_layer = InputLayer(input_shape=input_shape)
        self.conv1 = Conv2D(32, (6, 6), strides=(1, 1), activation='relu', padding='same', name='conv1')
        self.batch_norm1 = BatchNormalization(name='batch_norm1')
        self.conv2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', name='conv2')
        self.batch_norm2 = BatchNormalization(name='batch_norm2')
        self.conv3 = Conv2D(128, (4, 4), strides=(1, 1), activation='relu', padding='same', name='conv3')
        self.batch_norm3 = BatchNormalization(name='batch_norm3')

        self.flatten = Flatten(name='flatten')

        self.dense1 = Dense(128, activation='relu', name='dense1')
        self.dense2 = Dense(64, activation='relu', name='dense2')
        self.dense3 = Dense(32, activation='relu', name='dense3')

        self.dense_output = Dense(num_actions, activation='linear', name='dense_output')


    def call(self, inputs):
        # Corrigindo a forma de acessar a dimensão de entrada
        if inputs.shape[1:] != (7, 7, 3):
            raise ValueError(f"Input shape {inputs.shape} does not match expected shape (7, 7, 3)")
        x = self.input_layer(inputs)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        # x = self.dropout1(x)
        x = self.dense2(x)
        # x = self.dropout2(x)
        x = self.dense3(x)
        Q_values = self.dense_output(x)
        return Q_values

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
    
# Exemplo de como usar o modelo com input aleatório
if __name__ == "__main__":
    model = nn_dqn(num_actions=8, input_shape=(7, 7, 3))
    random_input = np.random.random((1, 7, 7, 3))
    predicted_actions = model(random_input)
    print("Predicted Q-values:", predicted_actions)