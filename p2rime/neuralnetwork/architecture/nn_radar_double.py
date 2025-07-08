# Biblioteca para operações de Deep Learning com TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, BatchNormalization, Layer, Activation, Concatenate, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K

# Model: nn_radar_double
# ==========================================


class nn_radar_double(tf.keras.Model):
    def __init__(self, num_actions=9, input_shape=(7, 7, 3), discount_factor=0.99):
        super(nn_radar_double, self).__init__()
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        lr = 0.0001
        l2_regularization = 0.01

        # Otimizador e função de perda
        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = MeanSquaredError()

        # Camada de entrada e processamento inicial de cores
        self.input_layer = InputLayer(input_shape=input_shape, name="input_layer")
        self.color_comb_layer = ColorCombDepthwiseConv2D(kernel_size=(7, 7), activation='relu', name="color_comb_layer")

        # Camadas convolucionais da rede principal
        self.conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation=None, padding="same")
        self.batch_norm1 = BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding="same")
        self.batch_norm2 = BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        # Módulo de atenção espacial
        self.spatial_attention = SpatialAttentionModule(kernel_size=4)

        # Camadas densas
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2(l2_regularization))
        self.dense2 = Dense(32, activation="relu", kernel_regularizer=l2(l2_regularization))
        self.output_layer = Dense(num_actions, activation="linear", kernel_regularizer=l2(l2_regularization))

        # Camadas da rede-alvo (adicionadas as mesmas modificações)
        self.target_color_comb_layer = ColorCombDepthwiseConv2D(kernel_size=(7, 7), activation='relu', name="target_color_comb_layer")
        self.target_conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation=None, padding="same")
        self.target_batch_norm1 = BatchNormalization()
        self.target_relu1 = tf.keras.layers.ReLU()

        self.target_conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding="same")
        self.target_batch_norm2 = BatchNormalization()
        self.target_relu2 = tf.keras.layers.ReLU()

        self.target_spatial_attention = SpatialAttentionModule(kernel_size=4)

        self.target_flatten = Flatten()
        self.target_dense1 = Dense(64, activation="relu", kernel_regularizer=l2(l2_regularization))
        self.target_dense2 = Dense(32, activation="relu", kernel_regularizer=l2(l2_regularization))
        self.target_output_layer = Dense(num_actions, activation="linear", kernel_regularizer=l2(l2_regularization))

    def call(self, inputs, training=False):
        """Chama a rede principal para inferência."""
        x = self.input_layer(inputs)
        x = self.color_comb_layer(x)

        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.relu2(x)

        x = self.spatial_attention(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

    def target_call(self, inputs, training=False):
        """Chama a rede-alvo para inferência."""
        x = self.input_layer(inputs)
        x = self.target_color_comb_layer(x)

        x = self.target_conv1(x)
        x = self.target_batch_norm1(x, training=training)
        x = self.target_relu1(x)

        x = self.target_conv2(x)
        x = self.target_batch_norm2(x, training=training)
        x = self.target_relu2(x)

        x = self.target_spatial_attention(x)
        x = self.target_flatten(x)
        x = self.target_dense1(x)

        x = self.target_dense2(x)
        return self.target_output_layer(x)

    def update_target_network(self):
        """Sincroniza os pesos da rede principal para a rede-alvo."""
        self.target_color_comb_layer.set_weights(self.color_comb_layer.get_weights())
        self.target_conv1.set_weights(self.conv1.get_weights())
        self.target_batch_norm1.set_weights(self.batch_norm1.get_weights())
        self.target_conv2.set_weights(self.conv2.get_weights())
        self.target_batch_norm2.set_weights(self.batch_norm2.get_weights())
        self.target_dense1.set_weights(self.dense1.get_weights())
        self.target_dense2.set_weights(self.dense2.get_weights())
        self.target_output_layer.set_weights(self.output_layer.get_weights())

    def training_step(self, batch_data):
        """Realiza uma etapa de treinamento com Double DQN."""
        states, actions, targetQ = batch_data
        next_states, rewards, dones = targetQ
        with tf.GradientTape() as tape:
            # Predições da rede principal
            Q_values = self(states, training=True)
            actions_onehot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            Q = tf.reduce_sum(Q_values * actions_onehot, axis=1)

            # Double DQN: Calcula o valor-alvo
            main_Q_values_next = self(next_states)  # Rede principal para selecionar a melhor ação
            next_actions = tf.argmax(main_Q_values_next, axis=1)
            target_Q_values_next = self.target_call(next_states)  # Rede-alvo para calcular Q
            target_Q = tf.reduce_sum(target_Q_values_next * tf.one_hot(next_actions, self.num_actions), axis=1)
            target_Q = rewards + (1 - dones) * self.discount_factor * target_Q

            # Calcula o loss
            loss = self.loss_fn(target_Q, Q)

        # Gradientes e atualização dos pesos
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def predict_action(self, state):
        """Prediz a melhor ação para um dado estado."""
        Q_values = self(tf.expand_dims(state, axis=0))
        return tf.argmax(Q_values, axis=1).numpy()[0]

    def save_model(self, file_path):
        """Salva os pesos do modelo principal."""
        self.save_weights(file_path)
        print(f"Modelo salvo em: {file_path}")

    def load_model(self, file_path):
        """Carrega os pesos do modelo principal."""
        self.load_weights(file_path)
        print(f"Modelo carregado de: {file_path}")


# Class - RADAR
# ==========================================


class ColorCombDepthwiseConv2D(Layer):
    def __init__(self, kernel_size=(7, 7), activation='relu', padding='same', **kwargs):
        super(ColorCombDepthwiseConv2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding

        # Convoluções individuais para cores puras
        self.conv_r = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_r")
        self.bn_r = BatchNormalization(name="bn_r")
        
        self.conv_g = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_g")
        self.bn_g = BatchNormalization(name="bn_g")
        
        self.conv_b = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_b")
        self.bn_b = BatchNormalization(name="bn_b")

        # Convoluções para combinações específicas (Magenta, Ciano, Amarelo)
        self.conv_magenta = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_magenta")
        self.bn_magenta = BatchNormalization(name="bn_magenta")
        
        self.conv_cyan = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_cyan")
        self.bn_cyan = BatchNormalization(name="bn_cyan")
        
        self.conv_yellow = Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, activation=None, name="conv_yellow")
        self.bn_yellow = BatchNormalization(name="bn_yellow")

    def call(self, inputs, training=False):
        r, g, b = tf.split(inputs, num_or_size_splits=3, axis=-1)

        # Convoluções individuais com BatchNormalization
        r_out = self.bn_r(self.conv_r(r), training=training)
        g_out = self.bn_g(self.conv_g(g), training=training)
        b_out = self.bn_b(self.conv_b(b), training=training)

        # Combinações de canais com BatchNormalization
        magenta_out = self.bn_magenta(self.conv_magenta(r + b), training=training)
        cyan_out = self.bn_cyan(self.conv_cyan(g + b), training=training)
        yellow_out = self.bn_yellow(self.conv_yellow(r + g), training=training)

        # Concatenando saídas
        outputs = Concatenate(axis=-1, name="concat_colors")([r_out, g_out, b_out, magenta_out, cyan_out, yellow_out])

        # Ativação
        outputs = Activation(self.activation, name="activation_colors")(outputs)
        return outputs


class SpatialAttentionModule(Layer):
    def __init__(self, kernel_size=4):
        super(SpatialAttentionModule, self).__init__()
        self.conv = Conv2D(1, kernel_size=kernel_size, padding='same', activation=None, name="attention_conv")
    
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        squared_inputs = K.square(inputs)
        l2_pool = tf.sqrt(tf.reduce_mean(squared_inputs, axis=-1, keepdims=True) + 1e-6)

        concat = Concatenate(axis=-1, name="concat_attention")([avg_pool, l2_pool])
        attention_map = self.conv(concat)
        attention_map = Activation('sigmoid', name="sigmoid_attention")(attention_map)
        return Multiply(name="apply_attention")([inputs, attention_map])