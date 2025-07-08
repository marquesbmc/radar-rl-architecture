# Biblioteca para operações de Deep Learning com TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, Layer, Activation, Concatenate, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K



# Model: nn_radar_dueling
# ==========================================



class nn_radar_dueling(tf.keras.Model):
    def __init__(self, num_actions=9, input_shape=(7, 7, 3)):
        super(nn_radar_dueling, self).__init__()
        self.num_actions = num_actions
        lr = 0.0001
        l2_regularization = 0.01
        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = MeanSquaredError()
        
        # Entrada e camada de combinação de cores
        self.input_layer = InputLayer(input_shape=input_shape, name="input_layer")
        self.color_comb_layer = ColorCombDepthwiseConv2D(kernel_size=(7, 7), activation='relu', name="color_comb_layer")

        # Camadas convolucionais
        self.conv1 = Conv2D(32, (4, 4), strides=(1, 1), activation=None, padding='same', name="conv1_layer")
        self.bn1 = BatchNormalization(name="bn1_layer")
        self.dropout_conv1 = Dropout(0.3, name="dropout_conv1_layer")  # Dropout na primeira camada convolucional

        self.conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding='same', name="conv2_layer")
        self.bn2 = BatchNormalization(name="bn2_layer")
        self.dropout_conv2 = Dropout(0.3, name="dropout_conv2_layer")  # Dropout na segunda camada convolucional
     
        # Camada de atenção espacial
        self.spatial_attention = SpatialAttentionModule(kernel_size=4)

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
        x = self.color_comb_layer(x)

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
        
        # Atenção espacial
        x = self.spatial_attention(x)
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

