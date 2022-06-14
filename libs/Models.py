#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

def Choose_Model(args, mean, std):
    if args.Model == 'ANN':
        model = ANN(args, mean, std)
    elif args.Model == 'ConvLSTM':
        model = ConvLSTM(args, mean, std)
    elif args.Model == 'GCN':
        model = GCN(args, mean, std)
    elif args.Model == "GRU":
        model = GRU(args, mean, std)
    # elif args.Model == "CASCNN":
    #     model = CASCNN(args, mean, std)
    elif args.Model == "GEML":
        model = GEML(args, mean, std)
    model.summary()
    return model

def ANN(args, mean, std):
    ANN_Units = [int(unit) for unit in args.ANN_Units.split(",")]
    ANN_Activations = [act for act in args.ANN_Activations.split(",")]
    K = len(ANN_Units)
    print("mean:{:.2f},std:{:.2f}".format(mean, std))

    input = tf.keras.Input(shape=(args.P, args.N, args.N))  # (None, P, N, N)
    output = tf.keras.layers.Permute((2, 3, 1))(input)  # (None, N, N, P)
    for k in range(K):
        output = tf.keras.layers.Dense(units=ANN_Units[k], activation=ANN_Activations[k])(output)  # (None,N,N,F)
        # output = tf.keras.layers.Dropout(0.5)(output)
    output = output * std + mean
    output = tf.keras.layers.Dense(units=1, activation="relu")(output)  # (None,N,N,1)
    output = tf.keras.layers.Reshape((args.N, args.N))(output)  # (None, N, N)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model



# def Ridge(args):
    # 数据变形:X=(M,P,N,N)=>(M,N,N,P)=>(M*N*N,P),Y=(M,N,N)=>(M*N*N,1)
    if args.Model == "Ridge":
        x = np.transpose(x, perms=[0,2,3,1])
        x = np.reshape(x, newshape=(-1, args.P))
        y = np.reshape(y, newshape=(-1, 1))
        inputs = x
        outputs = y

##################################################################################################  ANN
# # 以站点的D分布为预测目标的话，效果很差
# def ANN(args):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(args.P, args.N, args.N))) # 输入是(None, P, N, N)
#     model.add(tf.keras.layers.Permute((2, 1, 3))) # (None, N, P, N)
#     model.add(tf.keras.layers.Reshape((args.N, args.N*args.P))) # (None, N, PN)
#     model.add(tf.keras.layers.Dense(units=args.ANN_Units[0], activation="relu"))
#     model.add(tf.keras.layers.Dense(units=args.ANN_Units[1], activation="relu"))
#     model.add(tf.keras.layers.Dense(units=args.N, activation="relu"))
#     model.add(tf.keras.layers.Reshape((args.N, args.N)))
#     model.output_shape # 输出是 (None, N, N)
#     return model

'''
(1) 反向标准化，再relu,不线性变换，效果从1.02 变2.7，而mean=3.10
(2)调一下变成1.2-1.6，也很差
(3)不反向标准化，效果非常差，1.6,且不变
(4)以站点的D分布为预测目标的话，效果很差,从1.02变1.56
'''


##################################################################################################  GRU
# def GRU(args):
#     input = tf.keras.Input(shape=(args.P, args.N, args.N)) # (None, P, N, N)
#     output = tf.keras.layers.Permute((2, 1, 3))(input) # (None, N, P, N)
#     output = tf.reshape(output, shape=(-1, args.P, args.N)) # (None*N, P, N)
#     output = tf.keras.layers.GRU(units=args.GRU_Units[0], return_sequences=True)(output)
#     output = tf.keras.layers.GRU(units=args.GRU_Units[1], return_sequences=True)(output)
#     output = tf.keras.layers.GRU(units=args.GRU_Units[-1])(output) # (None*N, N)
#     output = tf.reshape(output, shape=(-1, args.N, args.N)) # (None*N, N) => (None, N, N)
#     model = tf.keras.Model(inputs=input, outputs=output)
#     model.output_shape
#     return model

def GRU(args, mean, std):
    GRU_Units = [int(unit) for unit in args.GRU_Units.split(",")]
    K = len(GRU_Units)
    print("mean:{:.2f},std:{:.2f}".format(mean, std))

    input = tf.keras.Input(shape=(args.P, args.N, args.N)) # (None, P, N, N)
    output = tf.keras.layers.Permute((2, 3, 1))(input) # (None, N, N, P)
    output = tf.reshape(output, shape=(-1, args.P, 1)) # (None*N*N, P, 1)

    for k in range(K-1):
        output = tf.keras.layers.GRU(units=GRU_Units[k], return_sequences=True)(output)
    output = tf.keras.layers.GRU(units=GRU_Units[k])(output) # (None*N*N, F)

    output = output * std + mean
    output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(output)
    output = tf.reshape(output, shape=(-1, args.N, args.N)) # 输出是 (None, N, N)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.output_shape
    return model

##################################################################################################  ANN2
def ConvLSTM(args, mean, std):
    Filters = [8, 8, 1]
    Kernel = (3, 3)
    input = tf.keras.Input(shape=(args.P, args.N, args.N))  # (None, P, N, N)
    output = tf.keras.layers.Reshape((args.P, args.N, args.N, 1))(input) # (None, P, N, N, 1)
    for k in range(len(Filters)-1):
        output = tf.keras.layers.ConvLSTM2D(filters=Filters[k], kernel_size=Kernel, padding='same', data_format='channels_last',return_sequences=True)(output)
    output = tf.keras.layers.ConvLSTM2D(filters=Filters[-1], kernel_size=Kernel, padding='same', data_format='channels_last',return_sequences=False)(output)# (None, N, N, 1)
    print("output:{}".format(output.shape))
    output = output * std + mean
    output = tf.keras.layers.Dense(units=1, activation="relu")(output) #(None,N,N,1)
    output = tf.keras.layers.Reshape((args.N, args.N))(output) # (None, N, N)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

##################################################################################################  GCN
# 输入 (B, N, F) (N, N)，输出 (B, N, F)
class GCN_Layer(tf.keras.layers.Layer):
    def __init__(self, L, Units, Activation="relu", **kwargs):
        super(GCN_Layer, self).__init__()
        self.L = L
        self.Units = Units
        self.D = tf.keras.layers.Dense(units=self.Units, activation = Activation)

    def get_config(self):
        config = {"L": self.L,"D":self.D,"Units":self.Units}
        base_config = super(GCN_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def convolution(self, L, X, K=3):
        y1 = X
        y2 = tf.matmul(L, y1)
        if K > 2:
            total = [y1, y2]
            for k in range(3, K + 1):
                y3 = 2 * tf.matmul(L, y2) - y1
                total.append(y3)
                y1, y2 = y2, y3
            total = tf.concat(total, axis=-1)
            y2 = total
        return y2

    def call(self, inputs):
        X = self.convolution(self.L, inputs)
        Y = self.D(X)
        return Y
'''
以D分布为预测目标的效果很差；
'''
# def GCN(args, mean, std):
#     L = np.load(args.GCN_A_Path)['arr_0'].astype(np.float32)
#     input = tf.keras.Input(shape=(args.P, args.N, args.N))
#     output = tf.keras.layers.Permute((2, 3, 1))(input)# (None, P, N, N) => (None, N, N, P)
#     output_1 = tf.keras.layers.Reshape((args.N, args.P*args.N))(output) # (None, N, N, P) => (None, N, PN)
#     output_2 = GCN_Layer(L, int(args.GCN_Units.split(",")[0]))(output_1)# (None, N, N) => (None, N, F)
#     output = GCN_Layer(L, args.N)(output_2) # (None, N, F) => (None, N, N)
#     output = tf.reshape(output, (-1, args.N, args.N, 1)) # (None, N, N, 1)
#
#     output = output * std + mean
#     output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(output)  # (None, N, N, 1)
#     output = tf.reshape(output, shape=(-1, args.N, args.N))  # 输出是 (None, N, N)
#     model = tf.keras.Model(inputs=input, outputs=output)
#     model.output_shape  # 输出是 (None, N, N)
#     return model

def GCN(args, mean, std):
    GCN_Units = [int(unit) for unit in args.GCN_Units.split(",")]
    GCN_Activations = [act for act in args.GCN_Activations.split(",")]
    K1 = len(GCN_Units)

    ANN_Units = [int(unit) for unit in args.ANN_Units.split(",")]
    ANN_Activations = [act for act in args.ANN_Activations.split(",")]
    K2 = len(ANN_Units)

    print("mean:{:.2f},std:{:.2f}".format(mean, std))

    L = np.load(args.GCN_A_Path)['arr_0'].astype(np.float32)
    input = tf.keras.Input(shape=(args.P, args.N, args.N))
    output = tf.reshape(input, shape=(-1, args.N, args.N)) # (None*P,N,N)
    for k in range(K1):
        output = GCN_Layer(L, Units=args.N, Activation=GCN_Activations[k])(output) # (None*P,N,N) => (None*P,N,N)

    output = tf.reshape(output, (-1, args.P, args.N, args.N)) # (None, P, N, N)
    output = tf.keras.layers.Permute((2, 3, 1))(output)  # (None, N, N, P)
    for k in range(K2):
        output = tf.keras.layers.Dense(units=ANN_Units[k], activation=ANN_Activations[k])(output)  # (None,N,N,F)
        # output = tf.keras.layers.Dropout(0.5)(output)
    output = output * std + mean
    output = tf.keras.layers.Dense(units=1, activation="relu")(output)  # (None,N,N,1)
    output = tf.keras.layers.Reshape((args.N, args.N))(output)  # (None, N, N)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.output_shape  # 输出是 (None, N, N)
    return model

##################################################################################################  CASCNN
# channel-wise attention: (B,N,N,C)=>(B,N,N,C)
class channel_wise(tf.keras.layers.Layer):
    def __init__(self, N=80, C=1, R=1, **kwargs):
        super(channel_wise, self).__init__()
        # channel_wise related layers
        self.AvgPool = tf.keras.layers.AveragePooling2D(pool_size=(N, N), strides=1)
        self.FC_1 = tf.keras.layers.Dense(units=C / R, activation='relu')
        self.FC_2 = tf.keras.layers.Dense(units=C / R, activation='sigmoid')

    def get_config(self):
        config = {"AvgPool": self.AvgPool, "FC_1": self.FC_1, "FC_2": self.FC_2}
        base_config = super(channel_wise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, Input):
        out = self.AvgPool(Input)  # (B,N,N,C) => (B,1,1,C)
        out = self.FC_1(out)  # (B,1,1,C) => (B, 1, 1, C/R)
        out = self.FC_2(out)  # (B, 1, 1, C/R) => (B,1,1,C)
        out = Input * out  # (B,N,N,C)*(B,1,1,C) =>(B,N,N,C)
        return out

# def CASCNN(args, mean, std):
#     (940, 8, 80)
#
#     inflow = tf.keras.Input(shape=(args.P, args.N))
#     Inflow = tf.keras.layers.Permute((2, 1))(inflow) #(None,P,N) =>(None,N,P)
#     Inflow = tf.keras.layers.Reshape((args.N, 1, args.P))(Inflow) # (None,N,P)=>(None,N,1,P)
#     outflow = tf.keras.Input(shape=(args.P, args.N))
#     Outflow = tf.keras.layers.Permute((2, 1))(outflow) #(None,P,N) =>(None,N,P)
#     Outflow = tf.keras.layers.Reshape((args.N, 1, args.P))(Outflow) # (None,N,P)=>(None,N,1,P)
#
#     matrices = tf.keras.Input(shape=(args.N, args.N))
#     Matrices = tf.keras.layers.Reshape((args.N, args.N, 1))(matrices)
#     C = int(args.CASCNN_C)
#     R = int(args.CASCNN_R)
#
#     # Real_Time
#     # (B,N,1,P)=>(B,N,1,1)
#     conv_in = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='SAME')(Inflow)
#     conv_out = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='SAME')(Outflow)
#     out = conv_in * conv_out  # (B,N,1,1) * (B,N,1,1)=>(B,N,1,1)
#     w = tf.Variable(initial_value=tf.random.normal(shape=[args.N, 1, 1]), trainable=True).numpy()
#     Flow_Output = out * w  # (B,N,1,1)*(N,1,1)=>(B,N,1,1)
#
#     # Historical Data
#     Y_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='SAME')(
#         Matrices)  # (B,N,N,C) => (B,N,N,F)
#     r_1 = channel_wise(args.N, C, R)(Y_1)  # (B,N,N,F)=>(B,N,N,F)
#     O_1 = Y_1 * r_1  # (B,N,N,F)*(B,N,N,F)=>(B,N,N,F)
#     Y_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='SAME')(
#         Matrices)  # (B,N,N,C) => (B,N,N,F)
#     r_2 = channel_wise(args.N, C, R)(Y_2)  # (B,N,N,F)=>(B,N,N,F)
#     O_2 = Y_2 * r_2  # (B,N,N,F)*(B,N,N,F)=>(B,N,N,F)
#     Output_1 = O_1 + O_2
#
#     Y_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='SAME')(
#         Output_1)  # (B,N,N,C) => (B,N,N,F)
#     r_1 = channel_wise(args.N, C, R)(Y_1)  # (B,N,N,F)=>(B,N,N,F)
#     O_1 = Y_1 * r_1  # (B,N,N,F)*(B,N,N,F)=>(B,N,N,F)
#     Y_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='SAME')(
#         Output_1)  # (B,N,N,C) => (B,N,N,F)
#     r_2 = channel_wise(args.N, C, R)(Y_2)  # (B,N,N,F)=>(B,N,N,F)
#     O_2 = Y_2 * r_2  # (B,N,N,F)*(B,N,N,F)=>(B,N,N,F)
#     Trunk_Output = O_1 + O_2
#
#     # Fusion [?,80,80,1], [?,80,8,1].
#     Output = Trunk_Output + Flow_Output  # (B,N,N,1) + (B,N,1,1) => (B, N, N, 1)
#     Output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='SAME')(
#         Output)  # (B, N, N, 1)=>(B, N, N, 1)
#     Output = Output * std + mean
#     Output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(Output)  # (B, N, N, 1)=>(B, N, N, 1)
#     Output = tf.keras.layers.Reshape((args.N, args.N))(Output)  # (B, N, N, 1)=>(B, N, N)
#     model = tf.keras.Model(inputs=[matrices, inflow, outflow], outputs=[Output])
#     model.output_shape  # 输出是 (None, N, N)
#     return model

##################################################################################################  GEML
# def GEML(args):
#     graph_geo = tf.random.normal(shape=(args.N, args.N))
#     graph_geo = np.load(Other.Obtain_Paths(args)[7])['arr_0']
#     L = np.load(Other.Obtain_Paths(args)[6])['arr_0']
#     input = tf.keras.Input(shape=(args.P, args.N, args.N, 1))
#     output = tf.keras.layers.Permute((2, 3, 1, 4))(input)# (None, P, N, N, Input_dim) => (None, N, N, P,Input_dim)
#     output = tf.keras.layers.Reshape((args.N, args.P*args.N))(output) # (None, N, N, P,Input_dim) => (None, N, N*P*Input_dim)
#     output = GCN_Layer(L, args.GCN_Units[0])(output) # (None, N, N*P*Input_dim) => (None, N, F)
#     output = GCN_Layer(L, args.GCN_Units[0])(output) # (None, N, F) => (None, N, F)
#     output = tf.keras.layers.Dense(args.N, tf.nn.relu)(output) # (None, N, F) => (None, N, N)
#     output = tf.expand_dims(output, axis=-1)  # (None, N, N) => (None, N, N,1)
#     model = tf.keras.Model(inputs=input, outputs=output)
#     model.output_shape  # 输出是 (None, N, N, Output_dim)
#     return model


def dynamic_L(inputs):
    G = inputs + tf.transpose(inputs, [0, 2, 1])  # (B, N, N)
    G_row = tf.reduce_sum(G, axis=-1)  # (B, N)
    G_row = tf.where(G_row != 0.0, G_row, 1.0)
    G = G / tf.expand_dims(G_row, axis=-1)  # (B, N, N) / (B, N, 1) => (B, N, N)
    G1 = tf.reduce_sum(G, axis=-1)
    G = G + tf.eye(G.shape[-1], dtype=tf.float32)  # (B, N, N)
    return G

# 输入 (B, N, F) (N, N)，输出 (B, N, F)
class GEML_GCN_Layer(tf.keras.layers.Layer):
    def __init__(self, L, Units, **kwargs):
        super(GEML_GCN_Layer, self).__init__()
        self.L = L
        self.Units = Units
        self.D1 = tf.keras.layers.Dense(units=self.Units, activation=tf.nn.tanh)
        self.D2 = tf.keras.layers.Dense(units=self.Units, activation=tf.nn.tanh)

    def get_config(self):
        config = {"L": self.L, "D1": self.D1, "D2": self.D2, "Units": self.Units}
        base_config = super(GEML_GCN_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, Dynamic_L):
        #print(self.L.dtype, inputs.dtype)
        G1 = tf.matmul(self.L, inputs)  # (N, N)*(B, N, F) => (B, N, F)
        G1 = self.D1(G1)  # (B, N, F)=>(B, N, F1)
        G2 = tf.matmul(Dynamic_L, inputs)  # (B, N, N)*(B, N, F) => (B, N, F)
        G2 = self.D2(G2)  # (B, N, F)=>(B, N, F1)
        G3 = tf.concat([G1, G2], axis=-1)  # (B, N, 2F1)
        return G3

def GEML(args, mean, std):
    L = np.load(args.GEML_Geo_Path)['arr_0'].astype(np.float32)
    input = tf.keras.Input(shape=(args.P, args.N, args.N)) # (B, P, N, N)
    output = tf.keras.layers.Reshape((args.N, args.N))(input) # (BP, N, N)
    Dynamic_L = dynamic_L(output) # (BP, N, N) => (BP, N, N)
    output = GEML_GCN_Layer(L, args.GEML_GCN_Units[0])(output, Dynamic_L)
    output = GEML_GCN_Layer(L, args.GEML_GCN_Units[1])(output, Dynamic_L) #(BP, N, F)
    output = tf.reshape(output, shape=(-1, args.P, args.N, output.shape[-1])) #(B,P,N,F)
    output = tf.transpose(output, perm=[0, 2, 1, 3]) # (B,P,N,F) => (B, N, P, F)
    output = tf.reshape(output, shape=(-1, args.P, output.shape[-1])) #(B,N,P,F)=>(BN, P, F)
    output = tf.keras.layers.LSTM(units=args.GEML_LSTM_Units)(output) # (BN, P, F) => (BN, F1)
    output = tf.reshape(output, shape=(-1, args.N, output.shape[-1])) #(BN, F1) => (B, N, F1)
    # OD Matrix
    OD = tf.keras.layers.Dense(args.GEML_LSTM_Units)(output) # (B, N, F1)=>(B, N, F1)
    OD_T = tf.transpose(OD, perm=[0, 2, 1]) # (B, N, F1)=>(B, F1, N)
    OD_Matrix = tf.matmul(OD, OD_T) # (B, N, F1)*(B, F1, N) =>(B, N, N)
    OD_Matrix = tf.expand_dims(OD_Matrix, axis=-1, name="od_matrix") #(B,N,N)=>(B,N,N,1)
    # Inflow
    Inflow = tf.keras.layers.Dense(1, name="inflow")(output) #(B, N, F1)=>(B, N, 1)
    # Outflow
    Outflow = tf.keras.layers.Dense(1, name="outflow")(output) # (B, N, F1)=>(B, N, 1)

    Output = OD_Matrix * std + mean
    Output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(Output) # (B, N, N, 1)=>(B, N, N, 1)
    Output = tf.keras.layers.Reshape((args.N, args.N))(Output) # (B, N, N, 1)=>(B, N, N)
    model = tf.keras.Model(inputs=input, outputs=Output)
    #model = tf.keras.Model(inputs=Input, outputs=[OD_Matrix, Inflow, Outflow])
    model.output_shape
    return model

##################################################################################################  AHGCSP
# ODt(B, N, N), OtD (B, N, N) => attention (B, N, N)
def Dynamic_Matrix(args, ODt, OtD):
    # (B, N, N) => (B, N, F)
    key_ODt = tf.keras.layers.Dense(units=args.AHGCSP_Dynamic_Units, activation = tf.nn.tanh)(ODt)
    query_ODt = tf.keras.layers.Dense(units=args.AHGCSP_Dynamic_Units, activation= tf.nn.tanh)(ODt)
    attention_ODt = tf.matmul(query_ODt, key_ODt, transpose_b=True) # (B,N,F)*(B, F,N)=>(B, N, N)
    attention_ODt = tf.nn.softmax(attention_ODt, axis = -1)

    # (B, N, N) => (B, N, F)
    key_OtD = tf.keras.layers.Dense(units=args.AHGCSP_Dynamic_Units, activation = tf.nn.tanh)(OtD)
    query_OtD = tf.keras.layers.Dense(units=args.AHGCSP_Dynamic_Units, activation= tf.nn.tanh)(OtD)
    attention_OtD = tf.matmul(query_OtD, key_OtD, transpose_b=True) # (B,N,F)*(B, F,N)=>(B, N, N)
    attention_OtD = tf.nn.softmax(attention_OtD, axis = -1)

    w_ODt = tf.Variable(initial_value=tf.random.normal(shape=(1,)), trainable=True)
    w_OtD = tf.Variable(initial_value=tf.random.normal(shape=(1,)), trainable=True)
    attention = attention_ODt * w_ODt + attention_OtD * w_OtD
    return attention

# 输入 (B, N, F) (N, N)，输出 (B, N, F)
class AHGCSP_GCN_Layer(tf.keras.layers.Layer):
    def __init__(self, Geo, KL, Units, **kwargs):
        super(AHGCSP_GCN_Layer, self).__init__()
        self.KL = KL
        self.Geo = Geo
        self.Units = Units
        self.D = tf.keras.layers.Dense(units=self.Units, activation=tf.nn.tanh)

    def get_config(self):
        config = {"KL": self.KL, "Geo": self.Geo, "D": self.D, "Units": self.Units}
        base_config = super(AHGCSP_GCN_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, Dynamic_L, W):
        print("W",W.shape)
        #Fusion = Dynamic_L+ self.Geo + self.KL  # (B, N, N)
        Fusion = Dynamic_L*W[...,0]+ self.Geo*W[...,1] + self.KL*W[...,2]  # (B, N, N)
        Fusion = tf.nn.softmax(Fusion, axis=-1)
        G1 = tf.matmul(Fusion, inputs)  # (B, N, N)*(B, N, F) => (B, N, F)
        G1 = self.D(G1)  # (B, N, F)=>(B, N, F1)
        return G1

def ST_Embedding(args, ST):

    S_Matrix = np.load(args.AHGCSP_S_Path)['arr_0'].astype(np.float32)
    D_Matrix = np.load(args.AHGCSP_D_Path)['arr_0'].astype(np.float32)
    W_Matrix = np.load(args.AHGCSP_W_Path)['arr_0'].astype(np.float32)

    or_embedding = tf.nn.embedding_lookup(S_Matrix, ST[..., 0])  # (B,P,N,N,S)
    de_embedding = tf.nn.embedding_lookup(S_Matrix, ST[..., 1])  # (B,P,N,N,S)
    od_embedding = tf.concat([or_embedding, de_embedding], axis=-1)  # (B,P,N,N,2S)
    week_embedding = tf.nn.embedding_lookup(W_Matrix, ST[..., 2])  # (B,P,N,N,W)
    day_embedding = tf.nn.embedding_lookup(D_Matrix, ST[..., 3])  # (B,P,N,N,D)
    # week_embedding = tf.one_hot(ST[..., 2], depth=7) #(B,P,N,N,W)
    # day_embedding = tf.one_hot(ST[..., 3], depth=6) #(B,P,N,N,D)
    t_embedding = tf.concat([week_embedding, day_embedding], axis=-1)  # (B,P,N,N,W+D)
    te = tf.keras.layers.Dense(units=args.AHGCSP_T_Units, activation=tf.nn.tanh)(t_embedding) #(B,P,N,N,F1)
    se = tf.keras.layers.Dense(units=args.AHGCSP_S_Units, activation=tf.nn.tanh)(od_embedding) #(B,P,N,N,F2)
    ste = tf.concat([se,te], axis=-1) #(B,P,N,N,F1+F2)
    W = tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid)(ste) #(B,P,N,N,3)
    return W

def AHGCSP(args):
    Geo = np.load(args.AHGCSP_Geo_Path)['arr_0'].astype(np.float32)
    KL = np.load(args.AHGCSP_KL_Path)['arr_0'].astype(np.float32)

    OtD = tf.keras.layers.Input(shape=(args.P, args.N, args.N))
    ODt = tf.keras.layers.Input(shape=(args.P, args.N, args.N))
    ST = tf.keras.layers.Input(shape=(args.P, args.N, args.N, 4), dtype=tf.int32) #输入的要是int32

    x_OtD = tf.reshape(OtD, shape=(-1, args.N, args.N)) # (BP, N, N)
    x_ODt = tf.reshape(ODt, shape=(-1, args.N, args.N)) # (BP, N, N)
    attention = Dynamic_Matrix(args, x_ODt, x_OtD) # (BP, N, N)

    x_ST = tf.reshape(ST, shape=(-1, args.N, args.N, 4)) # (BP, N, N, 4)
    W = ST_Embedding(args, x_ST)  # (BP,N,N,3)

    output = AHGCSP_GCN_Layer(Geo, KL, args.AHGCSP_GCN_Units)(x_ODt, attention, W) #(BP, N, F)
    output = tf.reshape(output, shape=(-1, args.P, args.N, output.shape[-1])) # (B, P, N, F)
    output = tf.transpose(output, perm=[0, 2, 1, 3])#(B, N, P, F)
    output = tf.reshape(output, shape=(-1, args.P, output.shape[-1])) #(BN, P, F)
    output = tf.keras.layers.GRU(units=args.N)(output)# (BN, N)
    output = tf.reshape(output, shape=(-1, args.N, args.N)) #(B, N, N)
    output = tf.expand_dims(output, axis=-1) #(B, N, ,N , 1)

    model = tf.keras.Model(inputs=[ODt, OtD, ST], outputs=[output])
    model.output_shape
    return model


_custom_objects = {"GCN_Layer": GCN_Layer, "channel_wise": channel_wise,
                   "GEML_GCN_Layer": GEML_GCN_Layer,
                   "AHGCSP_GCN_Layer":AHGCSP_GCN_Layer}