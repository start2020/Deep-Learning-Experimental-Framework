#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
from libs import Models, utils
import math
import datetime

# Metric Function
# Note that if preds has float('NaN'), it can't be masked. if there is 'NaN', the result will be zero

def MAE(labels, preds):
    Diff = tf.abs(preds - labels)
    loss = tf.reduce_mean(Diff)
    return loss.numpy()

def WMAPE(labels, preds):
    Diff = tf.abs(preds - labels)
    loss = tf.reduce_sum(Diff) / tf.reduce_sum(labels)
    return loss.numpy()

def MSE(labels, preds):
    loss = tf.square(tf.subtract(preds, labels))
    loss = tf.reduce_mean(loss)
    return loss.numpy()

def RMSE(labels, preds):
    mse = MSE(labels, preds)
    loss = math.sqrt(mse)
    return loss

def SMAPE(labels, preds, c=1.0):
    Diff = tf.abs(preds - labels)
    Sum = (preds + labels)*0.5 + c
    loss = Diff / Sum
    loss = tf.reduce_mean(loss)
    return loss.numpy()

# Write the Metrics
def write_metrics(args, outputs, preds, log):
    d = {}
    d['MAE'] = MAE(outputs, preds)
    d['RMSE'] = RMSE(outputs, preds)
    d['WMAPE'] = WMAPE(outputs, preds)
    d['SMAPE'] = SMAPE(outputs, preds)

    File_Name = args.Metrics_Save_Path
    S = ""
    for key, value in d.items():
        e = str(key) + ":" + str(round(value, 4)) + " "
        S += "{:<13} ".format(e)
    S += "{:<13} ".format(args.Time)
    S += "{:<13} ".format(args.hyper_para)
    S += "{:<13}".format(log)
    S += '\n'
    f = open(File_Name, mode = 'a')
    f.writelines(S)
    print(S)
    f.close()

def data_transform(args, sign=0):
    '''
    :param args: sign=0, train; sign=1,test
    Data = [OtD_b, ODt, OtD_d_W_P, Label, Label_W, Label_M, ST, Inflow, Delayed_Inflow, Outflow]
    :return:
    '''
    index = "arr_{}".format(sign)
    outputs = np.load(args.Save_Path+"Label.npz")[index].astype(np.float32)
    inputs = np.load(args.Save_Path+"OtD_b.npz")[index].astype(np.float32)

    if args.Model == "GEML":
        y_inflow = np.sum(outputs, axis=2) #(M,N,1)
        y_outflow = np.sum(outputs, axis=1) #(M,1,N)
        outputs = [y_inflow, y_outflow,outputs]

    if args.Model == "CASCNN":
        Label_M = np.load(args.Save_Path+"Label_M.npz")[index].astype(np.float32)
        Inflow = np.load(args.Save_Path+"Inflow.npz")[index].astype(np.float32)
        Outflow = np.load(args.Save_Path+"Outflow.npz")[index].astype(np.float32)
        inputs = [Label_M, Inflow, Outflow]

    if args.Model == "AHGCSP":
        ODt = np.load(args.Save_Path+"ODt.npz")[index].astype(np.float32)
        OtD_d_W_P = np.load(args.Save_Path+"OtD_d_W_P.npz")[index].astype(np.float32)
        Label_W = np.load(args.Save_Path+"Label_W.npz")[index].astype(np.float32)
        ST = np.load(args.Save_Path+"ST.npz")[index]
        Delayed_Inflow = np.load(args.Save_Path+"Delayed_Inflow.npz")[index].astype(np.float32)
        inputs = [inputs, ODt, OtD_d_W_P, Label_W, ST, Inflow, Delayed_Inflow]

    return inputs, outputs

def Loss_Observation(args, i, H, start_time):
    train_loss = H.history['loss']
    val_loss = H.history['val_loss']
    iterations = [i for i in range(len(train_loss))]
    plt.plot(iterations, train_loss, 'b-', label='Train_Loss')
    plt.plot(iterations, val_loss, 'r-', label='Val_Loss')
    plt.title('Train_Loss VS Val_Loss')

    if i == 0:
        plt.legend()
    if i == args.Repeat-1:
        end_time = datetime.datetime.now()
        Total_time = str(end_time - start_time).split(".")[0].split(":")
        Path = args.Loss_Save_Path + "_" + Total_time[0] + Total_time[1] + '.png'
        plt.savefig(Path)
        print("figure save!")


def predict(args):
    start_time = datetime.datetime.now()
    inputs, outputs = data_transform(args, sign=1) # Test
    end_time = datetime.datetime.now()
    print("Test_Loading Time:{}".format(end_time-start_time))

    if args.Model in ["HA"]:
        preds_1 = np.mean(inputs, axis=1)
        mean = np.load(args.Mean_Std_Path)['mean']
        std = np.load(args.Mean_Std_Path)['std']
        preds = preds_1 * std + mean
        log = ""
    else:
        # Load the best Model
        model = tf.keras.models.load_model(args.Model_Save_Path, custom_objects=Models._custom_objects)
        preds = model.predict(x=inputs)
        log = utils.print_parameters(model)

    # Write the Metrics
    if args.Model=="GEML":
        preds = preds[0]
        outputs = outputs[0]

    write_metrics(args, outputs, preds, log)
    end_time_1 = datetime.datetime.now()
    print("Total Test Time:{}".format(end_time_1-start_time))

def train(args,i):
    start_time = datetime.datetime.now()
    inputs, outputs = data_transform(args, sign=0) # Train dataset
    end_time_1 = datetime.datetime.now()
    print("Train_Loading Time:{}".format(end_time_1-start_time))

    # Model Construction
    mean = np.load(args.Mean_Std_Path)['mean']
    std = np.load(args.Mean_Std_Path)['std']
    print("mean:{},std:{}".format(mean, std))

    model = Models.Choose_Model(args, mean, std)

    if args.Model == "GEML":
        Loss = {"tf_op_layer_od_matrix":"mse","inflow":"mse","outflow":"mse"}
        Loss_Weights = {"tf_op_layer_od_matrix":0.5, "inflow":0.25, "outflow":0.25}
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.LR, beta_1=0.9, beta_2=0.999), loss=Loss, loss_weights = Loss_Weights,  metrics=['mae'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.LR), loss='mse', metrics=['mae'])
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=args.Total_Patience),
        tf.keras.callbacks.ModelCheckpoint(filepath=args.Model_Save_Path,
                                        monitor='val_loss',
                                        save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=args.Decay_Ratio,
            patience=args.Decay_Patience)
                     ]
    # Train
    H = model.fit(inputs, outputs, callbacks=callbacks_list, batch_size=args.Batch, epochs=args.Epochs, validation_split=0.2)
    Loss_Observation(args, i, H, start_time)
    end_time_2 = datetime.datetime.now()
    print("Total Train Time:{}".format(end_time_2-end_time_1))