import numpy as np
import tensorflow as tf



def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return loss #tf.reduce_mean(loss)


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)

    return loss #tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss

def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse



##################### ADDITIONAL ########################

def distance_to_mean_loss_vector(labels, mean_value, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    dtm_loss = abs(labels - mean_value)
    dtm_loss = dtm_loss*mask
    dtm_loss = tf.where(tf.is_nan(dtm_loss), tf.zeros_like(dtm_loss), dtm_loss)
    dtm_loss /= tf.reduce_max(labels)
    return dtm_loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels, feat_mean_values_np ,use_dtm=True):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)

        dtms = distance_to_mean_loss_vector(labels, feat_mean_values_np, null_val)
        dtm_weights = tf.convert_to_tensor(dtms, dtype='float32')

        # flag for dtm usage
        if (use_dtm):
            # multiply with weights
            shift = 1
            dtm = tf.multiply(dtm_weights + shift, mae)
            return tf.reduce_mean(dtm)

        # return default loss
        return tf.reduce_mean(mae)

    return loss

def masked_mse_loss(scaler, null_val):
    def loss(preds, labels, feat_mean_values_np ,use_dtm=True):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mse = masked_mse_tf(preds=preds, labels=labels, null_val=null_val)

        dtms = distance_to_mean_loss_vector(labels, feat_mean_values_np)
        dtm_weights = tf.convert_to_tensor(dtms, dtype='float32')

        # flag for dtm usage
        if (use_dtm):
            # multiply with weights
            shift = 1
            dtm = tf.multiply(dtm_weights + shift, mse)
            return tf.reduce_mean(dtm)

        # return default loss
        return tf.reduce_mean(mse)

    return loss

def masked_peak_mae(preds, labels, null_val, dtm_threshold): 
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        mae = np.multiply(dtm_threshold,mae)
        return np.mean(mae)


def masked_peak_mape(preds, labels, null_val, dtm_threshold): 
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        mape = np.multiply(dtm_threshold, mape)
        return np.mean(mape)

def masked_peak_mse_np(preds, labels, null_val, dtm_threshold):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.multiply(dtm_threshold,rmse)
        return np.mean(rmse)


def masked_peak_rmse(preds, labels, null_val, dtm_threshold): 
    return np.sqrt(masked_peak_mse_np(preds=preds, labels=labels, null_val=null_val, dtm_threshold=dtm_threshold))


# Tf error metrics
def masked_peak_mae_tf(preds, labels, null_val, dtm_above_thresh):
    # Null val check
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    # Create the mask
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    #loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.math.multiply(dtm_above_thresh, loss)

    return tf.reduce_mean(loss)


def masked_peak_mape_tf(preds, labels, null_val, dtm_above_thresh):
    if np.isnan(null_val):
        mask = ~tf.isnan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mape = abs(tf.math.divide(tf.cast(tf.subtract(preds, labels), 'float32'), labels))
    mape = tf.where(tf.is_nan(mape), tf.zeros_like(mape), mape)
    mape = tf.math.multiply(dtm_above_thresh, mape)
    return tf.reduce_mean(mape)

def masked_peak_rmse_tf(preds, labels, null_val, dtm_above_thresh):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    #loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.math.multiply(dtm_above_thresh, loss)
    return tf.sqrt(tf.reduce_mean(loss))



# For validation time peak metrics calculation on GPU.
def calculate_peak_metrics_tf(pred, label, null_val, feat_mean_values, max_value, DTM_TH):
    # DTM Phase Previous Approach
    #dtm_current = tf.math.abs(label - feat_mean_values)
    #print(dtm_current.dtype)
    # either tf.math.divide or tf.math.divide_no_nan(np.nan, 0.0)
    #dtm_current = tf.math.divide(dtm_current, max_value)
    #print(dtm_current.dtype)
    #dtm_above_thresh = tf.math.greater_equal(dtm_current, DTM_TH) 
    #print(dtm_above_thresh.dtype)
    #dtm_above_thresh = tf.cast(dtm_above_thresh, tf.float32)
    
    dtms = abs(label - feat_mean_values)
    dtms = tf.convert_to_tensor(dtms, dtype='float32')
    dtms /= max_value
    dtm_above_thresh = tf.math.greater_equal(dtms, DTM_TH)
    dtm_above_thresh = tf.cast(dtm_above_thresh, tf.float32)

    # Error phase
    pmae = masked_peak_mae_tf(pred, label, null_val, dtm_above_thresh)
    pmape = masked_peak_mape_tf(pred, label, null_val, dtm_above_thresh)
    prmse = masked_peak_rmse_tf(pred, label, null_val, dtm_above_thresh)
    return pmae, pmape, prmse


# Works fine with testing stage.
def calculate_peak_metrics(pred, label, null_val, feat_mean_values_np, max_value, DTM_TH): 
    # DTM Phase
    dtm_current = np.abs(label - feat_mean_values_np)
    dtm_current = np.divide(dtm_current, max_value)
    dtm_above_thresh = np.greater_equal(dtm_current, DTM_TH) * 1
    # Error phase
    pmae = masked_peak_mae(pred, label, null_val, dtm_above_thresh)
    pmape = masked_peak_mape(pred, label, null_val, dtm_above_thresh)
    prmse = masked_peak_rmse(pred, label, null_val, dtm_above_thresh)
    return pmae, pmape, prmse

