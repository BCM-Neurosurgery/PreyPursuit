import jax.numpy as jnp

def _p_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
        err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
        err_pos2 = x[k, :2] - inputs["prey2_pos"][k]

        err1 = jnp.vstack((err_pos1))
        err2 = jnp.vstack((err_pos2))

        return err1, err2, err_int_pos1, err_int_pos2
    
def _pv_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_vel1 = x[k, 2:] - inputs['prey1_vel'][k]

    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]
    err_vel2 = x[k, 2:] - inputs['prey2_vel'][k]

    err1 = jnp.vstack((err_pos1, err_vel1))
    err2 = jnp.vstack((err_pos2, err_vel2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pf_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]

    # calculate prediction error
    err_pred1 = x[k, :2] - (
        inputs['prey1_pos'][k] + 
        inputs['prey1_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey1_accel'][k] * inputs['dt'] ** 2
        )
    err_pred2 = x[k, :2] - (
        inputs['prey2_pos'][k] + 
        inputs['prey2_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey2_accel'][k] * inputs['dt'] ** 2
        )
    
    err1 = jnp.vstack((err_pos1, err_pred1))
    err2 = jnp.vstack((err_pos2, err_pred2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pi_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]

    # calculate error in integral of position
    err_int_pos1 = jnp.clip(err_int_pos1 + err_pos1 * inputs['dt'], -10.0, 10.0)
    err_int_pos2 = jnp.clip(err_int_pos2 + err_pos2 * inputs['dt'], -10.0, 10.0)

    err1 = jnp.vstack((err_pos1, err_int_pos1))
    err2 = jnp.vstack((err_pos2, err_int_pos2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pvi_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_vel1 = x[k, 2:] - inputs['prey1_vel'][k]

    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]
    err_vel2 = x[k, 2:] - inputs['prey2_vel'][k]

    # calculate error in integral of position
    err_int_pos1 = jnp.clip(err_int_pos1 + err_pos1 * inputs['dt'], -10.0, 10.0)
    err_int_pos2 = jnp.clip(err_int_pos2 + err_pos2 * inputs['dt'], -10.0, 10.0)

    err1 = jnp.vstack((err_pos1, err_vel1, err_int_pos1))
    err2 = jnp.vstack((err_pos2, err_vel2, err_int_pos2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pif_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]

    # calculate error in integral of position
    err_int_pos1 = jnp.clip(err_int_pos1 + err_pos1 * inputs['dt'], -10.0, 10.0)
    err_int_pos2 = jnp.clip(err_int_pos2 + err_pos2 * inputs['dt'], -10.0, 10.0)

    # calculate prediction error
    err_pred1 = x[k, :2] - (
        inputs['prey1_pos'][k] + 
        inputs['prey1_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey1_accel'][k] * inputs['dt'] ** 2
        )
    err_pred2 = x[k, :2] - (
        inputs['prey2_pos'][k] + 
        inputs['prey2_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey2_accel'][k] * inputs['dt'] ** 2
        )
    
    err1 = jnp.vstack((err_pos1, err_int_pos1, err_pred1))
    err2 = jnp.vstack((err_pos2, err_int_pos2, err_pred2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pvf_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_vel1 = x[k, 2:] - inputs['prey1_vel'][k]

    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]
    err_vel2 = x[k, 2:] - inputs['prey2_vel'][k]

    # calculate prediction error
    err_pred1 = x[k, :2] - (
        inputs['prey1_pos'][k] + 
        inputs['prey1_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey1_accel'][k] * inputs['dt'] ** 2
        )
    err_pred2 = x[k, :2] - (
        inputs['prey2_pos'][k] + 
        inputs['prey2_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey2_accel'][k] * inputs['dt'] ** 2
        )
    
    err1 = jnp.vstack((err_pos1, err_vel1, err_pred1))
    err2 = jnp.vstack((err_pos2, err_vel2, err_pred2))

    return err1, err2, err_int_pos1, err_int_pos2

def _pvif_control_err(x, k, err_int_pos1, err_int_pos2, inputs):
    err_pos1 = x[k, :2] - inputs['prey1_pos'][k]
    err_vel1 = x[k, 2:] - inputs['prey1_vel'][k]

    err_pos2 = x[k, :2] - inputs['prey2_pos'][k]
    err_vel2 = x[k, 2:] - inputs['prey2_vel'][k]

    # calculate error in integral of position
    err_int_pos1 = jnp.clip(err_int_pos1 + err_pos1 * inputs['dt'], -10.0, 10.0)
    err_int_pos2 = jnp.clip(err_int_pos2 + err_pos2 * inputs['dt'], -10.0, 10.0)

    # calculate prediction error
    err_pred1 = x[k, :2] - (
        inputs['prey1_pos'][k] + 
        inputs['prey1_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey1_accel'][k] * inputs['dt'] ** 2
        )
    err_pred2 = x[k, :2] - (
        inputs['prey2_pos'][k] + 
        inputs['prey2_vel'][k] * inputs['dt'] + 
        0.5 * inputs['prey2_accel'][k] * inputs['dt'] ** 2
        )
    
    err1 = jnp.vstack((err_pos1, err_vel1, err_int_pos1, err_pred1))
    err2 = jnp.vstack((err_pos2, err_vel2, err_int_pos2, err_pred2))

    return err1, err2, err_int_pos1, err_int_pos2

CONTROL_ERROR = {
    'p': _p_control_err,
    'pv': _pv_control_err,
    'pf': _pf_control_err,
    'pi': _pi_control_err,
    'pvi': _pvi_control_err,
    'pif': _pif_control_err,
    'pvf': _pvf_control_err,
    'pvif': _pvif_control_err,
}
