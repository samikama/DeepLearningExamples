# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer
from tensorflow.python.ops import init_ops
from tensorflow.python.training import training_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops



class NovoGrad(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-7,
        weight_decay = 0.0,
        exclude_from_weight_decay = None,
        grad_averaging = False,
        amsgrad = False,
        name = "NovoGrad",
        **kwargs
    ):
        super().__init__(False, name, **kwargs)
        if weight_decay < 0.0:
            raise ValueError("Weight decay rate cannot be negative")
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay

        # Tensor versions of the constructor arguments, created in _prepare().
        self.learning_rate_t = None
        self.beta_1_t = None
        self.beta_2_t = None
        self.epsilon_t = None
        self.weight_decay_t = None

        self.grad_averaging = grad_averaging
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self._zeros_slot(var, "m", self._name + "_m")
        initializer = init_ops.zeros_initializer()
        for var in var_list:
            self._get_or_make_slot_with_initializer(var=var, initializer=initializer, shape=tf.TensorShape([]), dtype=var.dtype, slot_name="v", op_name=self._name + "_v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare(self):
        learning_rate = self._call_if_callable(self.learning_rate)
        beta_1 = self._call_if_callable(self.beta_1)
        beta_2 = self._call_if_callable(self.beta_2)
        epsilon = self._call_if_callable(self.epsilon)
        weight_decay = self._call_if_callable(self.weight_decay)

        self.learning_rate_t = ops.convert_to_tensor(learning_rate, name="learning_rate")
        self.beta_1_t = ops.convert_to_tensor(beta_1, name="beta_1")
        self.beta_2_t = ops.convert_to_tensor(beta_2, name="beta_2")
        self.epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self.weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        v_t = tf.cond(tf.equal(v, 0),
                lambda: g_2,
                lambda: v * self.beta_2_t + g_2 * (1. - self.beta_2_t)
              )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            raise NotImplementedError
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon_t)

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
#            print_op = tf.print(var_name, self.weight_decay_t)
            grad = grad + self.weight_decay_t * var
#            with tf.control_dependencies([print_op]):
#                grad = tf.identity(grad)
#            grad = tf.cond(
#                tf.greater(self.weight_decay_t, 0),
#                    lambda: grad + self.weight_decay_t * var,
#                    lambda: grad
#            )

        if self.grad_averaging:
            raise NotImplementedError

        m_t = m * self.beta_1_t + grad
        m_t = m.assign(m_t, use_locking=self._use_locking)
        var_update = var - self.learning_rate_t * m_t
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True



class LambOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Layer-wise Adaptive Moments (LAMB).
    See paper [Large Batch Optimization for Deep Learning: Training BERT
    in 76 minutes](https://arxiv.org/abs/1904.00962).
    """

    def __init__(
        self,
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-6,
        weight_decay_rate = 0.0,
        exclude_from_weight_decay = None,
        exclude_from_layer_adaptation = None,
        name = "lamb",
        use_locking=False,
        **kwargs
    ):
        """Construct a new LAMB optimizer.
        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 1st moment estimates.
            beta_2: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay_rate: weight decay rate.
            exclude_from_weight_decay: List of regex patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.
            exclude_from_layer_adaptation: List of regex patterns of
              variables excluded from layer adaptation. Variables whose name
              contain a substring matching the pattern will be excluded.
            name: Optional name for the operations created when applying
              gradients. Defaults to "LAMB".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
              `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
              norm; `clipvalue` is clip gradients by value, `decay` is
              included for backward compatibility to allow time inverse
              decay of learning rate. `lr` is included for backward
              compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(use_locking, name)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay_rate = weight_decay_rate
        self.learning_rate = learning_rate

        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
        # the arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

        # Tensor versions of the constructor arguments, created in _prepare().
        self.learning_rate_t = None
        self.beta_1_t = None
        self.beta_2_t = None
        self.epsilon_t = None
        self.weight_decay_rate_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta_1_power", graph=graph),
                self._get_non_slot_variable("beta_2_power", graph=graph))


    def _create_slots(self, var_list):
        # Create the beta_1 and beta_2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self.beta_1, name="beta_1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self.beta_2, name="beta_2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self._zeros_slot(var, "m", self._name + "_m")
        for var in var_list:
            self._zeros_slot(var, "v", self._name + "_v")


    def _prepare(self):
        learning_rate = self._call_if_callable(self.learning_rate)
        beta_1 = self._call_if_callable(self.beta_1)
        beta_2 = self._call_if_callable(self.beta_2)
        epsilon = self._call_if_callable(self.epsilon)
        weight_decay_rate = self._call_if_callable(self.weight_decay_rate)

        self.learning_rate_t = ops.convert_to_tensor(learning_rate, name="learning_rate")
        self.beta_1_t = ops.convert_to_tensor(beta_1, name="beta_1")
        self.beta_2_t = ops.convert_to_tensor(beta_2, name="beta_2")
        self.epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self.weight_decay_rate_t = ops.convert_to_tensor(weight_decay_rate, name="weight_decay_rate")

    def _apply_dense(self, grad, var):
        beta_1_power, beta_2_power = self._get_beta_accumulators()
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        m_scaled_g_values = grad * (1.0 - self.beta_1_t)
        m_t = m * self.beta_1_t + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        v_scaled_g_values = (grad * grad) * (1.0 - self.beta_2_t)
        v_t = v * self.beta_2_t + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1.0 - beta_1_power)
        v_t_hat = v_t / (1.0 - beta_2_power)

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + self.epsilon_t)

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += self.weight_decay_rate_t * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

        var_update = var - ratio * self.learning_rate_t * update
        return var.assign(var_update, use_locking=self._use_locking)
      

    def _resource_apply_dense(self, grad, var):
        beta_1_power, beta_2_power = self._get_beta_accumulators()
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        m_scaled_g_values = grad * (1.0 - self.beta_1_t)
        m_t = m * self.beta_1_t + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        v_scaled_g_values = (grad * grad) * (1.0 - self.beta_2_t)
        v_t = v * self.beta_2_t + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1.0 - beta_1_power)
        v_t_hat = v_t / (1.0 - beta_2_power)

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + self.epsilon_t)

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += self.weight_decay_rate_t * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

        var_update = var - ratio * self.learning_rate_t * update
        return var.assign(var_update, use_locking=self._use_locking)


    def _resource_apply_sparse(self, grad, var, indices):
        beta_1_power, beta_2_power = self._get_beta_accumulators()
        # m_t = beta_1 * m + (1 - beta_1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1.0 - self.beta_1_t)
        m_t = m.assign(m * self.beta_1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta_2 * v + (1 - beta_2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1.0 - self.beta_2_t)
        v_t = v.assign(v * self.beta_2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1.0 - beta_1_power)
        v_t_hat = v_t / (1.0 - beta_2_power)

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + self.epsilon_t)

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += self.weight_decay_rate_t * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

        var_update = var.assign_sub(
            ratio * self.learning_rate_t * update, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t, v_t])


    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta_1_power, beta_2_power = self._get_beta_accumulators()
        with ops.colocate_with(beta_1_power):
            update_beta_1 = beta_1_power.assign(beta_1_power * self.beta_1_t, use_locking=self._use_locking)
            update_beta_2 = beta_2_power.assign(beta_2_power * self.beta_2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta_1, update_beta_2], name=name_scope)


    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for
        `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True


    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

