import tensorflow.compat.v1 as tf
import horovod.tensorflow as hvd
import sys
import re

def build_assigment_map(prefix=None, skip_variables_regex=None):
    """Generate assigment map for loading checkpoints."""
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
    if not prefix:
        prefix = ''
    assignment_map = {}
    for var in all_vars:
        var_name = var.name
        if (
                var_name[-11:] in "/Momentum:0" or
                var_name[-11:] in "/Adadelta:0" or
                var_name[-13:] in "/Adadelta_1:0" or
                var_name[-7:] in "/Adam:0" or
                var_name[-9:] in "/Adam_1:0" or
                var_name[-10:] in "/Adagrad:0" or
                var_name[-10:] in "/RMSProp:0" or
                var_name[-12:] in "/RMSProp_1:0" or
                var_name[-16:] in "/LARSOptimizer:0"
        ):
            continue
        # Trim the index of the variable.
        if ':' in var_name:
            var_name = var_name[:var_name.rindex(':')]
        if skip_variables_regex and re.match(skip_variables_regex, var_name[len(prefix):]):
            continue
        assignment_map[var_name[len(prefix):]] = var
        # assignment_map[var_name] = var
    return assignment_map

def assign_from_checkpoint(model_path, var_list, ignore_missing_vars=False):
    """Creates an operation to assign specific variables from a checkpoint.
    Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of (possibly partitioned) `Variable` objects or a
      dictionary mapping names in the checkpoint to the corresponding variables
      or list of variables to initialize from that checkpoint value. For
      partitioned Variables, the name in the checkpoint must be the full
      variable, not the name of the partitioned variable, eg. "my_var" rather
      than "my_var/part_4". If empty, returns no_op(), {}.
    ignore_missing_vars: Boolean, if True ignore variables missing in the
      checkpoint with a warning instead of failing.
    Returns:
    the restore_op and the feed_dict that need to be run to restore var_list.
    Raises:
    ValueError: If `ignore_missing_vars` is False and the checkpoint specified
        at `model_path` is missing one of the variables in `var_list`.
  """
    # Normalize var_list into a dictionary mapping names in the
    # checkpoint to the list of variables to initialize from that
    # checkpoint variable. Sliced (including partitioned) variables will
    # end up under the same key.
    grouped_vars = {}
    if isinstance(var_list, (tuple, list)):
        for var in var_list:
            ckpt_name = get_variable_full_name(var)
            if ckpt_name not in grouped_vars:
                grouped_vars[ckpt_name] = []
            grouped_vars[ckpt_name].append(var)

    else:
        for ckpt_name, value in var_list.items():
            if isinstance(value, (tuple, list)):
                grouped_vars[ckpt_name] = value
            else:
                grouped_vars[ckpt_name] = [value]

    # Read each checkpoint entry. Create a placeholder variable and
    # add the (possibly sliced) data from the checkpoint to the feed_dict.
    reader = tf.train.NewCheckpointReader(model_path)
    feed_dict = {}
    assign_ops = []
    for ckpt_name in grouped_vars:
        if not reader.has_tensor(ckpt_name):
            log_str = 'Checkpoint is missing variable [%s]' % ckpt_name
            if ignore_missing_vars:
                logging.warning(log_str)
                continue
            else:
                raise ValueError(log_str)
        ckpt_value = reader.get_tensor(ckpt_name)

        for var in grouped_vars[ckpt_name]:
            placeholder_tensor = tf.placeholder(
                dtype=var.dtype.base_dtype,
                shape=var.get_shape(),
                name='placeholder/' + var.op.name
            )

            assign_ops.append(var.assign(placeholder_tensor))

            if not var._save_slice_info:
                if var.get_shape() != ckpt_value.shape:
                    raise ValueError(
                        'Total size of new array must be unchanged for %s '
                        'lh_shape: [%s], rh_shape: [%s]' %
                        (ckpt_name, str(ckpt_value.shape), str(var.get_shape())))

                feed_dict[placeholder_tensor] = ckpt_value.reshape(ckpt_value.shape)

            else:
                slice_dims = zip(var._save_slice_info.var_offset,
                                 var._save_slice_info.var_shape)

                slice_dims = [(start, start + size) for (start, size) in slice_dims]
                slice_dims = [slice(*x) for x in slice_dims]

                slice_value = ckpt_value[slice_dims]
                slice_value = slice_value.reshape(var._save_slice_info.var_shape)

                feed_dict[placeholder_tensor] = slice_value

    print_op = tf.print(
        "[GPU %02d] Restoring pretrained weights (%d Tensors) from: %s" % (
            hvd.rank(),
            len(assign_ops),
            model_path
        ),
        output_stream=sys.stdout
    )

    with tf.control_dependencies([print_op]):
        assign_op = tf.group(*assign_ops)

    return assign_op, feed_dict