import argparse
import json
import os
from distutils.util import strtobool
from pathlib import Path

import torch
import torch.nn as nn
from attrdict import AttrDict
from sample_factory.model.actor_critic import create_actor_critic

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.sim2real.code_blocks import (
    headers_network_evaluate,
    headers_evaluation,
    linear_activation,
    sigmoid_activation,
    relu_activation,
    single_drone_eval
)
from swarm_rl.train import register_swarm_components


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_model_dir', type=str, default='swarm_rl/sim2real/torch_models/single',
                        help='Path where the policy and cfg is stored')
    parser.add_argument('--output_dir', type=str, default='swarm_rl/sim2real/c_models',
                        help='Where you want the c model to be saved')
    parser.add_argument('--output_model_name', type=str, default='model.c')
    parser.add_argument('--testing', type=lambda x: bool(strtobool(x)), default=False,
                        help='Whether or not to save the c model in testing mode. Enable this if you want to run the '
                             'unit test to make sure the output of the c model is the same as the pytorch model. Set '
                             'to False if you want to output a c model that will be actually used for sim2real')
    parser.add_argument('--model_type', type=str, choices=['single'], default='single',
                        help='What kind of model we are working with. '
                             'Currently only single drone models are supported.')
    args = parser.parse_args()
    return AttrDict(vars(args))


def torch_to_c_model(args):
    model_dir = Path(args.torch_model_dir)
    model = load_sf_model(model_dir)

    output_dir = Path(args.output_dir)
    output_path = output_dir.joinpath(args.model_type, args.output_model_name)
    output_folder = output_dir.joinpath(args.model_type)
    generate_c_model(model, str(output_path), str(output_folder), testing=args.testing)


def load_sf_model(model_dir: Path):
    """
        Load a trained SF pytorch model
    """
    assert model_dir.exists(), f'Path {str(model_dir)} is not a valid path'
    # Load hyper-parameters
    cfg_path = model_dir.joinpath('config.json')
    with open(cfg_path, 'r') as f:
        args = json.load(f)
    args = AttrDict(args)

    # Manually set some values
    args.visualize_v_value = False
    args.quads_encoder_type = 'corl'

    # Load model
    register_swarm_components()
    # spawn a dummy env, so we can get the obs and action space info
    env = make_quadrotor_env_multi(args)
    model = create_actor_critic(args, env.observation_space, env.action_space)
    model_path = list(model_dir.glob('*.pth'))[0]
    model.load_state_dict(torch.load(model_path)['model'])

    return model


def generate_c_weights(model: nn.Module, transpose: bool = False):
    """
        Generate c friendly weight strings for the c version of the model
    """
    weights, biases = [], []
    layer_names, bias_names, outputs = [], [], []
    n_bias = 0
    for name, param in model.named_parameters():
        if transpose:
            param = param.T
        name = name.replace('.', '_')
        if 'weight' in name and 'critic' not in name:
            layer_names.append(name)
            weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
            for row in param:
                weight += '{'
                for num in row:
                    weight += str(num.item()) + ','
                # get rid of comma after the last number
                weight = weight[:-1]
                weight += '},'
            # get rid of comma after the last curly bracket
            weight = weight[:-1]
            weight += '};\n'
            weights.append(weight)

        if 'bias' in name and 'critic' not in name:
            bias_names.append(name)
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            biases.append(bias)
            output = 'static float output_' + str(n_bias) + '[' + str(param.shape[0]) + '];\n'
            outputs.append(output)
            n_bias += 1

    return layer_names, bias_names, weights, biases, outputs


def generate_c_model(model: nn.Module, output_path: str, output_folder: str, testing=False):
    layer_names, bias_names, weights, biases, outputs = generate_c_weights(model, transpose=True)
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    for name, param in model.named_parameters():
        param = param.T
        if 'weight' in name and 'critic' not in name:
            structure += '{' + str(param.shape[0]) + ', ' + str(param.shape[1]) + '},'

    # complete the structure array
    # get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += '};\n'

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {layer_names[0]}[j][i];
            }}
            output_0[i] += {bias_names[0]}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
        for (int i = 0; i < structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n]}[i];
            output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
        }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
                for (int i = 0; i < structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n]}[i];
                }}
    '''
    for_loops.append(output_for_loop)

    # assign network outputs to control
    assignment = """
            control_n->thrust_0 = output_""" + str(n) + """[0];
            control_n->thrust_1 = output_""" + str(n) + """[1];
            control_n->thrust_2 = output_""" + str(n) + """[2];
            control_n->thrust_3 = output_""" + str(n) + """[3];	
    """

    # construct the network evaluate function
    controller_eval = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    for code in for_loops:
        controller_eval += code
    # assignment to control_n
    controller_eval += assignment

    # closing bracket
    controller_eval += """}"""

    # combine all the codes
    source = ""
    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation
    # network eval func
    source += structure
    for output in outputs:
        source += output
    for weight in weights:
        source += weight
    for bias in biases:
        source += bias
    source += controller_eval

    if testing:
        source += single_drone_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


if __name__ == '__main__':
    # example use case
    # cfg = AttrDict({
    #     'torch_model_dir': 'swarm_rl/sim2real/torch_models/single',
    #     'output_dir': 'swarm_rl/sim2real/c_models',
    #     'output_model_name': 'model.c',
    #     'testing': True,
    #     'model_type': 'single',
    # })
    # torch_to_c_model(cfg)

    cfg = parse_args()
    torch_to_c_model(args=cfg)
