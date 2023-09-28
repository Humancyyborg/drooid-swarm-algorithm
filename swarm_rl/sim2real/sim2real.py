import argparse
import json
import os
from distutils.util import strtobool
from pathlib import Path

import torch
import torch.nn as nn
from attrdict import AttrDict
from typing import List

from sample_factory.model.actor_critic import create_actor_critic

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.sim2real.code_blocks import (
    headers_network_evaluate,
    headers_evaluation,
    linear_activation,
    sigmoid_activation,
    relu_activation,
    single_drone_eval,
    multi_drone_attn_eval,
    headers_multi_agent_attention,
    attention_body
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
    parser.add_argument('--model_type', type=str, choices=['single', 'attention'],
                        help='What kind of model we are working with. '
                             'Currently only single drone models are supported.')
    args = parser.parse_args()
    return AttrDict(vars(args))


def torch_to_c_model(args):
    model_dir = Path(args.torch_model_dir)
    model = load_sf_model(model_dir, args.model_type)

    output_dir = Path(args.output_dir)
    output_path = output_dir.joinpath(args.model_type, args.output_model_name)
    output_folder = output_dir.joinpath(args.model_type)
    if args.model_type == 'single':
        generate_c_model(model, str(output_path), str(output_folder), testing=args.testing)
    else:
        generate_c_model_attention(model, str(output_path), str(output_folder), testing=args.testing)


def load_sf_model(model_dir: Path, model_type: str):
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
    args.quads_encoder_type = 'attention' if model_type == 'attention' else 'corl'
    args.quads_obstacle_scan_range = 0
    args.quads_obstacle_ray_num = 0
    args.quads_sim2real = True
    args.quads_domain_random = False
    args.quads_obst_density_random = False
    args.quads_obst_density_min = 0
    args.quads_obst_density_max = 0
    args.quads_obst_size_random = False
    args.quads_obst_size_min = 0
    args.quads_obst_size_max = 0

    # Load model
    register_swarm_components()
    # spawn a dummy env, so we can get the obs and action space info
    env = make_quadrotor_env_multi(args)
    model = create_actor_critic(args, env.observation_space, env.action_space)
    model_path = list(model_dir.glob('*.pth'))[0]
    model.load_state_dict(torch.load(model_path)['model'])

    return model


def process_layer(name: str, param: nn.Parameter, type: str):
    '''
    Convert a torch parameter from the NN into a c-equivalent represented as a string 
    '''
    if type == 'weight':
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
        return weight
    else:
        bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
        for num in param:
            bias += str(num.item()) + ','
        # get rid of comma after last number
        bias = bias[:-1]
        bias += '};\n'
        return bias


def generate_c_weights_attention(model: nn.Module, transpose: bool = False):
    """
            Generate c friendly weight strings for the c version of the attention model
            order is: self-encoder, neighbor-encoder, obst-encoder, attention, then final combined output layers 
    """
    self_weights, self_biases, self_layer_names, self_bias_names = [], [], [], []
    neighbor_weights, neighbor_biases, nbr_layer_names, nbr_bias_names = [], [], [], []
    obst_weights, obst_biases, obst_layer_names, obst_bias_names = [], [], [], []
    attn_weights, attn_biases, attn_layer_names, attn_bias_names = [], [], [], []
    out_weights, out_biases, out_layer_names, out_bias_names = [], [], [], [],
    outputs = []
    n_self, n_nbr, n_obst = 0, 0, 0
    for name, param in model.named_parameters():
        # get the self encoder weights 
        if transpose:
            param = param.T
        c_name = name.replace('.', '_')
        if 'weight' in c_name and 'critic' not in c_name and 'layer_norm' not in c_name:
            weight = process_layer(c_name, param, type='weight')
            if 'self_embed' in c_name:
                self_layer_names.append(name)
                self_weights.append(weight)
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
            elif 'neighbor_embed' in c_name:
                nbr_layer_names.append(name)
                neighbor_weights.append(weight)
                outputs.append('static float nbr_output_' + str(n_nbr) + '[' + str(param.shape[1]) + '];\n')
                n_nbr += 1
            elif 'obstacle_embed' in c_name:
                obst_layer_names.append(name)
                obst_weights.append(weight)
                outputs.append('static float obst_output_' + str(n_obst) + '[' + str(param.shape[1]) + '];\n')
                n_obst += 1
            elif 'attention' in c_name or 'layer_norm' in c_name:
                attn_layer_names.append(name)
                attn_weights.append(weight)
            else:
                # output layer
                out_layer_names.append(name)
                out_weights.append(weight)
                # these will be considered part of the self encoder
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
        if ('bias' in c_name or 'layer_norm' in c_name) and 'critic' not in c_name:
            bias = process_layer(c_name, param, type='bias')
            if 'self_embed' in c_name:
                self_bias_names.append(name)
                self_biases.append(bias)
            elif 'neighbor_embed' in c_name:
                nbr_bias_names.append(name)
                neighbor_biases.append(bias)
            elif 'obstacle_embed' in c_name:
                obst_bias_names.append(name)
                obst_biases.append(bias)
            elif 'attention' in c_name or 'layer_norm' in c_name:
                attn_bias_names.append(name)
                attn_biases.append(bias)
            else:
                # output layer
                out_bias_names.append(name)
                out_biases.append(bias)

    self_layer_names += out_layer_names
    self_bias_names += out_bias_names
    self_weights += out_weights
    self_biases += out_biases
    info = {
        'encoders': {
            'self': [self_layer_names, self_bias_names, self_weights, self_biases],
            'nbr': [nbr_layer_names, nbr_bias_names, neighbor_weights, neighbor_biases],
            'obst': [obst_layer_names, obst_bias_names, obst_weights, obst_biases],
            'attn': [attn_layer_names, attn_bias_names, attn_weights, attn_biases],
        },
        'out': [out_layer_names, out_bias_names, out_weights, out_biases],
        'outputs': outputs
    }

    return info


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
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
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

        if 'bias' in name or 'layer_norm' in name and 'critic' not in name:
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


def self_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            output_0[i] += {bias_names[0].replace('.', '_')}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
            }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                }}
    '''
    for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
                    control_n->thrust_0 = output_""" + str(n) + """[0];
                    control_n->thrust_1 = output_""" + str(n) + """[1];
                    control_n->thrust_2 = output_""" + str(n) + """[2];
                    control_n->thrust_3 = output_""" + str(n) + """[3];	
            """
        method += assignment
    # closing bracket
    method += """}\n\n"""
    return method


def self_encoder_attn_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop of self embed layer
    for_loops = []
    input_for_loop = f'''
        // Self embed layer
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            output_0[i] += {bias_names[0].replace('.', '_')}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # concat self embedding and attention embedding
    # for n in range(1, num_layers - 1):
    for_loop = f'''
        // Concat self_embed, neighbor_embed and obst_embed
        for (int i = 0; i < self_structure[0][1]; i++) {{
            output_embeds[i] = output_0[i];
            output_embeds[i + self_structure[0][1]] = attn_embeds[0][i];
            output_embeds[i + 2 * self_structure[0][1]] = attn_embeds[1][i];
        }}
    '''
    for_loops.append(for_loop)

    # forward-prop of feedforward layer
    output_for_loop = f'''
        // Feedforward layer
        for (int i = 0; i < self_structure[1][1]; i++) {{
            output_1[i] = 0;
            for (int j = 0; j < 3 * self_structure[0][1]; j++) {{
                output_1[i] += output_embeds[j] * actor_encoder_feed_forward_0_weight[j][i];
                }}
            output_1[i] += actor_encoder_feed_forward_0_bias[i];
            output_1[i] = tanhf(output_1[i]);
        }}
    '''
    for_loops.append(output_for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
        }}
        '''
    for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
        control_n->thrust_0 = output_""" + str(n) + """[0];
        control_n->thrust_1 = output_""" + str(n) + """[1];
        control_n->thrust_2 = output_""" + str(n) + """[2];
        control_n->thrust_3 = output_""" + str(n) + """[3];	
    """
        method += assignment
    # closing bracket
    method += """}\n\n"""
    return method


def neighbor_encoder_c_string(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void neighborEmbedder(const float neighbor_inputs[NEIGHBORS * NBR_DIM]) {
    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
                {prefix}_output_0[i] = 0; 
                for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                    {prefix}_output_0[i] += neighbor_inputs[j] * actor_encoder_neighbor_embed_layer_0_weight[j][i]; 
                }}
                {prefix}_output_0[i] += actor_encoder_neighbor_embed_layer_0_bias[i];
                {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
            }}
    '''
    for_loops.append(input_for_loop)

    # hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    {prefix}_output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
                }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    if n > 0:
        output_for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    neighbor_embeds[i] += output_{str(n)}[i]; 
                }}
            }}
        '''
        for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    # method closing bracket
    method += """}\n\n"""
    return method


def obstacle_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = f"""void obstacleEmbedder(const float obstacle_inputs[OBST_DIM]) {{
        //reset embeddings accumulator to zero
        memset(obstacle_embeds, 0, sizeof(obstacle_embeds));
        
    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            {prefix}_output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                {prefix}_output_0[i] += obstacle_inputs[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            {prefix}_output_0[i] += {bias_names[0].replace('.', '_')}[i];
            {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
            }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    if n > 0:
        output_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                obstacle_embeds[i] += output_{str(n)}[i];
            }}
        '''
        for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    # closing bracket
    method += """}\n\n"""
    return method


def generate_c_model_attention(model: nn.Module, output_path: str, output_folder: str, testing=False):
    info = generate_c_weights_attention(model, transpose=True)
    model_state_dict = model.state_dict()

    source = ""
    structures = ""
    methods = ""

    # setup all the encoders
    for enc_name, data in info['encoders'].items():
        # data contains [weight_names, bias_names, weights, biases]
        structure = f'static const int {enc_name}_structure [' + str(int(len(data[0]))) + '][2] = {'

        weight_names, bias_names = data[0], data[1]
        for w_name, b_name in zip(weight_names, bias_names):
            w = model_state_dict[w_name].T
            structure += '{' + str(w.shape[0]) + ', ' + str(w.shape[1]) + '},'

        # complete the structure array
        # get rid of the comma after the last curly bracket
        structure = structure[:-1]
        structure += '};\n'
        structures += structure

        method = ""
        if 'self' in enc_name:
            method = self_encoder_attn_c_str(enc_name, weight_names, bias_names)
        elif 'nbr' in enc_name:
            method = neighbor_encoder_c_string(enc_name, weight_names, bias_names)
        elif 'obst' in enc_name:
            method = obstacle_encoder_c_str(enc_name, weight_names, bias_names)
        else:
            # attention
            method = attention_body

        methods += method

    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    source += headers_multi_agent_attention

    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation

    # network eval func
    source += structures
    outputs = info['outputs']
    for output in outputs:
        source += output

    encoders = info['encoders']

    for key, vals in encoders.items():
        weights, biases = vals[-2], vals[-1]
        for w in weights:
            source += w
        for b in biases:
            source += b

    source += methods

    if testing:
        source += multi_drone_attn_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


def generate_c_model(model: nn.Module, output_path: str, output_folder: str, testing=False):
    layer_names, bias_names, weights, biases, outputs = generate_c_weights(model, transpose=True)
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    for name, param in model.named_parameters():
        param = param.T
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
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
