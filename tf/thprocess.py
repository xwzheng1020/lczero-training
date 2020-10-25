#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import multiprocessing as mp
import tensorflow as tf
from tfprocess import TFProcess
from chunkparser import ChunkParser
import numpy as np
import torch as th

SKIP = 32
SKIP_MULTIPLE = 1024


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_all_chunks(path):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)
    for d in glob.glob(path + '*'):
        if os.path.isdir(d):
            chunks += get_chunks(d + '/')
    return chunks


def get_latest_chunks(path, num_chunks, allow_less):
    chunks = get_all_chunks(path)
    if len(chunks) < num_chunks:
        if allow_less:
            print("sorting {} chunks...".format(len(chunks)), end='')
            chunks.sort(key=os.path.getmtime, reverse=True)
            print("[done]")
            print("{} - {}".format(os.path.basename(chunks[-1]),
                                   os.path.basename(chunks[0])))
            random.shuffle(chunks)
            return chunks
        else:
            print("Not enough chunks {}".format(len(chunks)))
            sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
                           os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


def extract_policy_bits(raw):
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 8, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(
        tf.reshape(
            tf.io.decode_raw(tf.strings.substr(raw, 7440, 832), tf.uint8),
            [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]),
                                        [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    return policy, bit_planes


def extract_byte_planes(raw):
    # 5 bytes in input are expanded and tiled
    unit_planes = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8272, 5), tf.uint8), -1),
        -1)
    unit_planes = tf.tile(unit_planes, [1, 1, 8, 8])
    return unit_planes


def extract_rule50_zero_one(raw):
    # rule50 count plane.
    rule50_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8277, 1), tf.uint8), -1),
        -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_rule50_100_zero_one(raw):
    # rule50 count plane.
    rule50_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8277, 1), tf.uint8), -1),
        -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 100.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_invariance(raw):
    # invariance plane.
    invariance_plane = tf.expand_dims(
        tf.expand_dims(
            tf.io.decode_raw(tf.strings.substr(raw, 8278, 1), tf.uint8), -1),
        -1)
    return tf.cast(tf.tile(invariance_plane, [1, 1, 8, 8]), tf.float32)


def extract_outputs(raw):
    # winner is stored in one signed byte and needs to be converted to one hot.
    winner = tf.cast(
        tf.io.decode_raw(tf.strings.substr(raw, 8279, 1), tf.int8), tf.float32)
    winner = tf.tile(winner, [1, 3])
    z = tf.cast(tf.equal(winner, [1., 0., -1.]), tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8284, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8292, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    ply_count = tf.io.decode_raw(tf.strings.substr(raw, 8304, 4), tf.float32)
    return z, q, ply_count


def extract_inputs_outputs_if1(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.ones_like(input_format))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = tf.cast(extract_byte_planes(raw), tf.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def extract_unit_planes_with_bitsplat(raw):
    unit_planes = extract_byte_planes(raw)
    bitsplat_unit_planes = tf.bitwise.bitwise_and(
        unit_planes, [1, 2, 4, 8, 16, 32, 64, 128])
    bitsplat_unit_planes = tf.minimum(
        1., tf.cast(bitsplat_unit_planes, tf.float32))
    unit_planes = tf.cast(unit_planes, tf.float32)
    return unit_planes, bitsplat_unit_planes


def make_frc_castling(bitsplat_unit_planes, zero_plane):
    queenside = tf.concat([
        bitsplat_unit_planes[:, :1, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 2:3, :1]
    ], 2)
    kingside = tf.concat([
        bitsplat_unit_planes[:, 1:2, :1], zero_plane[:, :, :6],
        bitsplat_unit_planes[:, 3:4, :1]
    ], 2)
    return queenside, kingside


def extract_inputs_outputs_if2(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 2))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 frc castling and 1 stm.
    # In order to do frc we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    # Although we only need bit unpacked for first 4 of 5 planes, its simpler just to create them all.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    # For FRC the old unit planes must be replaced with 0 and 2 merged, 1 and 3 merged, two zero planes and then original 4.
    queenside, kingside = make_frc_castling(bitsplat_unit_planes, zero_plane)
    unit_planes = tf.concat(
        [queenside, kingside, zero_plane, zero_plane, unit_planes[:, 4:]], 1)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def make_canonical_unit_planes(bitsplat_unit_planes, zero_plane):
    # For canonical the old unit planes must be replaced with 0 and 2 merged, 1 and 3 merged, two zero planes and then en-passant.
    queenside, kingside = make_frc_castling(bitsplat_unit_planes, zero_plane)
    enpassant = tf.concat(
        [zero_plane[:, :, :7], bitsplat_unit_planes[:, 4:, :1]], 2)
    unit_planes = tf.concat(
        [queenside, kingside, zero_plane, zero_plane, enpassant], 1)
    return unit_planes


def extract_inputs_outputs_if3(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def extract_inputs_outputs_if4(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_100_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1),
        [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def make_armageddon_stm(invariance_plane):
    # invariance_plane contains values of 128 or higher if its black side to move, 127 or lower otherwise.
    # Convert this to 0,1 by subtracting off 127 and then clipping.
    return tf.clip_by_value(invariance_plane - 127., 0., 1.)


def extract_inputs_outputs_if132(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    #input_format = tf.reshape(
    #    tf.io.decode_raw(tf.strings.substr(raw, 4, 4), tf.int32),
    #    [-1, 1, 1, 1])
    # tf.debugging.assert_equal(input_format, tf.multiply(tf.ones_like(input_format), 3))

    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 inputs are 4 castling and 1 enpassant.
    # In order to do the frc castling and if3 enpassant plane we need to make bit unpacked versions.  Note little endian for these fields so the bitwise_and array is reversed.
    unit_planes, bitsplat_unit_planes = extract_unit_planes_with_bitsplat(raw)

    rule50_plane, zero_plane, one_plane = extract_rule50_100_zero_one(raw)

    unit_planes = make_canonical_unit_planes(bitsplat_unit_planes, zero_plane)

    armageddon_stm = make_armageddon_stm(extract_invariance(raw))

    inputs = tf.reshape(
        tf.concat(
            [bit_planes, unit_planes, rule50_plane, armageddon_stm, one_plane],
            1), [-1, 112, 64])

    z, q, ply_count = extract_outputs(raw)

    return (inputs, policy, z, q, ply_count)


def select_extractor(mode):
    if mode == 1:
        return extract_inputs_outputs_if1
    if mode == 2:
        return extract_inputs_outputs_if2
    if mode == 3:
        return extract_inputs_outputs_if3
    if mode == 4:
        return extract_inputs_outputs_if4
    if mode == 132:
        return extract_inputs_outputs_if132
    assert (False)


def semi_sample(x):
    return tf.slice(tf.random.shuffle(x), [0], [SKIP_MULTIPLE])

class ResidualBlock(th.nn.Module):
    def __init__(self, filter=128, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv1 = th.nn.Conv2d(filter, filter, kernel_size, padding=padding, bias=bias)
        self.bn1 = th.nn.BatchNorm2d(filter)
        self.relu = th.nn.ReLU()
        self.conv2 = th.nn.Conv2d(filter, filter, kernel_size, padding=padding, bias=bias)
        self.bn2 = th.nn.BatchNorm2d(filter)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    allow_less = cfg['dataset'].get('allow_less_chunks', False)
    train_ratio = cfg['dataset']['train_ratio']
    experimental_parser = cfg['dataset'].get('experimental_v5_only_dataset',
                                             False)
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    if 'input_test' in cfg['dataset']:
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'],
                                         num_train, allow_less)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test,
                                        allow_less)
    else:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks,
                                   allow_less)
        if allow_less:
            num_train = int(len(chunks) * train_ratio)
            num_test = len(chunks) - num_train
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    ChunkParser.BATCH_SIZE = split_batch_size

    tfprocess = TFProcess(cfg)

    train_parser = ChunkParser(train_chunks,
                              tfprocess.INPUT_MODE,
                              shuffle_size=10000,
                              sample=SKIP,
                              batch_size=ChunkParser.BATCH_SIZE,
                              workers=4)
    batch_gen = train_parser.parse()
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    model = th.nn.Sequential(
        th.nn.Conv2d(112, 128, 3, padding=1, bias=False), 
        th.nn.ReLU(), 
        ResidualBlock(), 
        ResidualBlock(), 
        ResidualBlock(), 
        ResidualBlock(), 
        ResidualBlock(), 
        ResidualBlock(), 
        th.nn.Conv2d(128, 32, 1, bias=False), 
        th.nn.ReLU(), 
        th.nn.Flatten(), 
        th.nn.Linear(2048, 1858, bias=False)
    )
    model.train()
    model = model.to(device)
    optimizer = th.optim.SGD(model.parameters(), lr=0.01)
    train_batches = 100000
    for i in range(train_batches):
        # print(f'getting data {i} ...')
        x, y, z, q, m = next(batch_gen)
        x, y, z, q, m = ChunkParser.parse_function(x, y, z, q, m)
        x = x.numpy()
        x = th.Tensor(x)  
        x = x.reshape((-1, 112, 8, 8))
        y = th.Tensor(y.numpy())
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        policy = model(x)
        loss = th.mean(th.sum(-th.log_softmax(policy, 1) * th.nn.functional.relu(y), 1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'step = {i}, loss = {loss.item()}')

    train_parser.shutdown()


if __name__ == "__main__":
    # target0 = np.random.random((2048, 1858))
    # output0 = np.random.random((2048, 1858))
    # target = tf.convert_to_tensor(target0)
    # output = tf.convert_to_tensor(output0)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(target, output))
    # target2 = th.from_numpy(target0)
    # output2 = th.from_numpy(output0)
    # loss2 = th.mean(th.sum(-th.log_softmax(output2, 1) * target2, 1))
    # print(loss)
    # print(loss2)


    conf = '/root/lczero-training/tf/configs/example.yaml'
    class A():
        pass
    cmd = A()
    cmd.cfg = open(conf)
    cmd.output = '/tmp/test_tmp.txt'
    main(cmd)
    cmd.cfg.close()
