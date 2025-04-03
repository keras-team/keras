# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.flexbuffers - Code for parsing flexbuffers
"""

import struct


class FlexbufferParseException(Exception):
    pass


def read_int(buffer, offset, bit_size):
    size = 1 << bit_size
    format_char = 'bhiq'[bit_size]
    return struct.unpack('<' + format_char, buffer[offset:offset+size])[0]


def read_uint(buffer, offset, bit_size):
    size = 1 << bit_size
    format_char = 'BHIQ'[bit_size]
    return struct.unpack('<' + format_char, buffer[offset:offset+size])[0]


def read_float(buffer, offset, bit_size):
    if bit_size == 2:
        return struct.unpack('<f', buffer[offset:offset+4])[0]
    if bit_size == 3:
        return struct.unpack('<d', buffer[offset:offset+8])[0]
    raise FlexbufferParseException("Invalid bit size for flexbuffer float: %d" % bit_size)


def read_string(buffer, offset, size, decode_strings):
    data = buffer[offset:offset+size]
    if decode_strings:
        # Flexbuffer requires all strings to be valid UTF-8 but FlexOps don't always respect this.
        data = data.decode('utf-8')
    return data


def read_indirect(buffer, offset, bit_size):
    return offset - read_uint(buffer, offset, bit_size)


def read_bytes(buffer, offset, size):
    return buffer[offset:offset+size]


def read_array(buffer, offset, length, bit_size, packed_type, decode_strings):
    byte_size = 1 << bit_size
    arr = []
    for i in range(length):
        item_offset = offset + (i * byte_size)
        arr.append(read_buffer(buffer, item_offset, bit_size, packed_type, decode_strings))
    return arr


def read_buffer(buffer, offset, parent_bit_size, packed_type, decode_strings):
    """Recursively decode flatbuffer object into python representation"""
    bit_size = packed_type & 3
    value_type = packed_type >> 2
    byte_size = 1 << bit_size

    if value_type == 0x0:
        return None
    if value_type in [0x1, 0x2, 0x3]:
        read_fn = {0x1: read_int, 0x2: read_uint, 0x3: read_float}[value_type]
        return read_fn(buffer, offset, parent_bit_size)
    if value_type == 0x4:
        str_offset = read_indirect(buffer, offset, parent_bit_size)
        size = 0
        while read_int(buffer, str_offset + size, 0) != 0:
            size += 1
        return read_string(buffer, str_offset, size, decode_strings)
    if value_type == 0x5:
        str_offset = read_indirect(buffer, offset, parent_bit_size)
        size_bit_size = bit_size
        size_byte_size = 1 << size_bit_size
        size = read_uint(buffer, str_offset - size_byte_size, bit_size)
        while read_int(buffer, str_offset + size, 0) != 0:
            size_byte_size <<= 1
            size_bit_size += 1
            size = read_uint(buffer, str_offset - size_byte_size, size_bit_size)
        return read_string(buffer, str_offset, size, decode_strings)
    if value_type in [0x6, 0x7, 0x8]:
        read_fn = {0x6: read_int, 0x7: read_uint, 0x8: read_float}[value_type]
        data_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_fn(buffer, data_offset, bit_size)
    if value_type == 0x9:
        length = read_uint(buffer, read_indirect(buffer, offset, parent_bit_size) - byte_size, bit_size)
        keys_offset = read_indirect(buffer, offset, parent_bit_size) - (byte_size * 3)
        keys_vector_offset = read_indirect(buffer, keys_offset, bit_size)
        key_byte_size = read_uint(buffer, keys_offset + byte_size, bit_size)
        key_bit_size = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[key_byte_size]
        values_offset = read_indirect(buffer, offset, parent_bit_size)
        packed_types_offset = values_offset + length * byte_size
        obj = {}
        for i in range(length):
            key_offset = keys_vector_offset + i * key_byte_size
            key = read_buffer(buffer, key_offset, key_bit_size, (0x4 << 2) | key_bit_size, decode_strings)
            value_offset = values_offset + i * byte_size
            value_packed_type = read_uint(buffer, packed_types_offset + i, 0)
            value = read_buffer(buffer, value_offset, bit_size, value_packed_type, decode_strings)
            obj[key] = value
        return obj
    if value_type == 0xa:
        length = read_uint(buffer, read_indirect(buffer, offset, parent_bit_size) - byte_size, bit_size)
        arr = []
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        packed_types_offset = items_offset + (length * byte_size)
        for i in range(length):
            item_offset = items_offset + (i * byte_size)
            packed_type = read_uint(buffer, packed_types_offset + i, 0)
            arr.append(read_buffer(buffer, item_offset, bit_size, packed_type, decode_strings))
        return arr
    if value_type in [0xb, 0xc, 0xd, 0xe, 0xf, 0x24]:
        length_offset = read_indirect(buffer, offset, parent_bit_size) - byte_size
        length = read_uint(buffer, length_offset, bit_size)
        item_value_type = value_type - 0xb + 0x1
        packed_type = item_value_type << 2
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_array(buffer, items_offset, length, bit_size, packed_type, decode_strings)
    if 0x10 <= value_type <= 0x18:
        length = (value_type - 0x10) // 3 + 2
        value_type = ((value_type - 0x10) % 3) + 1
        packed_type = value_type << 2
        items_offset = read_indirect(buffer, offset, parent_bit_size)
        return read_array(buffer, items_offset, length, bit_size, packed_type, decode_strings)
    if value_type == 0x19:
        data_offset = read_indirect(buffer, offset, parent_bit_size)
        size_offset = data_offset - byte_size
        size = read_uint(buffer, size_offset, bit_size)
        return read_bytes(buffer, data_offset, size)
    if value_type == 0x1a:
        return read_uint(buffer, offset, parent_bit_size) > 0
    raise FlexbufferParseException("Invalid flexbuffer value type %r" % value_type)


def read_flexbuffer(buffer, decode_strings=True):
    byte_size = read_uint(buffer, len(buffer) - 1, 0)
    bit_size = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[byte_size]
    packed_type = read_uint(buffer, len(buffer) - 2, 0)
    offset = len(buffer) - 2 - byte_size
    return read_buffer(buffer, offset, bit_size, packed_type, decode_strings)
