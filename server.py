from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__)


LATIN_ALPHABET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LATIN_ALPHABET_LOWER = LATIN_ALPHABET_UPPER.lower()
CYRILLIC_ALPHABET_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
CYRILLIC_ALPHABET_LOWER = CYRILLIC_ALPHABET_UPPER.lower()

_ALPHABETS = [
    LATIN_ALPHABET_UPPER,
    LATIN_ALPHABET_LOWER,
    CYRILLIC_ALPHABET_UPPER,
    CYRILLIC_ALPHABET_LOWER,
]

_CHAR_TO_ALPHABET: Dict[str, Tuple[str, int]] = {
    char: (alphabet, idx)
    for alphabet in _ALPHABETS
    for idx, char in enumerate(alphabet)
}

_SHIFT_MAP: Dict[str, int] = {}
for uppercase, lowercase in [
    (LATIN_ALPHABET_UPPER, LATIN_ALPHABET_LOWER),
    (CYRILLIC_ALPHABET_UPPER, CYRILLIC_ALPHABET_LOWER),
]:
    for idx, (u_char, l_char) in enumerate(zip(uppercase, lowercase)):
        _SHIFT_MAP[u_char] = idx
        _SHIFT_MAP[l_char] = idx

# ----------------------------
# Compression (Huffman Coding)
# ----------------------------


@dataclass(order=True)
class _HuffmanNode:
    frequency: int
    character: Optional[str] = field(default=None, compare=False)
    left: Optional["_HuffmanNode"] = field(default=None, compare=False)
    right: Optional["_HuffmanNode"] = field(default=None, compare=False)


def _build_huffman_tree(frequencies: Dict[str, int]) -> Optional[_HuffmanNode]:
    if not frequencies:
        return None

    # Priority queue ensures lowest frequency nodes are merged first.
    heap: List[_HuffmanNode] = [
        _HuffmanNode(freq, char) for char, freq in frequencies.items()
    ]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = _HuffmanNode(left.frequency + right.frequency, None, left, right)
        heapq.heappush(heap, merged)

    return heap[0]


def _generate_huffman_codes(node: Optional[_HuffmanNode]) -> Dict[str, str]:
    codes: Dict[str, str] = {}

    def traverse(current: _HuffmanNode, path: str) -> None:
        if current.character is not None:
            # Leaf node
            codes[current.character] = path or "0"
            return
        if current.left is not None:
            traverse(current.left, path + "0")
        if current.right is not None:
            traverse(current.right, path + "1")

    if node is not None:
        traverse(node, "")
    return codes


def _huffman_compress(message: str) -> Dict[str, object]:
    frequencies: Dict[str, int] = {}
    for char in message:
        frequencies[char] = frequencies.get(char, 0) + 1

    tree = _build_huffman_tree(frequencies)
    codes = _generate_huffman_codes(tree)

    encoded_message = "".join(codes[char] for char in message) if message else ""
    original_size = len(message.encode("utf-8")) * 8
    compressed_size = len(encoded_message)

    ratio = "N/A"
    if original_size > 0:
        ratio_value = (compressed_size / original_size) * 100 if original_size else math.nan
        ratio = f"{ratio_value:.2f}%"

    return {
        "mode": "Compression (Huffman)",
        "original_message": message,
        "analysis": {
            "char_frequencies": frequencies,
            "huffman_codes": codes,
        },
        "processed_data": {
            "encoded_message": encoded_message,
        },
        "statistics": {
            "original_size_bits": original_size,
            "compressed_size_bits": compressed_size,
            "ratio": ratio,
        },
    }


# -------------------------------------
# Error Correction (Hamming (7, 4) Code)
# -------------------------------------


def _byte_string(message: str) -> str:
    return "".join(f"{byte:08b}" for byte in message.encode("utf-8"))


def _chunk_bits(bits: str, chunk_size: int) -> List[str]:
    chunks = [bits[i : i + chunk_size] for i in range(0, len(bits), chunk_size)]
    if chunks and len(chunks[-1]) < chunk_size:
        chunks[-1] = chunks[-1].ljust(chunk_size, "0")
    return chunks


def _encode_hamming_7_4(nibble: str) -> str:
    data_bits = [int(bit) for bit in nibble]
    block = [0] * 7
    # Data bits placed at positions 3, 5, 6, 7 (1-indexed)
    block[2] = data_bits[0]
    block[4] = data_bits[1]
    block[5] = data_bits[2]
    block[6] = data_bits[3]

    # Parity calculations per Hamming(7,4)
    block[0] = block[2] ^ block[4] ^ block[6]  # p1 covers positions 1,3,5,7
    block[1] = block[2] ^ block[5] ^ block[6]  # p2 covers positions 2,3,6,7
    block[3] = block[4] ^ block[5] ^ block[6]  # p4 covers positions 4,5,6,7

    return "".join(str(bit) for bit in block)


def _decode_hamming_7_4(block: str) -> Tuple[str, Dict[str, object]]:
    bits = [int(bit) for bit in block]
    s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
    s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
    s4 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
    syndrome = s1 + (s2 << 1) + (s4 << 2)

    corrected_bits = bits[:]
    correction_details: Dict[str, object] = {
        "syndrome_binary": f"{s4}{s2}{s1}",
        "syndrome_decimal": syndrome,
        "error_detected": syndrome != 0,
        "corrected_bit_index": None,
    }

    if syndrome != 0 and 1 <= syndrome <= 7:
        index = syndrome - 1  # Convert to 0-indexed
        corrected_bits[index] ^= 1
        correction_details["corrected_bit_index"] = syndrome

    data_bits = [corrected_bits[2], corrected_bits[4], corrected_bits[5], corrected_bits[6]]
    correction_details["corrected_block"] = "".join(str(bit) for bit in corrected_bits)
    correction_details["decoded_data_bits"] = "".join(str(bit) for bit in data_bits)
    return "".join(str(bit) for bit in data_bits), correction_details


def _hamming_process(message: str) -> Dict[str, object]:
    original_bits = _byte_string(message)
    nibble_chunks = _chunk_bits(original_bits or "0000", 4)

    encoded_blocks = [_encode_hamming_7_4(nibble) for nibble in nibble_chunks]
    encoded_stream = "".join(encoded_blocks)

    stream_with_error_blocks = encoded_blocks[:]
    error_log = "No bits flipped (empty input)."
    correction_log = "No correction performed."
    correction_details: Optional[Dict[str, object]] = None

    if encoded_blocks:
        block_index = random.randrange(len(encoded_blocks))
        bit_index = random.randrange(7)
        mutated_block = list(stream_with_error_blocks[block_index])
        original_bit = mutated_block[bit_index]
        mutated_block[bit_index] = "1" if original_bit == "0" else "0"
        stream_with_error_blocks[block_index] = "".join(mutated_block)

        overall_bit_position = block_index * 7 + bit_index + 1
        error_log = (
            f"Flipped bit at overall position {overall_bit_position} "
            f"(block {block_index + 1}, bit {bit_index + 1}) "
            f"from {original_bit} to {mutated_block[bit_index]}."
        )

        # Decode and correct
        decoded_bits = []
        for idx, block in enumerate(stream_with_error_blocks):
            decoded_nibble, details = _decode_hamming_7_4(block)
            if idx == block_index:
                details["received_block"] = block
                details["original_encoded_block"] = encoded_blocks[idx]
                correction_details = details
            decoded_bits.append(decoded_nibble)

        correction_log = (
            f"Block {block_index + 1}: received {correction_details['received_block']}. "
            f"Syndrome {correction_details['syndrome_binary']} "
            f"(decimal {correction_details['syndrome_decimal']}). "
            f"Error bit index: {correction_details['corrected_bit_index']} (1-based). "
            f"Corrected block: {correction_details['corrected_block']}. "
            f"Recovered data bits: {correction_details['decoded_data_bits']}."
        )
    else:
        decoded_bits = []

    stream_with_error = "".join(stream_with_error_blocks)
    return {
        "mode": "Error Correction (Hamming)",
        "original_message": message,
        "original_bits": original_bits,
        "processed_data": {
            "encoded_hamming_stream": encoded_stream,
            "stream_with_error": stream_with_error,
        },
        "demonstration_log": {
            "error_simulation": error_log,
            "detection_and_correction": correction_log,
        },
    }


# --------------------------------
# Encryption (Vigenère Cipher)
# --------------------------------


def _shift_char(char: str, shift: int) -> str:
    mapping = _CHAR_TO_ALPHABET.get(char)
    if mapping is None:
        return char
    alphabet, idx = mapping
    new_index = (idx + shift) % len(alphabet)
    return alphabet[new_index]


def _vigenere_encrypt(message: str, key: str) -> str:
    return _vigenere_transform(message, key)


def _vigenere_decrypt(ciphertext: str, key: str) -> str:
    return _vigenere_transform(ciphertext, key, decrypt=True)


def _vigenere_transform(message: str, key: str, *, decrypt: bool = False) -> str:
    key_shifts = [_SHIFT_MAP[char] for char in key if char in _SHIFT_MAP]
    if not key_shifts:
        return message

    transformed_chars: List[str] = []
    key_index = 0

    for char in message:
        if char not in _CHAR_TO_ALPHABET:
            transformed_chars.append(char)
            continue

        shift = key_shifts[key_index % len(key_shifts)]
        if decrypt:
            shift = -shift
        transformed_chars.append(_shift_char(char, shift))
        key_index += 1

    return "".join(transformed_chars)


def _vigenere_process(message: str) -> Dict[str, object]:
    key = "KALASHNIKOV"
    encrypted = _vigenere_encrypt(message, key)
    decrypted = _vigenere_decrypt(encrypted, key)
    return {
        "mode": "Encryption (Vigenère Cipher)",
        "original_message": message,
        "processed_data": {
            "encrypted_message": encrypted,
        },
        "details": {
            "key": key,
            "note": "This can be decrypted using the same key.",
            "decrypted_check": decrypted,
        },
    }


# -------------
# Flask route
# -------------


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    mode = data.get("mode", "")

    if mode not in {"compress", "hamming", "encrypt"}:
        return jsonify({"error": "Invalid mode. Choose from 'compress', 'hamming', or 'encrypt'."}), 400

    if not isinstance(message, str):
        return jsonify({"error": "Message must be a string."}), 400

    if mode == "compress":
        response = _huffman_compress(message)
    elif mode == "hamming":
        response = _hamming_process(message)
    else:
        response = _vigenere_process(message)

    return jsonify(response)


@app.route("/", methods=["GET"])
def index():
    # Serve the bundled frontend so the app can be accessed via the Flask server.
    return send_from_directory(app.root_path, "index.html")


def main() -> None:
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
