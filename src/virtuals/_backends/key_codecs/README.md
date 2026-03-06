# Codecs for encoding/decoding storage keys with lexicographic ordering preservation

This package provides codecs for converting tuple keys (containing strings and integers)
into formats suitable for key-value storage while preserving lexicographic ordering.

Available codecs:

- PyBinaryKeyCodec: Binary encoding
- BinaryKeyCodec: Optimized Cython implementation of binary key codec
- StringKeyCodec: Human-readable string encoding (extremely unnefficient, useful for debugging)

Key features:

- Lexicographic ordering preservation
- Support for mixed string/integer tuple keys

Example usage:

```py
>>> from . import BinaryKeyCodec, StringKeyCodec
>>>
>>> # Binary codec for production use
>>> binary_codec = BinaryKeyCodec()
>>> key = ("users", 42, "profile", -10)
>>> encoded = binary_codec.encode(key)
>>> decoded = binary_codec.decode(encoded)
>>> assert decoded == key
>>>
>>> # String codec for debugging/human readability
>>> string_codec = StringKeyCodec()
>>> key = ("users", 42, "profile")
>>> encoded = string_codec.encode(key)  # Human-readable output
>>> decoded = string_codec.decode(encoded)
>>> assert decoded == key
```
