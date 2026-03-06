# virtuals-binary-codec

Cython-optimized binary key codec for [Virtuals](https://github.com/everyabc/virtuals).

Encodes tuple keys into binary format while preserving lexicographic ordering. Supports mixed integer and string components.

## Install

```
pip install virtuals-binary-codec
```

Or as part of Virtuals:

```
pip install virtuals-py[binary]
```

## Usage

```python
from virtuals_binary_codec import BinaryKeyCodec

codec = BinaryKeyCodec()
encoded = codec.encode(("users", 42, "profile"))
decoded = codec.decode(encoded)
assert decoded == ("users", 42, "profile")
```

Virtuals automatically uses this codec when installed. Without it, Virtuals falls back to a pure Python implementation.
