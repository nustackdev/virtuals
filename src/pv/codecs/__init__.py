"""Codec adapters.

Individual codecs can be imported from their respective modules:
    from tkv.codecs.json import JSONCodec
    from tkv.codecs.msgpack import MessagePackCodec
    from tkv.codecs.micropack import MicroPackCodec
    from tkv.codecs.pickle import PickleCodec
    from tkv.codecs.passthrough import PassthroughCodec

Composite codecs (BinaryCodec, TextCodec, NoOpCodec) are available from this module:
    from tkv.codecs import BinaryCodec, TextCodec, NoOpCodec
"""

from __future__ import annotations

from functools import partial

from tkv._key_codecs import BinaryKeyCodec, StringKeyCodec
from tkv.tkv.codec import Codec

from .json import JSONCodec
from .passthrough import PassthroughCodec
from .pickle import PickleCodec  # nosec: B403


__all__ = [
    "BinaryCodec",
    "BinaryKeyCodec",
    "NoOpCodec",
    "StringKeyCodec",
    "TextCodec",
]

# =========================================================
# Composite codec factories
# =========================================================


# MicroPack-based binary codec
BinaryCodec = partial(
    Codec,
    key_codec_cls=BinaryKeyCodec,
    value_codec_cls=PickleCodec,
)

# JSON-based text codec
TextCodec = partial(
    Codec,
    key_codec_cls=StringKeyCodec,
    value_codec_cls=JSONCodec,
)

# No-op codec
NoOpCodec = partial(
    Codec,
    key_codec_cls=BinaryKeyCodec,
    value_codec_cls=PassthroughCodec,
)
