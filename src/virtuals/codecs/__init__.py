"""Codec adapters.

Individual codecs can be imported from their respective modules:
    from virtuals.codecs.json import JSONCodec
    from virtuals.codecs.msgpack import MessagePackCodec
    from virtuals.codecs.micropack import MicroPackCodec
    from virtuals.codecs.pickle import PickleCodec
    from virtuals.codecs.passthrough import PassthroughCodec

Composite codecs (BinaryCodec, TextCodec, NoOpCodec) are available from this module:
    from virtuals.codecs import BinaryCodec, TextCodec, NoOpCodec
"""

from __future__ import annotations

from functools import partial

from virtuals._backends.key_codecs import BinaryKeyCodec, StringKeyCodec
from virtuals.tkv.codec import Codec

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
