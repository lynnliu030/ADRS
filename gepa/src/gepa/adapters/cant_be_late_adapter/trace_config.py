"""Shared configuration for cant-be-late trace handling."""

TRACE_SAMPLE_IDS: list[int] = [
    0,
    8,
    9,
    20,
    21,
    33,
    42,
    51,
    61,
    70,
    99,
    107,
    117,
    126,
    135,
    145,
    154,
    163,
    172,
    182,
    191,
    219,
    228,
    238,
    247,
    256,
    266,
    275,
    284,
    294,
]

# Overheads that have been extracted in the reference archive.
TRACE_OVERHEADS: list[float] = [0.02, 0.20, 0.40]
