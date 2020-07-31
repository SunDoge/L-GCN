MSVD_QTYPES = [
    'what', 'who', 'how', 'when', 'where'
]

MSVD_QTYPE_DICT = {k: v for v, k in enumerate(MSVD_QTYPES)}

# ingore index for crossentropy
IGNORE_INDEX = -100
