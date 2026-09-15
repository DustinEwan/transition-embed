# Artifact spec

The trained codebook is exported as a single `.pt` file — the **portable tied
table**. The raw `(V, D)` values are a gradient carrier only; they are never
needed at inference. What is stored is the 2-bit interface plus the scalars.

## Fields

| field | shape / dtype | meaning |
|---|---|---|
| `bits` | `(V, D//8)` uint8 | bit-packed sign, **1 bit per code** |
| `c` | `(D,)` | centering vector (unigram-weighted mean of the sign) |
| `in_bias` | `(V,)` | per-token input scalar |
| `out_bias` | `(V,)` | per-token output scalar (carries frequency) |
| `alpha` | scalar | temperature (log-temperature, exponentiated) |
| `unigram_p` | `(V,)` | corpus unigram probabilities |

## Bit layout

`bits[i, b]` is a byte holding 8 codes. **Bit `j` (0-indexed, LSB first) of byte
`b` holds code `8*b + j`.** A set bit (1) = `+1`; a clear bit (0) = `-1`.

```
decode:  sign = unpack_bits(bits) * 2 - 1        # (V, D) in {-1, +1}
pack:    bits = pack_bits(sign > 0)              # (V, D//8) uint8
```

`unpack_bits` does `(byte & [1,2,4,...,128]).ne(0)`; `pack_bits` does
`(bool.view(V, D//8, 8) * [1,2,4,...,128]).sum(-1)`. They are exact inverses.

## Reconstructing the interface

```python
sign  = unpack_bits(bits) * 2 - 1              # (V, D) raw sign
codes = sign - c                                # (V, D) unigram-centered (the 2-bit interface)
# sum_j unigram_p[j] * codes[j] == 0  (exactly, by construction)

# input representation (for token id i):
in_repr = codes[i] + (in_bias[i] - sum_j unigram_p[j] * in_bias[j])

# output logits (for hidden state h):
logits  = alpha * (h @ codes.T) + out_bias
```

## Size

For `V = 151669`, `D = 512`:
- `bits`: `151669 * 64` bytes = **9.7 MB**
- `c`, `in_bias`, `out_bias`, `unigram_p`: ~`2.4 MB`
- **total ≈ 11.5 MB** (vs. ~310 MB for the raw fp32 table)

The per-token cost is `D + 32` bits on the input side (512 sign bits + a 32-bit
scalar); the V-dependence is weak (only `16384 / V` bits for `c` amortized).

## Round-trip guarantee

`unpack_bits(pack_bits(b)) == b` for any `(V, D)` bool tensor `b`. The signed
codes, the centering, and the scalars round-trip exactly; the raw values do not
need to (they are a gradient carrier only).
