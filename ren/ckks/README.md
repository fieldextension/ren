# CKKS

Ren implements CKKS using its polynomial engine. Import scheme operations from `ren.ckks`.

- [params.py](params.py) defines parameter sets, level schedules, rescaling, and bootstrapping parameters.
- [keys.py](keys.py) generates keys and groups them in a `CKKSKeySet`.
- [context.py](context.py) selects the active keys and device and caches transform plans.
- [ckks.py](ckks.py) defines `Plaintext`, `Ciphertext`, encoding, encryption, decryption, and ciphertext arithmetic.
- [arithmetic_ops.py](arithmetic_ops.py) implements ciphertext sums and dot products.
- [bootstrap.py](bootstrap.py) implements bootstrapping.
- [multiparty.py](multiparty.py) implements distributed key generation and partial decryption.
- [errors.py](errors.py) defines CKKS exceptions.
