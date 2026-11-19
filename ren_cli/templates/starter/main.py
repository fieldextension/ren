from ren.ckks import (
    Ciphertext,
    CKKSParams,
    decode,
    decrypt,
    encode,
    encrypt,
    keygen,
)


def circuit(x: Ciphertext) -> Ciphertext:
    return x**2 + x + 1


def main() -> None:
    inputs = [1.0, 2.0, 3.0]
    params = CKKSParams.default(boot_params=None)
    keys = keygen(params)

    # Encrypt the inputs.
    with keys.context():
        plaintext = encode(inputs, msg_bound=4)
        encrypted = encrypt(plaintext)

    # Evaluate the circuit using public keys.
    with keys.public().context():
        result = circuit(encrypted)

    # Decrypt and decode the result.
    with keys.context():
        plaintext_result = decrypt(result)
        decoded = decode(plaintext_result, length=len(inputs))

    print(f"inputs:    {inputs}")
    print(f"decrypted: {[slot.real for slot in decoded]}")


if __name__ == "__main__":
    main()
