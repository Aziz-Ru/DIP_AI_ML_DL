def caesar_encrypt(plaintext, shift=3):
    result = []
    for ch in plaintext:
        if ch.isupper():
            result.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        elif ch.islower():
            result.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        else:
            result.append(ch)
    return ''.join(result)

def caesar_decrypt(ciphertext, shift=3):
    return caesar_encrypt(ciphertext, -shift)

# Example
if __name__ == "__main__":
    pt = "Department of CSE Rajshahi University"
    ct = caesar_encrypt(pt)
    dt = caesar_decrypt(ct)
    print("Plaintext :", pt)
    print("Ciphertext:", ct)
    print("Decrypted :", dt)