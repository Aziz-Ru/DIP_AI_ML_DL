import random
import string

def generate_key_file(length, filename="otp_key.txt"):
    key = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
    with open(filename, 'w') as f:
        f.write(key)
    return key

def otp_encrypt(plaintext, key):
    text = plaintext.upper().replace(" ", "")
    cipher = []
    for p, k in zip(text, key):
        c = (ord(p) - 65 + ord(k) - 65) % 26 + 65
        cipher.append(chr(c))
    return ''.join(cipher)

def otp_decrypt(ciphertext, key):
    plain = []
    for c, k in zip(ciphertext, key):
        p = (ord(c) - 65 - (ord(k) - 65)) % 26 + 65
        plain.append(chr(p))
    return ''.join(plain)

# Example
if __name__ == "__main__":
    pt = "DEPARTMENTOFCSE"
    key = generate_key_file(len(pt))     # random key, same length as plaintext, non-repeating use
    ct = otp_encrypt(pt, key)
    dt = otp_decrypt(ct, key)
    print("Key       :", key)
    print("Ciphertext:", ct)
    print("Decrypted :", dt)