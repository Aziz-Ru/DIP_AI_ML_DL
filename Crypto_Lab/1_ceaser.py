def caesar_encrypt(plaintext:str, shift=3):
    result = []
    for ch in plaintext:
        if ch.isalpha():
            base = ord('A') if ch.isupper()else ord('a')
            result.append(chr((ord(ch)-base +shift)%26 + base))
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