import math

def transposition_encrypt(plaintext, width):
    text = plaintext.replace(" ", "")
    rows = math.ceil(len(text) / width)
    padded = text.ljust(rows * width, 'X')

    grid = [padded[i:i+width] for i in range(0, len(padded), width)]
    cipher = ''.join(''.join(row[c] for row in grid) for c in range(width))
    return cipher

def transposition_decrypt(ciphertext, width):
    rows = math.ceil(len(ciphertext) / width)
    cols = width
    grid = [''] * rows

    idx = 0
    for c in range(cols):
        for r in range(rows):
            if idx < len(ciphertext):
                grid[r] += ciphertext[idx]
                idx += 1
    plaintext = ''.join(grid)
    return plaintext.rstrip('X')


def double_transposition_encrypt(plaintext, width1, width2):
    stage1 = transposition_encrypt(plaintext, width1)
    stage2 = transposition_encrypt(stage1, width2)
    return stage2

def double_transposition_decrypt(ciphertext, width1, width2):
    stage1 = transposition_decrypt(ciphertext, width2)
    stage2 = transposition_decrypt(stage1, width1)
    return stage2

# Example
if __name__ == "__main__":
    pt = "DEPARTMENT OF COMPUTER SCIENCE AND TECHNOLOGY UNIVERSITY OF RAJSHAHI BANGLADESH"
    w1, w2 = 6, 4
    ct = double_transposition_encrypt(pt, w1, w2)
    dt = double_transposition_decrypt(ct, w1, w2)
    print("Ciphertext:", ct)
    print("Decrypted :", dt)
    
# Example
if __name__ == "__main__":
    pt = "DEPARTMENT OF COMPUTER SCIENCE AND TECHNOLOGY UNIVERSITY OF RAJSHAHI BANGLADESH"
    width = int(input("Enter width: ") or 6)
    ct = transposition_encrypt(pt, width)
    dt = transposition_decrypt(ct, width)
    print("Ciphertext:", ct)
    print("Decrypted :", dt)


