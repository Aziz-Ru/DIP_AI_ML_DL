import math

def transposition_encrypt(text:str,width:int =4):
    text = text.replace(' ','')
    rows = math.ceil(len(text)/width)
    padded = text.ljust(rows*width,'X')
    grid =[padded[i:i+width] for i in range(0,len(padded),width)]
    print(f"Grid: {grid}")
    cipher =[]
    
    for col in range(width):
        for row in range(rows):
            cipher.append(grid[row][col])
    
    return ''.join(cipher)

def transpostion_decrypt(cipher:str,width:int =4):
    rows = math.ceil(len(cipher)/width)
    grid = ['']*rows
    # print(grid)
    idx =0
    for row in range(rows):
            if idx <len(cipher):
                grid[row]+=cipher[idx]
                idx+=1
        

    pt= ''.join(grid)
    return pt.rstrip('X')

# def transposition_encrypt(plaintext, width):
#     text = plaintext.replace(" ", "")
#     rows = math.ceil(len(text) / width)
#     print(f"Rows: {rows}, Width: {width}, Length of text: {len(text)}")
#     padded = text.ljust(rows * width, 'X')
#     print(f"Padded text: {padded}")

#     grid = [padded[i:i+width] for i in range(0, len(padded), width)]
#     print(f"Grid: {grid}")
#     cipher = ''.join(''.join(row[c] for row in grid) for c in range(width))
#     print(f"Ciphertext: {cipher}")
#     return cipher

# def transposition_decrypt(ciphertext, width):
#     rows = math.ceil(len(ciphertext) / width)
#     cols = width
#     grid = [''] * rows

#     idx = 0
#     for c in range(cols):
#         for r in range(rows):
#             if idx < len(ciphertext):
#                 grid[r] += ciphertext[idx]
#                 idx += 1
#     plaintext = ''.join(grid)
#     return plaintext.rstrip('X')


# def double_transposition_encrypt(plaintext, width1, width2):
#     stage1 = transposition_encrypt(plaintext, width1)
#     stage2 = transposition_encrypt(stage1, width2)
#     return stage2

# def double_transposition_decrypt(ciphertext, width1, width2):
#     stage1 = transposition_decrypt(ciphertext, width2)
#     stage2 = transposition_decrypt(stage1, width1)
#     return stage2

# Example
if __name__ == "__main__":
    pt = "DEPARTMENT OF COMPUTER SCIENCE AND TECHNOLOGY UNIVERSITY OF RAJSHAHI BANGLADESH"
    w1, w2 = 6, 4
    # ct = double_transposition_encrypt(pt, w1, w2)
    # dt = double_transposition_decrypt(ct, w1, w2)
    ct = transposition_encrypt(pt,w1)
    print("Ciphertext:", ct)
    pt = transpostion_decrypt(ct,w1)
    print("Plaintext :", pt)
    
    # print("Decrypted :", dt)
    # 
# # Example
# if __name__ == "__main__":
#     pt = "DEPARTMENT OF COMPUTER SCIENCE AND TECHNOLOGY UNIVERSITY OF RAJSHAHI BANGLADESH"
#     width = int(input("Enter width: ") or 6)
#     ct = transposition_encrypt(pt, width)
#     dt = transposition_decrypt(ct, width)
#     print("Ciphertext:", ct)
#     print("Decrypted :", dt)


