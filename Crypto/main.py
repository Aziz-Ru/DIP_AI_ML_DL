import hashlib
# Return the Unicode code point for a one-character string.
x=ord('c')
print(x)
txt = "Hello, World!"
encode = txt.encode()
print(txt.encode().hex())
print(hashlib.md5(encode).hexdigest())