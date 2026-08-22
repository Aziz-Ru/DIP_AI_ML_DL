import os
def load_mapping(filename:str = 'mapping.txt'):
    mapping = {}
    path = os.path.join(os.path.dirname(__file__), filename)
    print(f"Loading mapping from: {path}")
    with open(path, 'r') as file:
      for line in file:
        key, value = line.strip().split()
        # print(f"Mapping: {key} -> {value}")
        mapping[key] = value
    return mapping

def single_substitution(text:str, mapping:dict):
  result = ''
  for ch in text:
    if ch.upper() in mapping:
      result += mapping[ch.upper()]
    else:
      result += ch
  return result

def block_substitution(text:str,key ='YDW'):
  text = text.upper().replace(" ", "")
  key = (key.upper()*(len(text)//len(key) +1))[:len(text)]
  result = []
  for p,k in zip(text,key):
    if p.isalpha():
      shift = ord(k) - 65
      result.append(chr((ord(p) - 65 + shift) % 26 + 65))
    else:
      result.append(p)
  return ''.join(result)

def ploygram_encrypt(plain_text:str, key:str='ABD',block_size=3):
  text = plain_text.upper().replace(" ", "")
  result = []
  # repeat the key to match the length of the text
  key = (key.upper()*((len(text)//len(key)) +1))[:len(text)]
  for p,k in zip(text,key):
    # calculate the shift value based on the key character
    if p.isalpha():
      shift = ord(k) - 65
      # apply the shift to the plaintext character
      result.append(chr((ord(p) - 65 + shift) % 26 + 65))
    else:
      result.append(p)
  return ''.join(result)


def ploygram_decrypt(cipher_text:str, key:str='ABD',block_size=3):
  text = cipher_text.upper().replace(" ", "")
  result = []
  # repeat the key to match the length of the text
  key = (key.upper()*((len(text)//len(key)) +1))[:len(text)]
  for c,k in zip(text,key):
    # calculate the shift value based on the key character
    if c.isalpha():
      shift = ord(k) - 65
      # apply the reverse shift to the ciphertext character
      result.append(chr((ord(c) - 65 - shift) % 26 + 65))
    else:
      result.append(c)
  return ''.join(result)


text = "Department of CSE Rajshahi University"
key = "ABD"
# cipher_text = ploygram_encrypt(text, key)
# decrypted_text = ploygram_decrypt(cipher_text, key)
# print("Plaintext :", text)
# print("Ciphertext:", cipher_text)
# print("Decrypted :", decrypted_text)

mapping = load_mapping()
substituted_text = single_substitution(text, mapping)
print("Substituted Text:", substituted_text)