import os
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP,AES
from Crypto.Signature import pss
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes

def generate_key():
    key = RSA.generate(2048)
    return key, key.publickey()


private_key, public_key = generate_key()
#  Authentication
# bytes of message
def sender():
  msg = 'Hello, this is a secret message.'
  msg_hash = SHA256.new(msg.encode())
  # digital signature
  signature = pss.new(private_key).sign(msg_hash)
  # print("Message:", msg)
  # print("Message Hash:", msg_hash.hexdigest())
  # print("Digital Signature:", signature.hex())

  signature_b64 = base64.b64encode(signature).decode()
  msg_b64 = base64.b64encode(msg.encode()).decode()
  return signature_b64+'||'+msg_b64

def receiver(cipher:str):
   sign_b64, msg_b64 = cipher.split('||')
   signature = base64.b64decode(sign_b64)
   msg = base64.b64decode(msg_b64).decode()
   msg_hash = SHA256.new(msg.encode())
   verifier = pss.new(public_key)
   try:
        verifier.verify(msg_hash, signature)
        print("Signature is valid.")
        print("Message:", msg)
        
   except (ValueError, TypeError):
        print("Signature is invalid.")

# Confidentaility
def msg_send(msg:str,receiver_pk:RSA.RsaKey):
    # Generate a random AES key
    aes_key = get_random_bytes(16)
    cipher_aes = AES.new(aes_key,AES.MODE_EAX )
    cihpertext, tag = cipher_aes.encrypt_and_digest(msg.encode())
    nonce = cipher_aes.nonce
    # Encrypt the AES key with the receiver's public key
    # r_pk = RSA.import_key(receiver_pk)
    cipher_rsa = PKCS1_OAEP.new(receiver_pk)
    encrypted_aes_key = cipher_rsa.encrypt(aes_key)
    encrypted_aes_key = base64.b64encode(encrypted_aes_key).decode()
    ciphertext_b64 = base64.b64encode(cihpertext).decode()
    tag_b64 = base64.b64encode(tag).decode()
    nonce_b64 = base64.b64encode(nonce).decode()

    rmsg = encrypted_aes_key+'||'+ciphertext_b64+'||'+tag_b64+'||'+nonce_b64
    return rmsg

def msg_receive(cipher:str,receiver_sk):
    enc_aes_key_b64, ciphertext_b64, tag_b64, nonce_b64 = cipher.split('||')
    enc_aes_key = base64.b64decode(enc_aes_key_b64)
    ciphertext = base64.b64decode(ciphertext_b64)
    tag = base64.b64decode(tag_b64)
    nonce = base64.b64decode(nonce_b64)
    # Decrypt the AES key with the receiver's private key
    # r_sk = RSA.import_key(receiver_sk)
    cipher_rsa = PKCS1_OAEP.new(receiver_sk)
    aes_key = cipher_rsa.decrypt(enc_aes_key)
    # Decrypt the message with the AES key
    cipher_aes = AES.new(aes_key, AES.MODE_EAX, nonce=nonce)
    msg = cipher_aes.decrypt_and_verify(ciphertext, tag).decode()
    return msg

if __name__ =='__main__':
   sender_msg = sender()
   print(sender_msg)
   receiver(sender_msg)
   msg = "Hello, this is a secret message."
   receiver_private_key, receiver_public_key = generate_key()
   cipher_msg = msg_send(msg,receiver_public_key)
   print("Ciphertext:", cipher_msg)
   print('Decrypting...')
   msg = msg_receive(cipher_msg,receiver_private_key)
   print("Decrypted Message:", msg)
