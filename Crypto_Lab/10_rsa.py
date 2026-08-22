from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
key_pair = RSA.generate(1024)

private_key = key_pair.export_key()
public_key = key_pair.publickey().export_key()

# print("Private Key:", private_key.decode())
# print("Public Key:", public_key.decode())
reciver_public_key = RSA.import_key(public_key)
# print("Reciver Public Key:", reciver_public_key)
msg = b'Hello, this is a secret message.'
cipher_rsa = PKCS1_OAEP.new(reciver_public_key)

ciphertext = cipher_rsa.encrypt(msg)
# print("Ciphertext:", ciphertext)
print("Encrypting...")
print("Ciphertext:", ciphertext.hex())
print('Decrypting...')
secret_key = RSA.import_key(private_key)
cipher_rsa = PKCS1_OAEP.new(secret_key)
decrypted_msg = cipher_rsa.decrypt(ciphertext)
print("Decrypted Message:", decrypted_msg.decode())

# encrypted_msg = public_key.encrypt(msg,None) 
# print("Encrypted Message:", encrypted_msg)
