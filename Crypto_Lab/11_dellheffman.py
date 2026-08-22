import random

def differ_hellman():
  p =23
  g =5
  a= random.randint(2,p-2)
  b = random.randint(2,p-2)
  A = pow(g,a,p)
  B = pow(g,b,p)
  print("Public Key A:",A)
  print("Public Key B:",B)
  K1 = pow(B,a,p)
  K2 = pow(A,b,p)
  print("Shared Secret Key K1:",K1)
  print("Shared Secret Key K2:",K2)
  if K1==K2:
    print("Shared Secret Key is same for both parties.")
  else:
    print("Shared Secret Key is different for both parties.")

differ_hellman()