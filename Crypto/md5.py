import hashlib

def md5_hash(message):
    return hashlib.md5(message.encode()).hexdigest()

# Example
if __name__ == "__main__":
    msg = "Department of CSE, University of Rajshahi"
    print("Message:", msg)
    print("MD5    :", md5_hash(msg))