import random
import typing as tp


def is_prime(n: int) -> bool:
    if n < 2: return False
    elif n == 2: return True

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
    


def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
    


def multiplicative_inverse(e: int, phi: int) -> int:
    a, b = e, phi
    x0, x1 = 1, 0
    y0, y1 = 0, 1
    
    while b != 0:
        q = a // b
        a, b = b, a % b
        
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return x0 % phi
    

def generate_keypair(p: int, q: int) -> tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]:
    if not (is_prime(p) and is_prime(q)):
        raise ValueError("Both numbers must be prime.")
    elif p == q:
        raise ValueError("p and q cannot be equal")

    n = p * q
    phi = (p - 1) * (q - 1)
    
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)

    d = multiplicative_inverse(e, phi)

    return ((e, n), (d, n))


def powm(a, n, m):
    a %= m
    res = 1
    while n > 0:
        if n % 2 == 1:
            res = res * a % m
        a = a * a % m
        n //= 2
    return res


def encrypt(pk: tp.Tuple[int, int], plaintext: str) -> tp.List[int]:
    key, n = pk
    cipher = [pow(ord(char), key, n) for char in plaintext]
    return cipher


def decrypt(pk: tp.Tuple[int, int], ciphertext: tp.List[int]) -> str:
    key, n = pk
    plain = [chr(pow(char, key, n)) for char in ciphertext]
    return "".join(plain)


if __name__ == "__main__":
    print("RSA Encrypter/ Decrypter")
    p = int(input("Enter a prime number (17, 19, 23, etc): "))
    q = int(input("Enter another prime number (Not one you entered above): "))
    print("Generating your public/private keypairs now . . .")
    public, private = generate_keypair(p, q)
    print("Your public key is ", public, " and your private key is ", private)
    message = input("Enter a message to encrypt with your private key: ")
    encrypted_msg = encrypt(private, message)
    print("Your encrypted message is: ")
    print("".join(map(lambda x: str(x), encrypted_msg)))
    print("Decrypting message with public key ", public, " . . .")
    print("Your message is:")
    print(decrypt(public, encrypted_msg))
