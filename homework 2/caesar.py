import typing as tp
from string import ascii_lowercase, ascii_uppercase

def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    shift_abc = lambda abc, k: abc[k:] + abc[:k]
    abc_lowercase = ascii_lowercase
    abc_uppercase = ascii_uppercase

    abc_lowercase_shifted = shift_abc(abc_lowercase, shift)
    abc_uppercase_shifted = shift_abc(abc_uppercase, shift)
    table = str.maketrans({x:y for (x,y) in zip(abc_lowercase + abc_uppercase,
                                                abc_lowercase_shifted + abc_uppercase_shifted)})
    return plaintext.translate(table)
    


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    abc_lowercase = ascii_lowercase
    abc_uppercase = ascii_uppercase
    de_shift = len(abc_lowercase) - shift
    shift_abc = lambda abc, k: abc[k:] + abc[:k]
    
    abc_lowercase_shifted = shift_abc(abc_lowercase, de_shift)
    abc_uppercase_shifted = shift_abc(abc_uppercase, de_shift)
    table = str.maketrans({x:y for (x,y) in zip(abc_lowercase + abc_uppercase,
                                                abc_lowercase_shifted + abc_uppercase_shifted)})
    return ciphertext.translate(table)
    

def powm(a, n, m):
    a %= m
    res = 1
    while n > 0:
        if n % 2 == 1:
            res = res * a % m
        a = a * a % m
        n //= 2
    return res


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    # PUT YOUR CODE HERE
    return best_shift
