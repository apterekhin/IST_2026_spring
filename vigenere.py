from caesar import encrypt_caesar, decrypt_caesar
from string import ascii_lowercase, ascii_uppercase


def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    cipher_text = list(plaintext)
    
    abc_upper = ascii_uppercase
    abc_lower = ascii_lowercase
    n = len(keyword)
    password = [abc_upper.index(i) if i.isupper() else abc_lower.index(i) for i in keyword]
    
    for i in range(n):
        indices = list(range(i, len(plaintext), n))
        substring = ''.join(plaintext[idx] for idx in indices)
        encrypted_substring = encrypt_caesar(substring, password[i])
        for j, idx in enumerate(indices):
            cipher_text[idx] = encrypted_substring[j]
    
    return ''.join(cipher_text)


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    plaintext = list(ciphertext)
    
    abc_upper = ascii_uppercase
    abc_lower = ascii_lowercase
    n = len(keyword)
    password = [abc_upper.index(i) if i.isupper() else abc_lower.index(i) for i in keyword]
    
    for i in range(n):
        indices = list(range(i, len(ciphertext), n))
        substring = ''.join(ciphertext[idx] for idx in indices)
        decrypted_substring = decrypt_caesar(substring, password[i])
        for j, idx in enumerate(indices):
            plaintext[idx] = decrypted_substring[j]
    
    return ''.join(plaintext)
