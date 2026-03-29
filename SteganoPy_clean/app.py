# ============================================================
#  SteganoPy — Adaptive LSB Steganography
#  Aplicație web pentru ascunderea mesajelor în imagini
#  folosind tehnica LSB Adaptivă (Least Significant Bit)
# ============================================================

# --- Importăm bibliotecile necesare ---
import streamlit as st          # Framework pentru interfața web
from PIL import Image           # Procesarea imaginilor
import numpy as np              # Calcule matematice pe matrici/array-uri
import io                       # Lucrul cu fluxuri de date (buffere în memorie)
import base64                   # Codificarea datelor binare în text (Base64)
import hashlib                  # Generarea de hash-uri (ex: SHA-256 pentru parolă)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # Algoritmi AES
from cryptography.hazmat.backends import default_backend                       # Backend criptografic
from cryptography.hazmat.primitives import padding                             # Padding pentru blocuri AES
import os                       # Funcții de sistem (ex: generare bytes aleatorii)
import zipfile                  # Crearea și citirea arhivelor ZIP
from datetime import datetime   # Data și ora curentă (pentru denumirea fișierelor)
import matplotlib.pyplot as plt # Generarea graficelor și vizualizărilor
import matplotlib.gridspec as gridspec  # Layout avansat pentru subplot-uri
from scipy import ndimage       # Filtre matematice pe imagini (ex: Laplacian)


# ─────────────────────────────────────────────
#  UTILITĂȚI: conversii text ↔ binar ↔ bytes
# ─────────────────────────────────────────────

def text_to_binary(text):
    """
    Convertește un șir de text în reprezentarea sa binară.
    Fiecare caracter devine un grup de 8 biți (un octet).
    
    Exemplu: 'A' → '01000001'
    """
    return ''.join(format(ord(c), '08b') for c in text)


def binary_to_text(binary):
    """
    Convertește un șir de biți înapoi în text.
    Procesează câte 8 biți odată și îi transformă în caracterul corespunzător.
    
    Exemplu: '01000001' → 'A'
    """
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]   # Luăm câte 8 biți pe rând
        if len(byte) == 8:     # Ignorăm grupurile incomplete (la final)
            text += chr(int(byte, 2))  # Convertim binarul în număr, apoi în caracter
    return text


def bytes_to_binary(data):
    """
    Convertește date binare (bytes) în șir de biți.
    Folosit pentru ascunderea fișierelor arbitrare în imagini.
    
    Exemplu: b'\x41' → '01000001'
    """
    return ''.join(format(b, '08b') for b in data)


def binary_to_bytes(binary):
    """
    Convertește un șir de biți înapoi în bytes.
    Inversul funcției bytes_to_binary.
    
    Exemplu: '01000001' → b'\x41'
    """
    ba = bytearray()
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            ba.append(int(byte, 2))  # Convertim 8 biți în număr (0-255) și adăugăm la bytearray
    return bytes(ba)


# ─────────────────────────────────────────────
#  CRIPTARE / DECRIPTARE  (AES-256 CBC)
# ─────────────────────────────────────────────

def encrypt_message(message, password):
    """
    Criptează un mesaj text folosind algoritmul AES-256 în modul CBC.
    
    Pași:
    1. Parola este transformată într-o cheie de 256 de biți cu SHA-256
    2. Se generează un IV (vector de inițializare) aleatoriu de 16 bytes
    3. Mesajul este completat (padding) la un multiplu de 16 bytes
    4. Mesajul completat este criptat
    5. Rezultatul (IV + date criptate) este codificat în Base64 pentru transport ușor
    
    Returnează: mesajul criptat ca șir Base64
    """
    # Transformăm parola într-o cheie de 32 bytes (256 biți) folosind SHA-256
    key = hashlib.sha256(password.encode()).digest()

    # Generăm un vector de inițializare (IV) aleatoriu — necesar pentru modul CBC
    # IV-ul diferit la fiecare criptare asigură că același mesaj produce rezultate diferite
    iv  = os.urandom(16)

    # Creăm obiectul de criptare AES-256 în modul CBC
    cipher    = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # AES operează pe blocuri de 16 bytes, deci completăm mesajul (PKCS7 padding)
    padder      = padding.PKCS7(128).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()

    # Criptăm efectiv datele
    encrypted = encryptor.update(padded_data) + encryptor.finalize()

    # Concatenăm IV + date criptate și codificăm în Base64
    # IV-ul trebuie salvat alături de date pentru a putea decripta mai târziu
    return base64.b64encode(iv + encrypted).decode('utf-8')


def decrypt_message(encrypted_message, password):
    """
    Decriptează un mesaj criptat cu encrypt_message.
    
    Pași:
    1. Se decodifică Base64 pentru a obține bytes bruti
    2. Primii 16 bytes sunt IV-ul
    3. Restul sunt datele criptate
    4. Se decriptează cu aceeași cheie derivată din parolă
    5. Se elimină padding-ul și se returnează textul original
    
    Aruncă excepție dacă parola este greșită.
    """
    # Derivăm aceeași cheie din parolă
    key = hashlib.sha256(password.encode()).digest()

    # Decodificăm din Base64
    encrypted_data = base64.b64decode(encrypted_message)

    # Separăm IV-ul (primii 16 bytes) de restul datelor criptate
    iv         = encrypted_data[:16]
    ciphertext = encrypted_data[16:]

    # Creăm obiectul de decriptare cu aceeași cheie și IV
    cipher    = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decriptăm datele
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Eliminăm padding-ul PKCS7 adăugat la criptare
    unpadder = padding.PKCS7(128).unpadder()
    data     = unpadder.update(padded_data) + unpadder.finalize()

    return data.decode('utf-8')


# ─────────────────────────────────────────────
#  NUCLEUL LSB ADAPTIV
# ─────────────────────────────────────────────

def compute_texture_map(img_array, bits_per_channel=1):
    """
    Calculează un scor de complexitate per pixel folosind filtrul Laplacian.
    
    Ideea de bază: pixelii de pe margini și zone texturate ale imaginii
    suferă modificări minore greu de observat, spre deosebire de zonele
    uniforme (cer, pereți) unde orice modificare este vizibilă.
    
    Filtrul Laplacian detectează schimbările bruște de intensitate (muchii).
    Cu cât valoarea este mai mare, cu atât pixelul este mai "texturat".
    
    IMPORTANT: Mascăm biții LSB înainte de calculul Laplacianului!
    Aceasta asigură că harta de textură este IDENTICĂ înainte și după
    codificarea mesajului. Fără această mascare, codificatorul și
    decodificatorul ar vedea hărți de textură diferite, ducând la
    ordini diferite ale pixelilor și, implicit, la decodare eșuată.
    
    Returnează: matricea de magnitudini (scor per pixel)
    """
    # Mascăm biții cel mai puțin semnificativi — aceștia sunt cei pe care
    # îi modificăm la codificare, deci trebuie ignorați la calculul texturii
    # Exemplu: pentru 1 bit/canal, masca = 0xFE = 11111110
    mask = np.uint8(0xFF ^ ((1 << bits_per_channel) - 1))
    cleaned = (img_array & mask).astype(np.float32)

    # Convertim imaginea curățată în nuanțe de gri (media canalelor R, G, B)
    gray = np.mean(cleaned, axis=2)

    # Aplicăm filtrul Laplacian pentru a detecta muchiile și textura
    lap = ndimage.laplace(gray)

    # Luăm valoarea absolută — nu ne interesează direcția, ci intensitatea
    magnitude = np.abs(lap)

    return magnitude


def build_pixel_order(magnitude, img_array, bits_needed, bits_per_channel,
                      password=None):
    """
    Construiește lista ordonată de poziții (rând, coloană, canal) în care
    vor fi ascunși biții mesajului.
    
    Strategia:
    - Sortăm pixelii de la cel mai complex (mult texturat) la cel mai simplu
    - Dacă există o parolă, amestecăm TOȚI pixelii (nu doar subsetul!) pentru
      a asigura consistența cu decodificatorul
    - Selectăm doar câți pixeli sunt necesari pentru mesaj
    
    IMPORTANT: Amestecarea trebuie făcută pe TOȚI pixelii sortați ÎNAINTE de
    selecția subsetului. Decodificatorul face aceeași amestecarea pe toți pixelii
    (fiindcă nu știe lungimea mesajului). Dacă am amesteca doar subsetul,
    ordinea ar fi diferită între codificator și decodificator.
    
    Aruncă excepție dacă imaginea este prea mică pentru mesaj.
    """
    h, w, c = img_array.shape
    channels_used = 3  # Folosim canalele R, G, B

    # Aplatizăm harta de magnitudini și sortăm descrescător (cei mai complecși primii)
    flat_mag = magnitude.flatten()
    sorted_px = np.argsort(flat_mag)[::-1]  # argsort crescător, inversăm cu [::-1]

    # IMPORTANT: Amestecăm TOȚI pixelii ÎNAINTE de selecția subsetului
    # Aceasta asigură consistența: decodificatorul face aceeași amestecarea
    # pe toți pixelii cu aceeași sămânță → obține aceeași ordine
    if password:
        rng = np.random.default_rng(
            int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
        )
        rng.shuffle(sorted_px)

    # Calculăm câți pixeli avem nevoie (fiecare pixel are 3 canale × biți_per_canal)
    pixels_needed = int(np.ceil(bits_needed / (bits_per_channel * channels_used)))

    # Verificăm că imaginea are suficienți pixeli
    if pixels_needed > h * w:
        raise ValueError(
            f"Message too long! Need {pixels_needed} pixels but image only has {h*w}."
        )

    # Luăm doar pixelii necesari din array-ul (deja amestecat, dacă parolă)
    selected_px = sorted_px[:pixels_needed]

    # Expandăm fiecare pixel în trei poziții: (rând, coloană, canal_R/G/B)
    # Folosim NumPy vectorizat în loc de buclă Python → mult mai rapid și eficient
    rows = (selected_px // w).astype(np.int32)
    cols = (selected_px % w).astype(np.int32)

    # Repetăm fiecare pixel de 3 ori (câte o intrare per canal R, G, B)
    rows_exp = np.repeat(rows, channels_used)
    cols_exp = np.repeat(cols, channels_used)
    # Canalele ciclează: 0, 1, 2, 0, 1, 2, ...
    chs_exp  = np.tile(np.arange(channels_used, dtype=np.int32), len(selected_px))

    # Returnăm ca matrice NumPy (N×3) — mult mai puțină memorie decât lista de tuple Python
    return np.column_stack([rows_exp, cols_exp, chs_exp])


def build_pixel_order_generator(magnitude, img_array, bits_per_channel,
                                password=None):
    """
    Versiune GENERATOR a funcției build_pixel_order, optimizată pentru DECODARE.
    
    La decodare nu știm câți biți are mesajul, deci trebuie să parcurgem
    pixelii pe rând până găsim delimitatorul. Generatorul produce pozițiile
    câte una, fără a le stoca pe toate în memorie.
    
    Aceasta rezolvă problema de memorie: o imagine de 4000×3000 pixeli ar
    necesita ~2.6 GB RAM pentru lista completă de tuple Python, dar generatorul
    folosește doar ~48 MB (array-ul NumPy sortat).
    
    CONSISTENȚĂ: Amestecarea se face pe TOȚI pixelii sortați — exact ca în
    build_pixel_order — asigurând că primele N poziții sunt identice cu cele
    folosite la codificare.
    """
    h, w, c = img_array.shape
    channels_used = 3

    flat_mag = magnitude.flatten()
    sorted_px = np.argsort(flat_mag)[::-1]

    # Amestecăm TOȚI pixelii cu aceeași sămânță → aceeași ordine ca la codificare
    if password:
        rng = np.random.default_rng(
            int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
        )
        rng.shuffle(sorted_px)

    # Producem pozițiile câte una — fără a construi o listă imensă
    for px_idx in sorted_px:
        row = int(px_idx) // w
        col = int(px_idx) % w
        for ch in range(channels_used):
            yield (row, col, ch)


def adaptive_encode(image, message, password=None, bits_per_channel=1):
    """
    Ascunde mesajul în imagine folosind LSB Adaptiv.
    
    Algoritmul:
    1. (Opțional) Criptează mesajul cu AES-256 dacă există parolă
    2. Adaugă un delimitator de final ("###END###") la mesaj
    3. Convertește mesajul în biți
    4. Calculează harta de textură a imaginii
    5. Selectează pixelii potriviți (cei mai complecși)
    6. Modifică bitul/biții cel(i) mai puțin semnificativ(i) ai fiecărui canal
       pentru a stoca biții mesajului
    
    Returnează: imaginea modificată (cu mesajul ascuns)
    """
    # Copiem array-ul imaginii pentru a nu modifica originalul
    img_array = np.array(image).copy()

    # Dacă există parolă, criptăm mesajul înainte de a-l ascunde
    if password:
        message = encrypt_message(message, password)

    # Adăugăm un marcaj de final pentru a ști unde se termină mesajul la extragere
    DELIMITER    = "###END###"
    full_message = message + DELIMITER
    binary_msg   = text_to_binary(full_message)
    bits_needed  = len(binary_msg)

    # Calculăm harta de textură și obținem ordinea pixelilor
    magnitude = compute_texture_map(img_array, bits_per_channel)
    positions = build_pixel_order(magnitude, img_array, bits_needed,
                                  bits_per_channel, password)

    # Verificare de siguranță: avem suficient spațiu?
    # positions este acum o matrice NumPy (N×3), deci folosim .shape[0] pentru numărul de rânduri
    if positions.shape[0] * bits_per_channel < bits_needed:
        raise ValueError("Not enough texture capacity in this image for the message.")

    bit_idx = 0
    # Masca curăță bitul(biții) cel(i) mai puțin semnificativ(i)
    # Exemplu pentru 1 bit: 0xFF ^ 0x01 = 0xFE = 11111110
    mask = 0xFF ^ ((1 << bits_per_channel) - 1)

    # Parcurgem pozițiile și scriem biții mesajului în LSB-urile pixelilor
    for (r, c, ch) in positions:
        if bit_idx >= bits_needed:
            break  # Am terminat de scris toți biții

        # Luăm grupul următor de biți din mesaj
        chunk = binary_msg[bit_idx: bit_idx + bits_per_channel]
        chunk = chunk.ljust(bits_per_channel, '0')  # Completăm cu 0 dacă e ultimul grup

        # Modificăm pixelul: ștergem LSB-urile cu masca, apoi scriem biții mesajului
        # Operația '&' cu masca = ștergere LSB; '|' cu chunk = scriere biți
        val = (int(img_array[r, c, ch]) & mask) | int(chunk, 2)
        img_array[r, c, ch] = val
        bit_idx += bits_per_channel

    # Returnăm imaginea modificată ca obiect PIL
    return Image.fromarray(img_array.astype('uint8'))


def adaptive_decode(image, password=None, bits_per_channel=1):
    """
    Extrage mesajul ascuns dintr-o imagine codificată cu adaptive_encode.
    
    Procesul este inversul codificării:
    1. Recalculăm harta de textură (identică cu cea de la codificare)
    2. Reconstruim aceeași ordine a pixelilor (cu aceeași parolă dacă există)
    3. Citim biții din LSB-urile pixelilor
    4. Căutăm delimitatorul de final
    5. Convertim biții în text
    6. (Opțional) Decriptăm textul dacă există parolă
    
    OPTIMIZARE: Folosim un generator pentru poziții (nu construim toată lista
    în memorie) și o listă Python pentru acumularea biților (în loc de
    concatenare de string-uri care este O(n²)).
    
    Aruncă excepție dacă nu găsește mesaj sau parola este greșită.
    """
    img_array = np.array(image)

    DELIMITER        = "###END###"
    delimiter_binary = text_to_binary(DELIMITER)
    delim_len        = len(delimiter_binary)

    # Recalculăm harta de textură
    magnitude = compute_texture_map(img_array, bits_per_channel)

    # Folosim GENERATORUL — produce poziții câte una, fără a le stoca pe toate
    # Aceasta economisește ~2 GB RAM pentru imaginile mari (vs. build_pixel_order)
    positions = build_pixel_order_generator(magnitude, img_array,
                                           bits_per_channel, password)

    # Acumulăm biții într-o listă (append = O(1)) în loc de string (concat = O(n))
    bits_list = []
    bit_mask  = (1 << bits_per_channel) - 1

    for (r, c, ch) in positions:
        # Citim bitul/biții LSB din pixelul curent
        val = int(img_array[r, c, ch]) & bit_mask
        bits_list.append(format(val, f'0{bits_per_channel}b'))

        # Verificăm dacă am găsit delimitatorul de final după fiecare caracter complet
        total_bits = len(bits_list) * bits_per_channel
        if total_bits % 8 == 0 and total_bits >= delim_len:
            # Construim doar sfârșitul string-ului pentru comparație
            # (nu concatenăm tot la fiecare pas)
            tail = ''.join(bits_list[-(delim_len // bits_per_channel):])
            if tail[-delim_len:] == delimiter_binary:
                break  # Am găsit sfârșitul mesajului, ne oprim

    # Concatenăm o singură dată la final (eficient)
    binary_message = ''.join(bits_list)

    # Dacă nu există delimitator, imaginea nu conține un mesaj valid
    if delimiter_binary not in binary_message:
        raise ValueError("No hidden message found in this image!")

    # Eliminăm delimitatorul de la final
    binary_message = binary_message[: -len(delimiter_binary)]

    # Convertim biții în text
    message = binary_to_text(binary_message)

    # Dacă mesajul a fost criptat, îl decriptăm
    if password:
        try:
            message = decrypt_message(message, password)
        except Exception:
            raise ValueError("Wrong password or message is not encrypted!")

    return message


# ─────────────────────────────────────────────
#  CLASSIC LSB  (păstrat pentru comparație)
# ─────────────────────────────────────────────

def encode_message_classic(image, message, password=None):
    """
    Varianta clasică (simplă) de steganografie LSB.
    
    Spre deosebire de metoda adaptivă, aceasta scrie biții secvențial
    începând cu primul pixel (0,0) și continuând la rând.
    
    Dezavantaj: această metodă este detectabilă cu testul chi-pătrat
    deoarece creează un pattern statistic nenatural în planul LSB.
    
    Se folosește pentru comparație cu metoda adaptivă.
    """
    img_array = np.array(image).copy()

    # Criptăm mesajul dacă există parolă
    if password:
        message = encrypt_message(message, password)

    DELIMITER  = "###END###"
    binary_msg = text_to_binary(message + DELIMITER)
    max_bytes  = img_array.shape[0] * img_array.shape[1] * 3  # Capacitatea totală în biți

    # Verificăm că mesajul încape în imagine
    if len(binary_msg) > max_bytes:
        raise ValueError(f"Message too long! Max {max_bytes // 8} characters.")

    # Aplatizăm imaginea într-un vector 1D pentru scriere secvențială simplă
    flat = img_array.flatten()

    # Scriem fiecare bit în LSB-ul pixelului corespunzător
    for i, bit in enumerate(binary_msg):
        # '& 0xFE' șterge LSB-ul, '| int(bit)' scrie noul bit
        flat[i] = (flat[i] & 0xFE) | int(bit)

    # Recompunem imaginea din vectorul modificat
    return Image.fromarray(flat.reshape(img_array.shape).astype('uint8'))


def decode_message_classic(image, password=None):
    """
    Extrage mesajul dintr-o imagine codificată cu encode_message_classic.
    
    Citește biții secvențial din LSB-urile pixelilor
    și se oprește când găsește delimitatorul de final.
    """
    img_array = np.array(image)
    flat      = img_array.flatten()  # Aplatizăm imaginea în 1D

    DELIMITER        = "###END###"
    delimiter_binary = text_to_binary(DELIMITER)
    binary_message   = ''

    # Citim câte un bit din LSB-ul fiecărui pixel
    for px in flat:
        binary_message += str(px & 1)  # Extragem LSB-ul cu '& 1'

        # Verificăm periodic dacă am găsit delimitatorul
        if len(binary_message) >= len(delimiter_binary):
            if binary_message[-len(delimiter_binary):] == delimiter_binary:
                break

    if delimiter_binary not in binary_message:
        raise ValueError("No hidden message found in this image!")

    # Eliminăm delimitatorul și convertim biții în text
    binary_message = binary_message[: -len(delimiter_binary)]
    message = binary_to_text(binary_message)

    # Decriptăm dacă e necesar
    if password:
        try:
            message = decrypt_message(message, password)
        except Exception:
            raise ValueError("Wrong password or message is not encrypted!")

    return message


# ─────────────────────────────────────────────
#  ASCUNDERE / EXTRAGERE FIȘIERE (LSB clasic)
# ─────────────────────────────────────────────

def encode_file(image, file_data, filename):
    """
    Ascunde un fișier arbitrar într-o imagine PNG folosind LSB clasic.
    
    Structura datelor ascunse:
    [info_fișier] + [DELIM] + [date_fișier_binare] + [DELIM]
    
    'info_fișier' conține numele și dimensiunea fișierului,
    separate de '|||' pentru a putea fi parsate la extragere.
    
    Exemplu de info: "document.pdf|||102400"
    """
    img_array = np.array(image).copy()

    # Creăm header-ul cu metadatele fișierului
    file_info = f"{filename}|||{len(file_data)}"
    DELIM     = "###FILEEND###"

    # Construim șirul binar complet: [header][DELIM][date fișier][DELIM]
    full_binary = (text_to_binary(file_info) + text_to_binary(DELIM)
                   + bytes_to_binary(file_data) + text_to_binary(DELIM))

    # Verificăm că fișierul încape în imagine
    max_bytes = img_array.shape[0] * img_array.shape[1] * 3
    if len(full_binary) > max_bytes:
        raise ValueError(f"File too large! Max ~{max_bytes // 8} bytes.")

    # Scriem biții secvențial în LSB-urile pixelilor
    flat = img_array.flatten()
    for i, bit in enumerate(full_binary):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    return Image.fromarray(flat.reshape(img_array.shape).astype('uint8'))


def decode_file(image):
    """
    Extrage un fișier ascuns dintr-o imagine codificată cu encode_file.
    
    Pași:
    1. Citește biții secvențiali din LSB-uri
    2. Caută primul delimitator → înainte de el este header-ul cu metadate
    3. Caută al doilea delimitator → între cele două este fișierul propriu-zis
    4. Parsează header-ul pentru a afla numele și dimensiunea fișierului
    5. Extrage și returnează datele fișierului
    
    Returnează: (date_fișier, nume_fișier)
    """
    img_array  = np.array(image)
    flat       = img_array.flatten()

    DELIM     = "###FILEEND###"
    delim_bin = text_to_binary(DELIM)

    binary_data  = ''
    delim_count  = 0  # Numărăm câte delimitatoare am găsit (avem nevoie de 2)

    # Citim biți până găsim ambele delimitatoare
    for px in flat:
        binary_data += str(px & 1)
        if len(binary_data) >= len(delim_bin) and binary_data[-len(delim_bin):] == delim_bin:
            delim_count += 1
            if delim_count == 2:
                break  # Am găsit ambele delimitatoare, oprim citirea

    if delim_count < 2:
        raise ValueError("No hidden file found or data is incomplete!")

    # Extragem header-ul (informațiile despre fișier) de dinaintea primului delimitator
    info_end  = binary_data.index(delim_bin)
    file_info = binary_to_text(binary_data[:info_end])

    # Parsăm header-ul: "nume_fisier|||dimensiune"
    parts = file_info.split('|||')
    if len(parts) != 2:
        raise ValueError("Invalid file format in metadata!")

    filename, file_size = parts[0], int(parts[1])

    # Extragem datele fișierului (între cele două delimitatoare)
    start = info_end + len(delim_bin)
    end   = len(binary_data) - len(delim_bin)
    fb    = binary_data[start:end][: file_size * 8]  # Luăm exact câți biți trebuie

    # Convertim biții în bytes
    fd = binary_to_bytes(fb)

    # Verificăm că am extras exact câți bytes trebuie
    if len(fd) != file_size:
        raise ValueError("Extracted file size mismatch!")

    return fd, filename


# ─────────────────────────────────────────────
#  METRICI DE CALITATE (PSNR / capacitate)
# ─────────────────────────────────────────────

def calculate_psnr(original, encoded):
    """
    Calculează PSNR (Peak Signal-to-Noise Ratio) între imaginea originală
    și cea codificată — o măsură a calității vizuale.
    
    PSNR se măsoară în decibeli (dB):
    - > 45 dB  → practic identice vizual (excelent)
    - 40-45 dB → imperceptibil pentru ochiul uman (bun)
    - 35-40 dB → zgomot minor acceptabil
    - < 35 dB  → degradare vizibilă
    
    Formula: PSNR = 20 × log10(255 / √MSE)
    unde MSE = Mean Squared Error (eroarea pătratică medie)
    """
    orig = np.array(original).astype(np.float64)
    enc  = np.array(encoded).astype(np.float64)

    # MSE = media pătratelor diferențelor între pixeli corespondenți
    mse = np.mean((orig - enc) ** 2)

    # Dacă MSE = 0, imaginile sunt identice → PSNR infinit
    if mse == 0:
        return float('inf')

    # Calculăm PSNR (255 este valoarea maximă a unui pixel)
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_capacity(image):
    """
    Calculează capacitatea maximă de stocare pentru metoda clasică LSB.
    
    Formula: (înălțime × lățime × 3 canale) / 8 biți_per_caracter - 10 (rezervat delimitator)
    
    Returnează numărul maxim de caractere care pot fi ascunse.
    """
    arr = np.array(image)
    return (arr.shape[0] * arr.shape[1] * 3) // 8 - 10


def adaptive_capacity(image, threshold_pct=50):
    """
    Calculează câte caractere pot fi ascunse cu metoda LSB Adaptivă,
    folosind doar cei mai complecși 50% dintre pixeli.
    
    Parametru threshold_pct: procentul pixelilor utilizați (implicit 50%)
    → mai mare = mai multă capacitate, dar mai ușor de detectat
    
    Returnează numărul de caractere pentru metoda adaptivă.
    """
    arr = np.array(image)
    mag = compute_texture_map(arr)
    flat = mag.flatten()

    # Calculăm pragul: valoarea care separă top X% cei mai complecși pixeli
    threshold = np.percentile(flat, 100 - threshold_pct)

    # Numărăm pixelii cu complexitate suficientă
    usable_pixels = int(np.sum(flat >= threshold))

    # Capacitate: pixeli_utilizabili × 3 canale / 8 biți_per_caracter - rezervat delimitator
    return usable_pixels * 3 // 8 - 10


# ─────────────────────────────────────────────
#  FUNCȚII DE VIZUALIZARE
# ─────────────────────────────────────────────

def make_texture_heatmap(image):
    """
    Generează un grafic cu 3 panouri pentru vizualizarea hărții de textură:
    1. Imaginea originală
    2. Harta de textură (Laplacian) — zonele roșii sunt cele mai complexe
    3. Imaginea cu zonele utilizate evidențiate (zonele nefolosite sunt întunecare)
    
    Returnează un obiect matplotlib Figure gata de afișat.
    """
    arr = np.array(image)
    mag = compute_texture_map(arr)

    # Creăm figura cu 3 subplot-uri alăturate
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0e1117')  # Fundal întunecat (tema Streamlit)

    # Panoul 1: imaginea originală
    axes[0].imshow(image)
    axes[0].set_title("Original Image", color='white', fontsize=11)
    axes[0].axis('off')

    # Panoul 2: harta de textură (gradient de culori — roșu = complex, negru = simplu)
    im = axes[1].imshow(mag, cmap='hot', interpolation='nearest')
    axes[1].set_title("Texture / Edge Map (Laplacian)", color='white', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])  # Adăugăm bara de culori pentru interpretare

    # Panoul 3: evidențiem pixelii care vor fi folosiți pentru ascundere
    threshold = np.percentile(mag, 50)  # Pragul: top 50% cei mai complecși
    mask      = mag >= threshold        # Mască: True = pixel utilizabil
    overlay   = arr.copy()
    # Întunecem pixelii care NU vor fi utilizați (la 25% din luminozitate)
    overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Pixels used for hiding (bright)", color='white', fontsize=11)
    axes[2].axis('off')

    for ax in axes:
        ax.tick_params(colors='white')

    plt.tight_layout()
    return fig


def make_comparison_chart(original, encoded):
    """
    Generează un grafic detaliat de comparație între imaginea originală
    și cea codificată (stego).
    
    Conține:
    - Rândul de sus: original | codificată | diferența amplificată (×50)
    - Rândul de jos: histograma canalelor R, G, B (original vs. codificată)
    
    Histograma identică pentru ambele versiuni = steganografie imperceptibilă.
    Diferența amplificată ×50 face vizibile modificările altfel invizibile.
    
    Returnează un obiect matplotlib Figure.
    """
    orig_arr = np.array(original).astype(np.float64)
    enc_arr  = np.array(encoded).astype(np.float64)
    diff     = np.abs(orig_arr - enc_arr)  # Diferența pixel cu pixel

    # Creăm layout-ul figurii: 2 rânduri, 3 coloane (ultimul rând se întinde pe toate)
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#0e1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Imaginea originală
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title("Original", color='white')
    ax1.axis('off')

    # Imaginea codificată
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(encoded)
    ax2.set_title("Encoded", color='white')
    ax2.axis('off')

    # Diferența amplificată ×50 (altfel ar fi aproape negru total)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(diff.astype(np.uint8) * 50, cmap='hot')
    ax3.set_title("Difference (×50)", color='white')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3)

    # Histograma canalelor de culoare (rândul 2, întins pe toate 3 coloanele)
    colors = ['red', 'green', 'blue']
    labels = ['Red channel', 'Green channel', 'Blue channel']
    ax4 = fig.add_subplot(gs[1, :])

    for ch, (col, lbl) in enumerate(zip(colors, labels)):
        # Histograma originalului (linie întreruptă, semi-transparentă)
        hist_orig, bins = np.histogram(orig_arr[:, :, ch], bins=64, range=(0, 255))
        # Histograma imaginii codificate (linie plină)
        hist_enc, _     = np.histogram(enc_arr[:, :, ch],  bins=64, range=(0, 255))
        ax4.plot(bins[:-1], hist_orig, color=col, alpha=0.5, linestyle='--', label=f"{lbl} original")
        ax4.plot(bins[:-1], hist_enc,  color=col, alpha=1.0,                 label=f"{lbl} encoded")

    ax4.set_title("Pixel Histogram Comparison (dashed = original)", color='white')
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white')
    ax4.legend(fontsize=7, ncol=2)
    for sp in ax4.spines.values():
        sp.set_color('gray')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  STEGANALIZA: testul chi-pătrat pe LSB-uri
# ─────────────────────────────────────────────

def chi_square_test(image):
    """
    Implementare simplificată a atacului chi-pătrat pentru detecția steganografiei.
    
    Principiu:
    Într-o imagine naturală (fără steganografie), valorile pare și impare
    ale pixelilor sunt distribuite aproximativ egal în mod natural.
    
    Steganografia LSB CLASICĂ (secvențială) face această distribuție
    PREA uniformă (valorile pare ≈ valorile impare pentru fiecare pereche),
    ceea ce este statistic nenatural.
    
    Testul evaluează fiecare canal (R, G, B) separat:
    - Score normalizat < 1.5 → distribuție suspectă (posibil steganografie clasică)
    - Score normalizat ≥ 1.5 → distribuție naturală (curat sau LSB adaptiv)
    
    Metoda LSB Adaptivă este rezistentă la acest test deoarece modifică
    doar pixelii din zonele de textură, nu secvențial.
    
    Returnează: (rezultate_per_canal, verdict_general, canale_suspecte)
    """
    arr     = np.array(image)
    results = {}    # Rezultatele per canal
    suspects = []   # Lista canalelor suspecte

    for ch, name in enumerate(['Red', 'Green', 'Blue']):
        channel = arr[:, :, ch].flatten().astype(np.int32)

        pairs_expected = []  # Valorile așteptate (distribuție uniformă)
        pairs_observed = []  # Valorile observate în realitate

        # Analizăm perechile de valori (0,1), (2,3), (4,5), ..., (254,255)
        # Teoria: dacă LSB e random (steganografie), atunci n(v) ≈ n(v+1)
        for v in range(0, 255, 2):
            n0    = int(np.sum(channel == v))      # Câți pixeli au valoarea exactă v
            n1    = int(np.sum(channel == v + 1))  # Câți pixeli au valoarea v+1
            total = n0 + n1

            if total > 0:
                # Dacă ar fi perfect random: ne-am aștepta la total/2 pentru fiecare
                expected = total / 2
                pairs_expected.append(expected)
                pairs_observed.append(n0)

        pe = np.array(pairs_expected, dtype=np.float64)
        po = np.array(pairs_observed, dtype=np.float64)

        # Formula chi-pătrat: Σ (observat - așteptat)² / așteptat
        # Adăugăm 1e-9 la numitor pentru a evita împărțirea la zero
        chi2 = float(np.sum((po - pe) ** 2 / (pe + 1e-9)))

        # Normalizăm scorul la numărul de grade de libertate
        dof  = len(pe) - 1
        norm = chi2 / max(dof, 1)

        results[name] = {'chi2': chi2, 'normalized': norm}

        # Un scor normalizat mic înseamnă distribuție prea uniformă = suspect
        if norm < 1.5:
            suspects.append(name)

    # Verdictul general: suspect dacă cel puțin 2 din 3 canale sunt suspecte
    overall = len(suspects) >= 2
    return results, overall, suspects


# ─────────────────────────────────────────────
#  APLICAȚIA STREAMLIT — CONFIGURARE
# ─────────────────────────────────────────────

# Configurăm pagina web (titlu, icon, layout)
st.set_page_config(
    page_title="SteganoPy – LSB Adaptiv",
    page_icon="🔐",
    layout="wide"  # Layout extins pe toată lățimea ecranului
)

# Titlul și descrierea principală a aplicației
st.title("🔐 SteganoPy — Steganografie LSB Adaptivă")
st.markdown(
    "Ascunde mesaje **doar în zonele cu textură bogată** ale unei imagini — "
    "mult mai greu de detectat statistic față de LSB clasic."
)

# Creăm cele 8 tab-uri ale aplicației
tabs = st.tabs([
    "🧠 Codificare adaptivă",    # Tab 1: Codificare adaptivă
    "🔍 Decodificare adaptivă",    # Tab 2: Decodificare adaptivă
    "📊 Criteri de calitate",    # Tab 3: Metrici de calitate
    "🔬 Steganaliza",       # Tab 4: Steganaliza
    "📝 Codificare clasică",     # Tab 5: Codificare clasică
    "🔓 Decodificare clasică",     # Tab 6: Decodificare clasică
    "📁 Ascundere/extragere fișiere",  # Tab 7: Ascundere/extragere fișiere
    "📦 Procesare în masă",              # Tab 8: Procesare în masă
])
tab_aenc, tab_adec, tab_metrics, tab_steg, tab_cenc, tab_cdec, tab_file, tab_batch = tabs

#-----------------------------------------------
#  TAB 1 — CODIFICARE ADAPTIVĂ
# ----------------------------------------------
with tab_aenc:
    st.header("🧠 LSB Adaptiv — Codificare")
    st.info(
        "Această metodă analizează imaginea mai întâi cu un **filtru Laplacian**, "
        "apoi ascunde biți **doar în pixelii cu textura cea mai complexă**. "
        "Zonele uniforme rămân neatinse — imaginea stego este mult mai greu "
        "de detectat statistic."
    )

    # Widget pentru încărcarea imaginii
    up = st.file_uploader("Încarcă imaginea copertă", type=['png','jpg','jpeg','bmp','tiff','webp'], key="aenc_img")

    if up:
        # Deschidem imaginea și o convertim la RGB (elimină canalul alpha dacă există)
        image = Image.open(up).convert('RGB')
        col1, col2 = st.columns(2)  # Împărțim interfața în 2 coloane

        with col1:
            st.subheader("Imaginea originală")
            st.image(image, use_container_width=True)

            # Afișăm capacitatea de stocare pentru ambele metode
            cap_classic  = calculate_capacity(image)
            cap_adaptive = adaptive_capacity(image)
            st.metric("Capacitate LSB clasic",  f"{cap_classic:,} caractere")
            st.metric("Capacitate LSB adaptiv", f"{cap_adaptive:,} caractere", help="Doar pixelii cu textură utilizabilă")

        # Slider pentru numărul de biți per canal (1 = invizibil, 3 = mai mult spațiu dar vizibil)
        bits_per_ch = st.select_slider(
            "Biți per canal (1 = imperceptibil, 2 = capacitate 2×, 3 = zgomot ușor)",
            options=[1, 2, 3], value=1
        )

        # Textbox pentru mesajul secret
        message = st.text_area("Mesaj secret:", height=130, placeholder="Scrie mesajul tău secret aici…")

        # Checkbox și câmp pentru parola de criptare
        use_enc  = st.checkbox("🔐 Criptare AES-256", value=True)
        password = None
        if use_enc:
            password = st.text_input("Parolă:", type="password", key="aenc_pass")

        # Bara de progres care arată procentul din capacitate utilizat de mesaj
        if message:
            estimated_bits = len(text_to_binary(message + "###END###"))

            # Dacă criptăm, mesajul va fi mai lung (adăugăm overhead-ul de criptare)
            if use_enc and password:
                enc_preview    = encrypt_message(message, password)
                estimated_bits = len(text_to_binary(enc_preview + "###END###"))

            capacity_bits = adaptive_capacity(image) * 8

            if estimated_bits > capacity_bits:
                st.error("⚠️ Mesajul este prea lung pentru capacitatea de textură a acestei imagini.")
            else:
                # Afișăm procentul utilizat
                pct = estimated_bits / max(capacity_bits, 1) * 100
                st.progress(min(pct/100, 1.0), text=f"Capacitate utilizată: {pct:.1f}%")

        # Butonul de codificare (dezactivat dacă lipsesc mesajul sau parola)
        disabled = not message or (use_enc and not password)
        if st.button("🔒 Codifică adaptiv", type="primary", disabled=disabled, key="aenc_btn"):
            try:
                with st.spinner("Analizez harta de textură și incorporez mesajul…"):
                    # Apelăm funcția principală de codificare
                    encoded = adaptive_encode(image, message,
                                              password if use_enc else None,
                                              bits_per_channel=bits_per_ch)

                # Salvăm perechea original/codificat în sesiune pentru tab-ul Metrics
                st.session_state['last_original'] = image
                st.session_state['last_encoded']  = encoded

                with col2:
                    st.subheader("Imaginea stego")
                    st.image(encoded, use_container_width=True)

                    # Calculăm și afișăm PSNR
                    psnr = calculate_psnr(image, encoded)
                    st.metric("PSNR", f"{psnr:.2f} dB", help=">40 dB = imperceptibil vizual")

                    # Buton de descărcare pentru imaginea codificată
                    buf = io.BytesIO()
                    encoded.save(buf, format='PNG')
                    st.download_button("⬇️ Descarcă imaginea stego", buf.getvalue(),
                                       "stego_adaptiv.png", "image/png")
                    st.success("✅ Mesajul a fost ascuns cu succes!")

            except Exception as e:
                st.error(f"❌ {e}")

        # Buton separat pentru previzualizarea hărții de textură
        if st.button("🗺️ Arată harta de textură", key="aenc_heatmap"):
            with st.spinner("Calculez harta de textură…"):
                fig = make_texture_heatmap(image)
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════
#  TAB 2 — DECODIFICARE ADAPTIVĂ
# ══════════════════════════════════════════════
with tab_adec:
    st.header("🔍 LSB Adaptiv — Decodificare")

    up2 = st.file_uploader("Încarcă imaginea stego", type=['png','jpg','jpeg','bmp','tiff','webp'], key="adec_img")

    if up2:
        stego = Image.open(up2).convert('RGB')
        st.image(stego, width=400)

        # Trebuie să folosim exact aceleași setări ca la codificare!
        bits_per_ch2 = st.select_slider(
            "Biți per canal (trebuie să corespundă cu setarea de la codificare)",
            options=[1, 2, 3], value=1, key="adec_bpc"
        )

        # Checkbox și câmp pentru parola (dacă mesajul a fost criptat)
        use_dec  = st.checkbox("🔐 Mesajul este criptat", value=True, key="adec_enc")
        password2 = None
        if use_dec:
            password2 = st.text_input("Parolă:", type="password", key="adec_pass")

        # Butonul de decodificare
        if st.button("🔓 Decodifică adaptiv", type="primary",
                     disabled=(use_dec and not password2), key="adec_btn"):
            try:
                with st.spinner("Reconstruiesc ordinea pixelilor și extrag biții…"):
                    msg = adaptive_decode(stego,
                                         password2 if use_dec else None,
                                         bits_per_channel=bits_per_ch2)

                st.success("✅ Mesaj găsit!")
                st.text_area("Mesaj ascuns:", value=msg, height=150, disabled=True)
                st.info(f"Lungime: {len(msg)} caractere")

            except Exception as e:
                st.error(f"❌ {e}")


# ══════════════════════════════════════════════
#  TAB 3 — METRICI DE CALITATE
# ══════════════════════════════════════════════
with tab_metrics:
    st.header("📊 Metrici de calitate & Comparație vizuală")
    st.info("Compară imaginea originală cu imaginea stego pentru a măsura imperceptibilitatea.")

    # Două coloane pentru încărcarea celor două imagini
    col_a, col_b = st.columns(2)
    with col_a:
        orig_up = st.file_uploader("Încarcă imaginea ORIGINALĂ", type=['png','jpg','jpeg','bmp'], key="met_orig")
    with col_b:
        enc_up  = st.file_uploader("Încarcă imaginea STEGO (codificată)", type=['png','jpg','jpeg','bmp'], key="met_enc")

    # Opțiune rapidă: folosim ultimul rezultat din Tab 1 fără a reîncărca imagini
    if 'last_original' in st.session_state and 'last_encoded' in st.session_state:
        if st.button("📥 Folosește ultima pereche din tab-ul Codificare adaptivă"):
            orig_up = None  # Resetăm, va folosi datele din sesiune

    # Determinăm sursa imaginilor: sesiunea sau fișierele încărcate manual
    use_session = ('last_original' in st.session_state and 'last_encoded' in st.session_state
                   and not orig_up and not enc_up)

    if use_session:
        orig_img = st.session_state['last_original']
        enc_img  = st.session_state['last_encoded']
        st.success("Se folosesc rezultatele ultimei codificări adaptive.")
    elif orig_up and enc_up:
        orig_img = Image.open(orig_up).convert('RGB')
        enc_img  = Image.open(enc_up).convert('RGB')
    else:
        orig_img = enc_img = None  # Nu avem încă ambele imagini

    if orig_img and enc_img:
        # Calculăm metricile de calitate
        psnr     = calculate_psnr(orig_img, enc_img)
        orig_arr = np.array(orig_img)
        enc_arr  = np.array(enc_img)
        diff     = np.abs(orig_arr.astype(int) - enc_arr.astype(int))  # Diferența pixel cu pixel

        # Afișăm cele 4 metrici principale în coloane
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR", f"{psnr:.2f} dB")
        m2.metric("Diferență maximă per pixel", f"{diff.max()}")
        m3.metric("Diferență medie per pixel", f"{diff.mean():.4f}")
        changed = int(np.sum(np.any(diff > 0, axis=2)))
        m4.metric("Pixeli modificați", f"{changed:,}")

        # Verdictul calitativ bazat pe PSNR
        verdict = "✅ Excelent — imperceptibil vizual" if psnr > 45 else \
                  "✅ Bun — imperceptibil pentru ochiul uman" if psnr > 40 else \
                  "⚠️ Acceptabil, dar poate prezenta zgomot minor" if psnr > 35 else \
                  "❌ Degradare vizibilă"
        st.info(f"Verdict calitate: **{verdict}**")

        # Generăm și afișăm graficele de comparație
        with st.spinner("Generez graficele de comparație…"):
            fig = make_comparison_chart(orig_img, enc_img)
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════
#  TAB 4 — STEGANALIZA
# ══════════════════════════════════════════════
with tab_steg:
    st.header("🔬 Steganaliza — Atacul chi-pătrat")
    st.info(
        "**Atacul chi-pătrat** testează dacă planul LSB al unei imagini pare "
        "'prea aleatoriu' — un semn caracteristic al steganografiei LSB secvențiale. "
        "LSB Adaptiv distribuie biții în zonele texturate, făcând detectarea mai dificilă."
    )

    sa_up = st.file_uploader("Încarcă imaginea de analizat", type=['png','jpg','jpeg','bmp'], key="steg_up")

    if sa_up:
        sa_img = Image.open(sa_up).convert('RGB')
        st.image(sa_img, width=400)

        if st.button("🔬 Rulează analiza chi-pătrat", type="primary", key="steg_btn"):
            with st.spinner("Rulez analiza statistică…"):
                results, suspected, suspect_channels = chi_square_test(sa_img)

            # Afișăm rezultatele per canal în 3 coloane
            st.subheader("Rezultate per canal")
            cols = st.columns(3)
            for i, (ch, data) in enumerate(results.items()):
                with cols[i]:
                    norm  = data['normalized']
                    label = "🟢 Curat" if norm >= 1.5 else "🔴 Suspect"
                    st.metric(f"Canal {ch} χ² (norm)", f"{norm:.3f}", label)

            # Afișăm verdictul general
            if suspected:
                st.error(
                    f"⚠️ **Date ascunse suspectate!** "
                    f"Canale cu anomalie: {', '.join(suspect_channels)}. "
                    "Consistent cu steganografia LSB clasică secvențială."
                )
            else:
                st.success(
                    "✅ **Nicio steganografie detectată** prin testul chi-pătrat. "
                    "Distribuția LSB pare naturală — consistent cu LSB Adaptiv "
                    "sau o imagine curată."
                )

            arr  = np.array(sa_img)
            fig2, axs = plt.subplots(1, 3, figsize=(12, 4))
            fig2.patch.set_facecolor('#0e1117')
            titles = ['Planul LSB Roșu', 'Planul LSB Verde', 'Planul LSB Albastru']

            for ch, (ax, title) in enumerate(zip(axs, titles)):
                lsb = (arr[:, :, ch] & 1) * 255
                ax.imshow(lsb, cmap='gray')
                ax.set_title(title, color='white')
                ax.axis('off')

            plt.suptitle("Planele LSB (aleatoriu = curat, structurat = date ascunse)",
                         color='white', y=1.01)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)


# ══════════════════════════════════════════════
#  TAB 5 — CODIFICARE CLASICĂ
# ══════════════════════════════════════════════
with tab_cenc:
    st.header("📝 LSB Clasic — Codificare")
    st.warning("LSB clasic scrie biții secvențial din pixelul (0,0) — detectabil prin testul chi-pătrat. Folosește tab-ul Adaptiv pentru securitate mai bună.")

    up_c = st.file_uploader("Încarcă imaginea copertă", type=['png','jpg','jpeg','bmp','tiff','webp'], key="cenc_img")

    if up_c:
        img_c = Image.open(up_c).convert('RGB')
        col1c, col2c = st.columns(2)

        with col1c:
            st.image(img_c, use_container_width=True)
            st.info(f"Capacitate: {calculate_capacity(img_c):,} caractere")

        msg_c  = st.text_area("Mesaj secret:", height=120, key="cenc_msg")
        use_ec = st.checkbox("🔐 Criptare AES-256", key="cenc_enc")
        pass_c = st.text_input("Parolă:", type="password", key="cenc_pass") if use_ec else None

        if st.button("🔒 Codifică clasic", type="primary",
                     disabled=(not msg_c or (use_ec and not pass_c)), key="cenc_btn"):
            try:
                # Codificăm cu metoda clasică (secvențială)
                enc_c = encode_message_classic(img_c, msg_c, pass_c if use_ec else None)
                with col2c:
                    st.image(enc_c, use_container_width=True)
                    buf = io.BytesIO()
                    enc_c.save(buf, format='PNG')
                    st.download_button("⬇️ Descarcă", buf.getvalue(), "stego_clasic.png", "image/png")
                    st.metric("PSNR", f"{calculate_psnr(img_c, enc_c):.2f} dB")
            except Exception as e:
                st.error(f"❌ {e}")


# ══════════════════════════════════════════════
#  TAB 6 — DECODIFICARE CLASICĂ
# ══════════════════════════════════════════════
with tab_cdec:
    st.header("🔓 LSB Clasic — Decodificare")

    up_cd = st.file_uploader("Încarcă imaginea stego", type=['png','jpg','jpeg','bmp','tiff','webp'], key="cdec_img")

    if up_cd:
        img_cd = Image.open(up_cd).convert('RGB')
        st.image(img_cd, width=400)

        use_dcd = st.checkbox("🔐 Mesajul este criptat", key="cdec_enc")
        pass_cd = st.text_input("Parolă:", type="password", key="cdec_pass") if use_dcd else None

        if st.button("🔓 Decodifică clasic", type="primary",
                     disabled=(use_dcd and not pass_cd), key="cdec_btn"):
            try:
                # Extragem mesajul cu metoda clasică
                msg_cd = decode_message_classic(img_cd, pass_cd if use_dcd else None)
                st.success("✅ Mesaj găsit!")
                st.text_area("Mesaj:", value=msg_cd, height=150, disabled=True)
            except Exception as e:
                st.error(f"❌ {e}")


# ══════════════════════════════════════════════
#  TAB 7 — ASCUNDERE / EXTRAGERE FIȘIERE
# ══════════════════════════════════════════════
with tab_file:
    st.header("📁 Ascunde / Extrage un fișier")

    # Două sub-tab-uri: Ascunde și Extrage
    ft1, ft2 = st.tabs(["Ascunde fișier", "Extrage fișier"])

    with ft1:
        # Încărcăm imaginea "copertă" și fișierul de ascuns
        fi_up = st.file_uploader("Imagine copertă", type=['png','jpg','jpeg','bmp','tiff','webp'], key="fenc_img")
        fh_up = st.file_uploader("Fișier de ascuns (orice tip)", type=None, key="fenc_file")

        if fi_up and fh_up:
            fi_img = Image.open(fi_up).convert('RGB')
            fb     = fh_up.read()

            cap_f = (np.array(fi_img).shape[0] * np.array(fi_img).shape[1] * 3) // 8 - 100

            col1f, col2f = st.columns(2)
            with col1f:
                st.image(fi_img, use_container_width=True)
                st.info(f"Capacitate: {cap_f:,} bytes | Fișier: {len(fb):,} bytes")
                if len(fb) > cap_f:
                    st.error("Fișierul este prea mare!")

            if st.button("🔒 Ascunde fișierul", type="primary", disabled=len(fb) > cap_f, key="fenc_btn"):
                try:
                    enc_fi = encode_file(fi_img, fb, fh_up.name)
                    with col2f:
                        st.image(enc_fi, use_container_width=True)
                        buf = io.BytesIO()
                        enc_fi.save(buf, format='PNG')
                        st.download_button("⬇️ Descarcă", buf.getvalue(),
                                           f"stego_fisier_{fh_up.name}.png", "image/png")
                        st.success("✅ Fișier ascuns cu succes!")
                except Exception as e:
                    st.error(f"❌ {e}")

    with ft2:
        fd_up = st.file_uploader("Imagine stego cu fișier ascuns", type=['png','jpg','jpeg','bmp','tiff','webp'], key="fdec_img")

        if fd_up:
            fd_img = Image.open(fd_up).convert('RGB')
            st.image(fd_img, width=400)

            if st.button("🔓 Extrage fișierul", type="primary", key="fdec_btn"):
                try:
                    data, fname = decode_file(fd_img)
                    st.success(f"✅ Extras: {fname} ({len(data):,} bytes)")
                    st.download_button(f"⬇️ Descarcă {fname}", data, fname, "application/octet-stream")
                except Exception as e:
                    st.error(f"❌ {e}")


# ══════════════════════════════════════════════
#  TAB 8 — PROCESARE ÎN MASĂ (BATCH)
# ══════════════════════════════════════════════
with tab_batch:
    st.header("📦 Procesare în masă")
    st.info("Ascunde același mesaj în mai multe imagini simultan — folosește LSB Adaptiv.")

    # Permitem încărcarea mai multor imagini simultan
    batch_imgs = st.file_uploader("Încarcă imagini", type=['png','jpg','jpeg','bmp','tiff','webp'],
                                   accept_multiple_files=True, key="batch_imgs")

    if batch_imgs:
        st.success(f"{len(batch_imgs)} imagini încărcate")

        b_msg  = st.text_area("Mesaj de ascuns în toate imaginile:", height=100, key="b_msg")
        b_enc  = st.checkbox("🔐 Criptare AES-256", key="b_enc")
        b_pass = st.text_input("Parolă:", type="password", key="b_pass") if b_enc else None
        b_bpc  = st.select_slider("Biți per canal", options=[1,2,3], value=1, key="b_bpc")

        disabled_b = not b_msg or (b_enc and not b_pass)

        if st.button("🔒 Codifică toate", type="primary", disabled=disabled_b, key="batch_btn"):
            progress = st.progress(0)   # Bara de progres (0-100%)
            zip_buf  = io.BytesIO()     # Buffer pentru arhiva ZIP în memorie
            ok = err = 0                # Contoare pentru imagini reușite/eșuate

            # Creăm arhiva ZIP direct în memorie (fără fișiere temporare pe disc)
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for idx, img_f in enumerate(batch_imgs):
                    try:
                        img_f.seek(0)  # Resetăm cursorul fișierului la început
                        img_b = Image.open(img_f).convert('RGB')

                        # Codificăm imaginea cu LSB Adaptiv
                        enc_b = adaptive_encode(img_b, b_msg, b_pass if b_enc else None, b_bpc)

                        # Salvăm imaginea codificată în buffer
                        ibuf = io.BytesIO()
                        enc_b.save(ibuf, format='PNG')

                        # Adăugăm fișierul PNG în arhiva ZIP (fără extensia originală)
                        base = os.path.splitext(img_f.name)[0]
                        zf.writestr(f"{base}_stego_adaptiv.png", ibuf.getvalue())
                        ok += 1

                    except Exception as e:
                        st.warning(f"⚠️ {img_f.name}: {e}")  # Raportăm erorile per imagine
                        err += 1

                    # Actualizăm bara de progres
                    progress.progress((idx + 1) / len(batch_imgs))

            progress.empty()  # Ascundem bara de progres după finalizare

            if ok:
                # Generăm un timestamp pentru numele unic al arhivei
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_buf.seek(0)  # Resetăm cursorul arhivei la început
                st.success(f"✅ {ok} imagini codificate ({err} eșuate)")
                st.download_button("⬇️ Descarcă ZIP", zip_buf.getvalue(),
                                   f"batch_stego_{ts}.zip", "application/zip")


# ─────────────────────────────────────────────
#  FOOTER — Explicație tehnică
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### 📖 Cum funcționează LSB Adaptiv

**LSB Clasic** scrie biții secvențial începând cu pixelul (0,0).  
Aceasta creează o anomalie statistică în planul LSB, detectabilă prin **testul chi-pătrat**.

**LSB Adaptiv** (această implementare):
1. Aplică un **filtru Laplacian** pentru a calcula un scor de complexitate per pixel.
2. Sortează toți pixelii de la cel mai complex la cel mai simplu.
3. Înglobează biți **doar în cei mai complecși pixeli** — muchii, colțuri, zgomot.
4. Zonele uniforme (cer, pereți) rămân complet neatinse.
5. Opțional, selecția pixelilor este **pseudo-randomizată folosind parola** ca sămânță.

Avantaje:
- 🔬 **Rezistent statistic** — distribuția LSB în zonele uniforme pare naturală.
- 👁️ **Optim perceptual** — modificările cad acolo unde ochiul uman este cel mai puțin sensibil.
- 🔐 **Dublu securizat** când e combinat cu criptare AES-256.

**PSNR > 40 dB** este considerat imperceptibil vizual. LSB Adaptiv atinge tipic 48–55 dB.
""")
