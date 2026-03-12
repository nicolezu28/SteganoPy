import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import zipfile
from datetime import datetime

def text_to_binary(text):
    #Convertește text în binary string
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def binary_to_text(binary):
    """Convertește binary string în text"""
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

def encrypt_message(message, password):
    """Criptează mesajul folosind AES-256"""
    key = hashlib.sha256(password.encode()).digest()
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()
    
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    
    return base64.b64encode(iv + encrypted).decode('utf-8')

def decrypt_message(encrypted_message, password):
    """Decriptează mesajul folosind AES-256"""
    key = hashlib.sha256(password.encode()).digest()
    
    encrypted_data = base64.b64decode(encrypted_message)
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    
    return data.decode('utf-8')

def encode_message(image, message, password=None):
    """Encodează mesajul în imagine folosind LSB"""
    img_array = np.array(image)
    
    if password:
        message = encrypt_message(message, password)
    
    delimiter = "###END###"
    full_message = message + delimiter
    binary_message = text_to_binary(full_message)
    
    max_bytes = img_array.shape[0] * img_array.shape[1] * 3
    
    if len(binary_message) > max_bytes:
        raise ValueError(f"Mesajul este prea lung! Maximum {max_bytes // 8} caractere.")
    
    flat_image = img_array.flatten()
    
    for i in range(len(binary_message)):
        flat_image[i] = (flat_image[i] & 0xFE) | int(binary_message[i])
    
    encoded_image = flat_image.reshape(img_array.shape)
    
    return Image.fromarray(encoded_image.astype('uint8'))

def decode_message(image, password=None):
    """Decodează mesajul din imagine"""
    img_array = np.array(image)
    flat_image = img_array.flatten()
    
    binary_message = ''
    delimiter = "###END###"
    delimiter_binary = text_to_binary(delimiter)
    
    for i in range(len(flat_image)):
        binary_message += str(flat_image[i] & 1)
        
        if len(binary_message) >= len(delimiter_binary):
            if binary_message[-len(delimiter_binary):] == delimiter_binary:
                break
    
    if delimiter_binary not in binary_message:
        raise ValueError("Nu s-a găsit niciun mesaj ascuns în această imagine!")
    
    binary_message = binary_message[:-len(delimiter_binary)]
    
    message = binary_to_text(binary_message)
    
    if password:
        try:
            message = decrypt_message(message, password)
        except Exception:
            raise ValueError("Parolă incorectă sau mesajul nu este criptat!")
    
    return message

def calculate_capacity(image):
    """Calculează capacitatea maximă de caractere"""
    img_array = np.array(image)
    max_chars = (img_array.shape[0] * img_array.shape[1] * 3) // 8 - 10
    return max_chars

def bytes_to_binary(data):
    """Convertește bytes în binary string"""
    binary = ''.join(format(byte, '08b') for byte in data)
    return binary

def binary_to_bytes(binary):
    """Convertește binary string în bytes"""
    byte_array = bytearray()
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            byte_array.append(int(byte, 2))
    return bytes(byte_array)

def encode_file(image, file_data, filename):
    """Encodează un fișier binar în imagine"""
    img_array = np.array(image)
    
    file_info = f"{filename}|||{len(file_data)}"
    file_info_binary = text_to_binary(file_info)
    
    delimiter = "###FILEEND###"
    delimiter_binary = text_to_binary(delimiter)
    
    file_binary = bytes_to_binary(file_data)
    
    full_binary = file_info_binary + delimiter_binary + file_binary + delimiter_binary
    
    max_bytes = img_array.shape[0] * img_array.shape[1] * 3
    
    if len(full_binary) > max_bytes:
        raise ValueError(f"Fișierul este prea mare! Maximum ~{max_bytes // 8} bytes.")
    
    flat_image = img_array.flatten()
    
    for i in range(len(full_binary)):
        flat_image[i] = (flat_image[i] & 0xFE) | int(full_binary[i])
    
    encoded_image = flat_image.reshape(img_array.shape)
    
    return Image.fromarray(encoded_image.astype('uint8'))

def decode_file(image):
    """Decodează un fișier binar din imagine"""
    img_array = np.array(image)
    flat_image = img_array.flatten()
    
    binary_data = ''
    delimiter = "###FILEEND###"
    delimiter_binary = text_to_binary(delimiter)
    
    first_delimiter_found = False
    delimiter_count = 0
    
    for i in range(len(flat_image)):
        binary_data += str(flat_image[i] & 1)
        
        if len(binary_data) >= len(delimiter_binary):
            if binary_data[-len(delimiter_binary):] == delimiter_binary:
                delimiter_count += 1
                if delimiter_count == 2:
                    break
    
    if delimiter_count < 2:
        raise ValueError("Nu s-a găsit niciun fișier ascuns în această imagine sau datele sunt incomplete!")
    
    info_end = binary_data.index(delimiter_binary)
    file_info_binary = binary_data[:info_end]
    file_info = binary_to_text(file_info_binary)
    
    parts = file_info.split('|||')
    if len(parts) != 2:
        raise ValueError("Format invalid de fișier!")
    
    filename = parts[0]
    try:
        file_size = int(parts[1])
    except ValueError:
        raise ValueError("Dimensiune fișier invalidă în metadate!")
    
    start_pos = info_end + len(delimiter_binary)
    end_pos = len(binary_data) - len(delimiter_binary)
    file_binary = binary_data[start_pos:end_pos]
    
    expected_bits = file_size * 8
    if len(file_binary) < expected_bits:
        raise ValueError(f"Date incomplete! Așteptat {expected_bits} biți, găsit doar {len(file_binary)} biți.")
    
    file_binary = file_binary[:expected_bits]
    file_data = binary_to_bytes(file_binary)
    
    if len(file_data) != file_size:
        raise ValueError(f"Dimensiunea fișierului extras ({len(file_data)} bytes) nu corespunde cu metadatele ({file_size} bytes)!")
    
    return file_data, filename

st.set_page_config(
    page_title="Steganografie - Ascunde Mesaje în Imagini",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Aplicație de Steganografie")
st.markdown("**Ascunde mesaje secrete și fișiere în imagini**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Encodare Mesaj", "🔍 Decodare Mesaj", "📁 Encodare Fișier", "📂 Decodare Fișier", "📦 Batch Processing"])

with tab1:
    st.header("Ascunde un mesaj în imagine")
    
    uploaded_file = st.file_uploader(
        "Încarcă imaginea în care vrei să ascunzi mesajul",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
        key="encode_upload"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imaginea originală")
            st.image(image, use_container_width=True)
            
            capacity = calculate_capacity(image)
            st.info(f"📊 Capacitate maximă: ~{capacity} caractere")
        
        message = st.text_area(
            "Introdu mesajul secret pe care vrei să-l ascunzi:",
            height=150,
            placeholder="Scrie mesajul tău secret aici..."
        )
        
        use_encryption = st.checkbox("🔐 Folosește criptare AES (opțional)", value=False)
        password = None
        
        if use_encryption:
            password = st.text_input(
                "Parolă pentru criptare:",
                type="password",
                help="Mesajul va fi criptat cu AES-256 înainte de a fi ascuns în imagine"
            )
        
        if message:
            if len(message) > capacity - 20:
                st.error(f"⚠️ Mesajul este prea lung! Te rog limitează-l la {capacity - 20} caractere.")
            else:
                st.success(f"✅ Lungime mesaj: {len(message)} caractere")
        
        encode_disabled = not message or (use_encryption and not password)
        
        if st.button("🔒 Encodează mesajul", type="primary", disabled=encode_disabled):
            try:
                with st.spinner("Se encodează mesajul..."):
                    encoded_image = encode_message(image, message, password if use_encryption else None)
                
                with col2:
                    st.subheader("Imaginea cu mesaj ascuns")
                    st.image(encoded_image, use_container_width=True)
                    st.success("✅ Mesajul a fost ascuns cu succes!")
                    
                    buf = io.BytesIO()
                    encoded_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="⬇️ Descarcă imaginea",
                        data=byte_im,
                        file_name="imagine_cu_mesaj_ascuns.png",
                        mime="image/png"
                    )
                    
                    st.info("💡 Imaginea arată identic, dar conține mesajul tău secret!")
            
            except ValueError as e:
                st.error(f"❌ Eroare: {str(e)}")
            except Exception as e:
                st.error(f"❌ A apărut o eroare: {str(e)}")

with tab2:
    st.header("Decodează un mesaj ascuns")
    
    uploaded_encoded = st.file_uploader(
        "Încarcă imaginea care conține mesajul ascuns",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
        key="decode_upload"
    )
    
    if uploaded_encoded is not None:
        encoded_image = Image.open(uploaded_encoded)
        
        if encoded_image.mode != 'RGB':
            encoded_image = encoded_image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imaginea încărcată")
            st.image(encoded_image, use_container_width=True)
        
        use_decryption = st.checkbox("🔐 Mesajul este criptat (necesită parolă)", value=False, key="decrypt_check")
        decrypt_password = None
        
        if use_decryption:
            decrypt_password = st.text_input(
                "Parolă pentru decriptare:",
                type="password",
                help="Introdu aceeași parolă folosită la criptare",
                key="decrypt_pass"
            )
        
        decode_disabled = use_decryption and not decrypt_password
        
        if st.button("🔓 Decodează mesajul", type="primary", disabled=decode_disabled):
            try:
                with st.spinner("Se decodează mesajul..."):
                    decoded_message = decode_message(encoded_image, decrypt_password if use_decryption else None)
                
                with col2:
                    st.subheader("Mesajul descoperit")
                    st.success("✅ Mesaj găsit!")
                    st.text_area(
                        "Mesajul secret este:",
                        value=decoded_message,
                        height=150,
                        disabled=True
                    )
                    
                    st.info(f"📝 Lungime mesaj: {len(decoded_message)} caractere")
            
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ A apărut o eroare la decodare: {str(e)}")

with tab3:
    st.header("Ascunde un fișier în imagine")
    
    uploaded_image_for_file = st.file_uploader(
        "Încarcă imaginea în care vrei să ascunzi fișierul",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
        key="file_encode_image"
    )
    
    uploaded_file_to_hide = st.file_uploader(
        "Încarcă fișierul pe care vrei să-l ascunzi",
        type=None,
        key="file_to_hide"
    )
    
    if uploaded_image_for_file is not None and uploaded_file_to_hide is not None:
        image_for_file = Image.open(uploaded_image_for_file)
        
        if image_for_file.mode != 'RGB':
            image_for_file = image_for_file.convert('RGB')
        
        file_bytes = uploaded_file_to_hide.read()
        file_name = uploaded_file_to_hide.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imaginea originală")
            st.image(image_for_file, use_container_width=True)
            
            capacity_bytes = (np.array(image_for_file).shape[0] * np.array(image_for_file).shape[1] * 3) // 8 - 100
            st.info(f"📊 Capacitate maximă: ~{capacity_bytes:,} bytes ({capacity_bytes / 1024:.1f} KB)")
            st.info(f"📄 Fișier: {file_name} ({len(file_bytes):,} bytes)")
            
            if len(file_bytes) > capacity_bytes:
                st.error(f"⚠️ Fișierul este prea mare pentru această imagine!")
            else:
                st.success(f"✅ Fișierul poate fi ascuns în această imagine")
        
        if st.button("🔒 Encodează fișierul", type="primary", disabled=len(file_bytes) > capacity_bytes):
            try:
                with st.spinner("Se encodează fișierul..."):
                    encoded_image_with_file = encode_file(image_for_file, file_bytes, file_name)
                
                with col2:
                    st.subheader("Imaginea cu fișier ascuns")
                    st.image(encoded_image_with_file, use_container_width=True)
                    st.success("✅ Fișierul a fost ascuns cu succes!")
                    
                    buf = io.BytesIO()
                    encoded_image_with_file.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="⬇️ Descarcă imaginea",
                        data=byte_im,
                        file_name=f"imagine_cu_fisier_{file_name}.png",
                        mime="image/png"
                    )
                    
                    st.info("💡 Fișierul este complet ascuns în imagine!")
            
            except ValueError as e:
                st.error(f"❌ Eroare: {str(e)}")
            except Exception as e:
                st.error(f"❌ A apărut o eroare: {str(e)}")

with tab4:
    st.header("Extrage un fișier din imagine")
    
    uploaded_image_with_file = st.file_uploader(
        "Încarcă imaginea care conține fișierul ascuns",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
        key="file_decode_image"
    )
    
    if uploaded_image_with_file is not None:
        image_with_file = Image.open(uploaded_image_with_file)
        
        if image_with_file.mode != 'RGB':
            image_with_file = image_with_file.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imaginea încărcată")
            st.image(image_with_file, use_container_width=True)
        
        if st.button("🔓 Extrage fișierul", type="primary", key="extract_file_btn"):
            try:
                with st.spinner("Se extrage fișierul..."):
                    extracted_file_data, extracted_filename = decode_file(image_with_file)
                
                with col2:
                    st.subheader("Fișier extras")
                    st.success("✅ Fișier găsit!")
                    st.info(f"📄 Nume fișier: {extracted_filename}")
                    st.info(f"📊 Dimensiune: {len(extracted_file_data):,} bytes ({len(extracted_file_data) / 1024:.1f} KB)")
                    
                    st.download_button(
                        label=f"⬇️ Descarcă {extracted_filename}",
                        data=extracted_file_data,
                        file_name=extracted_filename,
                        mime="application/octet-stream"
                    )
            
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ A apărut o eroare la extragere: {str(e)}")

with tab5:
    st.header("Procesare în lot - Encodează mesaj în multiple imagini")
    
    st.info("📦 Încarcă multiple imagini și ascunde același mesaj în toate deodată")
    
    uploaded_images_batch = st.file_uploader(
        "Încarcă imaginile (selectează multiple fișiere)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'],
        accept_multiple_files=True,
        key="batch_images"
    )
    
    if uploaded_images_batch:
        st.success(f"✅ {len(uploaded_images_batch)} imagini încărcate")
        
        batch_message = st.text_area(
            "Introdu mesajul care va fi ascuns în toate imaginile:",
            height=100,
            placeholder="Scrie mesajul tău secret aici...",
            key="batch_message"
        )
        
        use_encryption_batch = st.checkbox("🔐 Folosește criptare AES (opțional)", value=False, key="batch_encrypt")
        batch_password = None
        
        if use_encryption_batch:
            batch_password = st.text_input(
                "Parolă pentru criptare:",
                type="password",
                help="Mesajul va fi criptat cu AES-256 înainte de a fi ascuns",
                key="batch_password"
            )
        
        with st.expander("👁️ Previzualizare imagini încărcate", expanded=False):
            cols = st.columns(min(3, len(uploaded_images_batch)))
            for idx, img_file in enumerate(uploaded_images_batch[:6]):
                with cols[idx % 3]:
                    img = Image.open(img_file)
                    st.image(img, caption=img_file.name, use_container_width=True)
            if len(uploaded_images_batch) > 6:
                st.info(f"... și încă {len(uploaded_images_batch) - 6} imagini")
        
        batch_disabled = not batch_message or (use_encryption_batch and not batch_password)
        
        if st.button("🔒 Procesează toate imaginile", type="primary", disabled=batch_disabled, key="batch_encode_btn"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    success_count = 0
                    error_count = 0
                    
                    for idx, img_file in enumerate(uploaded_images_batch):
                        try:
                            status_text.text(f"Procesare {idx + 1}/{len(uploaded_images_batch)}: {img_file.name}")
                            
                            img_file.seek(0)
                            image = Image.open(img_file)
                            
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            encoded_img = encode_message(
                                image, 
                                batch_message, 
                                batch_password if use_encryption_batch else None
                            )
                            
                            img_buffer = io.BytesIO()
                            encoded_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            base_name = os.path.splitext(img_file.name)[0]
                            zip_file.writestr(
                                f"{base_name}_encoded.png",
                                img_buffer.getvalue()
                            )
                            
                            success_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            st.warning(f"⚠️ Eroare la {img_file.name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_images_batch))
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ {success_count} imagini procesate cu succes!")
                    
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} imagini au eșuat")
                    
                    zip_buffer.seek(0)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label=f"⬇️ Descarcă toate imaginile ({success_count} fișiere)",
                        data=zip_buffer.getvalue(),
                        file_name=f"imagini_encodate_{timestamp}.zip",
                        mime="application/zip"
                    )
                else:
                    st.error("❌ Nicio imagine nu a putut fi procesată!")
                    
            except Exception as e:
                st.error(f"❌ A apărut o eroare: {str(e)}")

st.markdown("---")
st.markdown("""
### 📖 Cum funcționează?

**Steganografia** este arta de a ascunde informații în plain sight. Această aplicație folosește metoda **LSB (Least Significant Bit)**:

- 🔹 **Encodare Mesaje**: Modifică cel mai puțin semnificativ bit din fiecare pixel pentru a stoca text
- 🔹 **Encodare Fișiere**: Ascunde orice tip de fișier (documente, imagini, arhive) în imagini
- 🔹 **Decodare**: Extrage bitii ascunși și reconstruiește conținutul original
- 🔹 **Invizibil**: Schimbările sunt imperceptibile ochiului uman
- 🔐 **Criptare AES-256**: Opțional, mesajele text pot fi criptate pentru securitate maximă

**Formate imagine suportate:**
- PNG, JPG, JPEG, BMP, TIFF, WebP

**Capabilități:**
- Ascunde mesaje text cu sau fără criptare AES-256
- Ascunde orice tip de fișier binar (PDF, ZIP, documente, etc.)
- Procesare în lot: encodează același mesaj în multiple imagini simultan
- Descarcă imaginile modificate și fișierele extrase
- Exportă rezultate batch ca arhivă ZIP

**Sfaturi:**
- Folosește imagini PNG pentru cea mai bună calitate și fără pierdere de date
- Imaginile mai mari pot stoca mai multe date (mesaje lungi sau fișiere mari)
- Activează criptarea pentru mesaje text sensibile
- Verifică capacitatea imaginii înainte de encodare
- Batch processing economisește timp când ai nevoie să ascunzi același mesaj în multe imagini
""")
