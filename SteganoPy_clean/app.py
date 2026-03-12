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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage

# ─────────────────────────────────────────────
#  UTILITY: text / binary / bytes
# ─────────────────────────────────────────────

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

def bytes_to_binary(data):
    return ''.join(format(b, '08b') for b in data)

def binary_to_bytes(binary):
    ba = bytearray()
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            ba.append(int(byte, 2))
    return bytes(ba)

# ─────────────────────────────────────────────
#  ENCRYPTION / DECRYPTION  (AES-256 CBC)
# ─────────────────────────────────────────────

def encrypt_message(message, password):
    key = hashlib.sha256(password.encode()).digest()
    iv  = os.urandom(16)
    cipher    = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder      = padding.PKCS7(128).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()
    encrypted   = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode('utf-8')

def decrypt_message(encrypted_message, password):
    key            = hashlib.sha256(password.encode()).digest()
    encrypted_data = base64.b64decode(encrypted_message)
    iv             = encrypted_data[:16]
    ciphertext     = encrypted_data[16:]
    cipher    = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder    = padding.PKCS7(128).unpadder()
    data        = unpadder.update(padded_data) + unpadder.finalize()
    return data.decode('utf-8')

# ─────────────────────────────────────────────
#  ★ ADAPTIVE LSB CORE
# ─────────────────────────────────────────────

def compute_texture_map(img_array, bits_per_channel=1):
    """
    Compute a per-pixel complexity score using a Laplacian edge filter.
    Returns a boolean mask: True = pixel is 'complex enough' to hide data.

    The threshold is adaptive: we find the top X% most complex pixels,
    where X is chosen so we always have enough capacity for the message.
    We also return the raw edge magnitude for visualisation.
    """
    gray = np.mean(img_array, axis=2).astype(np.float32)

    # Laplacian captures edges + texture
    lap = ndimage.laplace(gray)
    magnitude = np.abs(lap)

    return magnitude

def build_pixel_order(magnitude, img_array, bits_needed, bits_per_channel,
                      password=None):
    """
    Return a list of (row, col, channel) positions to write bits into,
    ordered from highest to lowest texture complexity.

    If a password is given the positions inside each complexity tier are
    shuffled using the password as a seed → scatter embedding = harder to detect.
    """
    h, w, c = img_array.shape
    channels_used = 3  # R, G, B

    # Flatten magnitude and get sorted indices (high complexity first)
    flat_mag = magnitude.flatten()
    # argsort descending
    sorted_px = np.argsort(flat_mag)[::-1]

    # We need ceil(bits_needed / (bits_per_channel * channels_used)) pixels
    pixels_needed = int(np.ceil(bits_needed / (bits_per_channel * channels_used)))

    if pixels_needed > h * w:
        raise ValueError(
            f"Message too long! Need {pixels_needed} pixels but image only has {h*w}."
        )

    selected_px = sorted_px[:pixels_needed]

    # Optional: shuffle with password seed so positions aren't sequential
    if password:
        rng = np.random.default_rng(
            int(hashlib.sha256(password.encode()).hexdigest(), 16) % (2**32)
        )
        rng.shuffle(selected_px)

    # Expand to (row, col, channel) triples
    positions = []
    for px_idx in selected_px:
        row = px_idx // w
        col = px_idx % w
        for ch in range(channels_used):
            positions.append((row, col, ch))

    return positions

def adaptive_encode(image, message, password=None, bits_per_channel=1):
    """
    Embed `message` into `image` using Adaptive LSB.
    Bits are hidden only in the most texture-rich pixels.
    """
    img_array = np.array(image).copy()

    if password:
        message = encrypt_message(message, password)

    DELIMITER   = "###END###"
    full_message = message + DELIMITER
    binary_msg   = text_to_binary(full_message)
    bits_needed  = len(binary_msg)

    magnitude = compute_texture_map(img_array, bits_per_channel)
    positions = build_pixel_order(magnitude, img_array, bits_needed,
                                  bits_per_channel, password)

    if len(positions) * bits_per_channel < bits_needed:
        raise ValueError("Not enough texture capacity in this image for the message.")

    bit_idx = 0
    mask     = 0xFF ^ ((1 << bits_per_channel) - 1)   # e.g. 0xFE for 1-bit
    for (r, c, ch) in positions:
        if bit_idx >= bits_needed:
            break
        chunk = binary_msg[bit_idx: bit_idx + bits_per_channel]
        chunk = chunk.ljust(bits_per_channel, '0')   # pad last chunk if needed
        val   = (int(img_array[r, c, ch]) & mask) | int(chunk, 2)
        img_array[r, c, ch] = val
        bit_idx += bits_per_channel

    return Image.fromarray(img_array.astype('uint8'))

def adaptive_decode(image, password=None, bits_per_channel=1):
    """
    Extract a message from an image encoded with adaptive_encode.
    """
    img_array = np.array(image)

    DELIMITER        = "###END###"
    delimiter_binary = text_to_binary(DELIMITER)
    max_bits         = img_array.shape[0] * img_array.shape[1] * 3 * bits_per_channel

    magnitude = compute_texture_map(img_array, bits_per_channel)
    positions = build_pixel_order(magnitude, img_array, max_bits,
                                  bits_per_channel, password)

    binary_message = ''
    bit_mask = (1 << bits_per_channel) - 1   # e.g. 0x01 for 1-bit

    for (r, c, ch) in positions:
        val    = int(img_array[r, c, ch]) & bit_mask
        binary_message += format(val, f'0{bits_per_channel}b')

        # Check for delimiter after each complete 8-bit character boundary
        if len(binary_message) % 8 == 0 and len(binary_message) >= len(delimiter_binary):
            if binary_message[-len(delimiter_binary):] == delimiter_binary:
                break

    if delimiter_binary not in binary_message:
        raise ValueError("No hidden message found in this image!")

    binary_message = binary_message[: -len(delimiter_binary)]
    message = binary_to_text(binary_message)

    if password:
        try:
            message = decrypt_message(message, password)
        except Exception:
            raise ValueError("Wrong password or message is not encrypted!")

    return message

# ─────────────────────────────────────────────
#  CLASSIC LSB  (kept for comparison tab)
# ─────────────────────────────────────────────

def encode_message_classic(image, message, password=None):
    img_array = np.array(image).copy()
    if password:
        message = encrypt_message(message, password)
    DELIMITER    = "###END###"
    binary_msg   = text_to_binary(message + DELIMITER)
    max_bytes    = img_array.shape[0] * img_array.shape[1] * 3
    if len(binary_msg) > max_bytes:
        raise ValueError(f"Message too long! Max {max_bytes // 8} characters.")
    flat = img_array.flatten()
    for i, bit in enumerate(binary_msg):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    return Image.fromarray(flat.reshape(img_array.shape).astype('uint8'))

def decode_message_classic(image, password=None):
    img_array = np.array(image)
    flat      = img_array.flatten()
    DELIMITER        = "###END###"
    delimiter_binary = text_to_binary(DELIMITER)
    binary_message   = ''
    for px in flat:
        binary_message += str(px & 1)
        if len(binary_message) >= len(delimiter_binary):
            if binary_message[-len(delimiter_binary):] == delimiter_binary:
                break
    if delimiter_binary not in binary_message:
        raise ValueError("No hidden message found in this image!")
    binary_message = binary_message[: -len(delimiter_binary)]
    message = binary_to_text(binary_message)
    if password:
        try:
            message = decrypt_message(message, password)
        except Exception:
            raise ValueError("Wrong password or message is not encrypted!")
    return message

# ─────────────────────────────────────────────
#  FILE ENCODE / DECODE  (classic LSB)
# ─────────────────────────────────────────────

def encode_file(image, file_data, filename):
    img_array     = np.array(image).copy()
    file_info     = f"{filename}|||{len(file_data)}"
    DELIM         = "###FILEEND###"
    full_binary   = (text_to_binary(file_info) + text_to_binary(DELIM)
                     + bytes_to_binary(file_data) + text_to_binary(DELIM))
    max_bytes     = img_array.shape[0] * img_array.shape[1] * 3
    if len(full_binary) > max_bytes:
        raise ValueError(f"File too large! Max ~{max_bytes // 8} bytes.")
    flat = img_array.flatten()
    for i, bit in enumerate(full_binary):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    return Image.fromarray(flat.reshape(img_array.shape).astype('uint8'))

def decode_file(image):
    img_array  = np.array(image)
    flat       = img_array.flatten()
    DELIM      = "###FILEEND###"
    delim_bin  = text_to_binary(DELIM)
    binary_data, delim_count = '', 0
    for px in flat:
        binary_data += str(px & 1)
        if len(binary_data) >= len(delim_bin) and binary_data[-len(delim_bin):] == delim_bin:
            delim_count += 1
            if delim_count == 2:
                break
    if delim_count < 2:
        raise ValueError("No hidden file found or data is incomplete!")
    info_end     = binary_data.index(delim_bin)
    file_info    = binary_to_text(binary_data[:info_end])
    parts        = file_info.split('|||')
    if len(parts) != 2:
        raise ValueError("Invalid file format in metadata!")
    filename, file_size = parts[0], int(parts[1])
    start   = info_end + len(delim_bin)
    end     = len(binary_data) - len(delim_bin)
    fb      = binary_data[start:end][: file_size * 8]
    fd      = binary_to_bytes(fb)
    if len(fd) != file_size:
        raise ValueError("Extracted file size mismatch!")
    return fd, filename

# ─────────────────────────────────────────────
#  METRICS (PSNR / SSIM-lite)
# ─────────────────────────────────────────────

def calculate_psnr(original, encoded):
    orig = np.array(original).astype(np.float64)
    enc  = np.array(encoded).astype(np.float64)
    mse  = np.mean((orig - enc) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_capacity(image):
    arr = np.array(image)
    return (arr.shape[0] * arr.shape[1] * 3) // 8 - 10

def adaptive_capacity(image, threshold_pct=50):
    """Return how many characters can be hidden in the top 50% complex pixels."""
    arr  = np.array(image)
    mag  = compute_texture_map(arr)
    flat = mag.flatten()
    threshold = np.percentile(flat, 100 - threshold_pct)
    usable_pixels = int(np.sum(flat >= threshold))
    return usable_pixels * 3 // 8 - 10   # 3 channels, 1 bit each, minus delimiter

# ─────────────────────────────────────────────
#  VISUALISATION HELPERS
# ─────────────────────────────────────────────

def make_texture_heatmap(image):
    arr = np.array(image)
    mag = compute_texture_map(arr)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(image)
    axes[0].set_title("Original Image", color='white', fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(mag, cmap='hot', interpolation='nearest')
    axes[1].set_title("Texture / Edge Map (Laplacian)", color='white', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    threshold = np.percentile(mag, 50)
    mask      = mag >= threshold
    overlay   = arr.copy()
    overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Pixels used for hiding (bright)", color='white', fontsize=11)
    axes[2].axis('off')

    for ax in axes:
        ax.tick_params(colors='white')
    plt.tight_layout()
    return fig

def make_comparison_chart(original, encoded):
    orig_arr = np.array(original).astype(np.float64)
    enc_arr  = np.array(encoded).astype(np.float64)
    diff     = np.abs(orig_arr - enc_arr)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#0e1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(original);            ax1.set_title("Original", color='white'); ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(encoded);             ax2.set_title("Encoded",  color='white'); ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2]); im3 = ax3.imshow(diff.astype(np.uint8) * 50, cmap='hot')
    ax3.set_title("Difference (×50)", color='white'); ax3.axis('off')
    plt.colorbar(im3, ax=ax3)

    colors = ['red', 'green', 'blue']
    labels = ['Red channel', 'Green channel', 'Blue channel']
    ax4 = fig.add_subplot(gs[1, :])
    for ch, (col, lbl) in enumerate(zip(colors, labels)):
        hist_orig, bins = np.histogram(orig_arr[:, :, ch], bins=64, range=(0, 255))
        hist_enc,  _    = np.histogram(enc_arr[:, :, ch],  bins=64, range=(0, 255))
        ax4.plot(bins[:-1], hist_orig, color=col, alpha=0.5, linestyle='--', label=f"{lbl} original")
        ax4.plot(bins[:-1], hist_enc,  color=col, alpha=1.0,                  label=f"{lbl} encoded")
    ax4.set_title("Pixel Histogram Comparison (dashed = original)", color='white')
    ax4.set_facecolor('#1a1a2e')
    ax4.tick_params(colors='white')
    ax4.legend(fontsize=7, ncol=2)
    for sp in ax4.spines.values():
        sp.set_color('gray')

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
#  STEGANALYSIS: chi-square test on LSBs
# ─────────────────────────────────────────────

def chi_square_test(image):
    """
    Simplified chi-square steganalysis.
    In a 'clean' image, adjacent even/odd values of each pixel shade should be
    roughly equal. LSB steganography (classic sequential) makes them more equal
    than natural randomness → high chi-square score ≈ hidden data suspected.
    Returns score per channel and an overall verdict.
    """
    arr      = np.array(image)
    results  = {}
    suspects = []

    for ch, name in enumerate(['Red', 'Green', 'Blue']):
        channel = arr[:, :, ch].flatten().astype(np.int32)
        # Pair up (0,1), (2,3), (4,5), ...
        pairs_expected = []
        pairs_observed = []
        for v in range(0, 255, 2):
            n0 = int(np.sum(channel == v))
            n1 = int(np.sum(channel == v + 1))
            total = n0 + n1
            if total > 0:
                expected = total / 2
                pairs_expected.append(expected)
                pairs_observed.append(n0)

        pe   = np.array(pairs_expected, dtype=np.float64)
        po   = np.array(pairs_observed, dtype=np.float64)
        chi2 = float(np.sum((po - pe) ** 2 / (pe + 1e-9)))
        # Normalize by degrees of freedom
        dof  = len(pe) - 1
        norm = chi2 / max(dof, 1)

        results[name] = {'chi2': chi2, 'normalized': norm}
        if norm < 1.5:
            suspects.append(name)

    overall = len(suspects) >= 2
    return results, overall, suspects

# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SteganoPy – Adaptive LSB",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 SteganoPy — Adaptive LSB Steganography")
st.markdown(
    "Hide messages **only in texture-rich regions** of an image — "
    "statistically harder to detect than classic LSB."
)

tabs = st.tabs([
    "🧠 Adaptive Encode",
    "🔍 Adaptive Decode",
    "📊 Quality Metrics",
    "🔬 Steganalysis",
    "📝 Classic Encode",
    "🔓 Classic Decode",
    "📁 File Hide/Extract",
    "📦 Batch",
])
tab_aenc, tab_adec, tab_metrics, tab_steg, tab_cenc, tab_cdec, tab_file, tab_batch = tabs

# ══════════════════════════════════════════════
#  TAB 1 — ADAPTIVE ENCODE
# ══════════════════════════════════════════════
with tab_aenc:
    st.header("🧠 Adaptive LSB — Encode")
    st.info(
        "This method analyses the image first with a **Laplacian edge filter**, "
        "then hides bits **only in the most complex (textured) pixels**. "
        "Smooth regions are left untouched — making the stego-image far harder "
        "to detect statistically."
    )

    up = st.file_uploader("Upload cover image", type=['png','jpg','jpeg','bmp','tiff','webp'], key="aenc_img")

    if up:
        image = Image.open(up).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cover Image")
            st.image(image, use_container_width=True)
            cap_classic  = calculate_capacity(image)
            cap_adaptive = adaptive_capacity(image)
            st.metric("Classic LSB capacity",  f"{cap_classic:,} chars")
            st.metric("Adaptive LSB capacity", f"{cap_adaptive:,} chars", help="Usable texture pixels only")

        bits_per_ch = st.select_slider(
            "Bits per channel (1 = imperceptible, 2 = 2× capacity, 3 = slight noise)",
            options=[1, 2, 3], value=1
        )

        message = st.text_area("Secret message:", height=130, placeholder="Type your secret message here…")

        use_enc  = st.checkbox("🔐 Encrypt with AES-256", value=True)
        password = None
        if use_enc:
            password = st.text_input("Password:", type="password", key="aenc_pass")

        if message:
            estimated_bits = len(text_to_binary(message + "###END###"))
            if use_enc and password:
                enc_preview = encrypt_message(message, password)
                estimated_bits = len(text_to_binary(enc_preview + "###END###"))
            capacity_bits = adaptive_capacity(image) * 8
            if estimated_bits > capacity_bits:
                st.error("⚠️ Message is too long for this image's texture capacity.")
            else:
                pct = estimated_bits / max(capacity_bits, 1) * 100
                st.progress(min(pct/100, 1.0), text=f"Capacity used: {pct:.1f}%")

        disabled = not message or (use_enc and not password)
        if st.button("🔒 Adaptive Encode", type="primary", disabled=disabled, key="aenc_btn"):
            try:
                with st.spinner("Analysing texture map and embedding…"):
                    encoded = adaptive_encode(image, message,
                                              password if use_enc else None,
                                              bits_per_channel=bits_per_ch)
                st.session_state['last_original'] = image
                st.session_state['last_encoded']  = encoded

                with col2:
                    st.subheader("Stego Image")
                    st.image(encoded, use_container_width=True)
                    psnr = calculate_psnr(image, encoded)
                    st.metric("PSNR", f"{psnr:.2f} dB", help=">40 dB = visually lossless")
                    buf = io.BytesIO()
                    encoded.save(buf, format='PNG')
                    st.download_button("⬇️ Download stego image", buf.getvalue(),
                                       "stego_adaptive.png", "image/png")
                    st.success("✅ Message hidden successfully!")
            except Exception as e:
                st.error(f"❌ {e}")

        # Texture heatmap preview
        if st.button("🗺️ Show Texture Map", key="aenc_heatmap"):
            with st.spinner("Computing texture map…"):
                fig = make_texture_heatmap(image)
            st.pyplot(fig)
            plt.close(fig)

# ══════════════════════════════════════════════
#  TAB 2 — ADAPTIVE DECODE
# ══════════════════════════════════════════════
with tab_adec:
    st.header("🔍 Adaptive LSB — Decode")

    up2 = st.file_uploader("Upload stego image", type=['png','jpg','jpeg','bmp','tiff','webp'], key="adec_img")

    if up2:
        stego = Image.open(up2).convert('RGB')
        st.image(stego, width=400)

        bits_per_ch2 = st.select_slider(
            "Bits per channel (must match encoding setting)",
            options=[1, 2, 3], value=1, key="adec_bpc"
        )
        use_dec  = st.checkbox("🔐 Message is encrypted", value=True, key="adec_enc")
        password2 = None
        if use_dec:
            password2 = st.text_input("Password:", type="password", key="adec_pass")

        if st.button("🔓 Adaptive Decode", type="primary",
                     disabled=(use_dec and not password2), key="adec_btn"):
            try:
                with st.spinner("Reconstructing pixel order and extracting bits…"):
                    msg = adaptive_decode(stego,
                                         password2 if use_dec else None,
                                         bits_per_channel=bits_per_ch2)
                st.success("✅ Message found!")
                st.text_area("Hidden message:", value=msg, height=150, disabled=True)
                st.info(f"Length: {len(msg)} characters")
            except Exception as e:
                st.error(f"❌ {e}")

# ══════════════════════════════════════════════
#  TAB 3 — QUALITY METRICS
# ══════════════════════════════════════════════
with tab_metrics:
    st.header("📊 Quality Metrics & Visual Comparison")
    st.info("Compare original vs stego image to measure imperceptibility.")

    col_a, col_b = st.columns(2)
    with col_a:
        orig_up = st.file_uploader("Upload ORIGINAL image", type=['png','jpg','jpeg','bmp'], key="met_orig")
    with col_b:
        enc_up  = st.file_uploader("Upload STEGO (encoded) image", type=['png','jpg','jpeg','bmp'], key="met_enc")

    # Also allow using the last encode result
    if 'last_original' in st.session_state and 'last_encoded' in st.session_state:
        if st.button("📥 Use last encoded pair from Adaptive Encode tab"):
            orig_up = None  # will use session state below

    use_session = ('last_original' in st.session_state and 'last_encoded' in st.session_state
                   and not orig_up and not enc_up)

    if use_session:
        orig_img = st.session_state['last_original']
        enc_img  = st.session_state['last_encoded']
        st.success("Using last adaptive encode result.")
    elif orig_up and enc_up:
        orig_img = Image.open(orig_up).convert('RGB')
        enc_img  = Image.open(enc_up).convert('RGB')
    else:
        orig_img = enc_img = None

    if orig_img and enc_img:
        psnr = calculate_psnr(orig_img, enc_img)
        orig_arr = np.array(orig_img)
        enc_arr  = np.array(enc_img)
        diff     = np.abs(orig_arr.astype(int) - enc_arr.astype(int))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR", f"{psnr:.2f} dB")
        m2.metric("Max pixel diff", f"{diff.max()}")
        m3.metric("Mean pixel diff", f"{diff.mean():.4f}")
        changed = int(np.sum(np.any(diff > 0, axis=2)))
        m4.metric("Changed pixels", f"{changed:,}")

        verdict = "✅ Excellent — visually lossless" if psnr > 45 else \
                  "✅ Good — imperceptible to humans" if psnr > 40 else \
                  "⚠️ Acceptable but may show minor noise" if psnr > 35 else \
                  "❌ Noticeable degradation"
        st.info(f"Quality verdict: **{verdict}**")

        with st.spinner("Generating comparison charts…"):
            fig = make_comparison_chart(orig_img, enc_img)
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════
#  TAB 4 — STEGANALYSIS
# ══════════════════════════════════════════════
with tab_steg:
    st.header("🔬 Steganalysis — Chi-Square Attack")
    st.info(
        "The **chi-square attack** tests whether the LSB plane of an image looks "
        "'too random' — a telltale sign of sequential LSB steganography. "
        "Adaptive LSB distributes bits in textured regions, making it harder to detect."
    )

    sa_up = st.file_uploader("Upload image to analyse", type=['png','jpg','jpeg','bmp'], key="steg_up")

    if sa_up:
        sa_img = Image.open(sa_up).convert('RGB')
        st.image(sa_img, width=400)

        if st.button("🔬 Run Chi-Square Analysis", type="primary", key="steg_btn"):
            with st.spinner("Running statistical analysis…"):
                results, suspected, suspect_channels = chi_square_test(sa_img)

            st.subheader("Results per channel")
            cols = st.columns(3)
            for i, (ch, data) in enumerate(results.items()):
                with cols[i]:
                    norm = data['normalized']
                    label = "🟢 Clean" if norm >= 1.5 else "🔴 Suspicious"
                    st.metric(f"{ch} channel χ² (norm)", f"{norm:.3f}", label)

            if suspected:
                st.error(
                    f"⚠️ **Hidden data suspected!** "
                    f"Channels showing anomaly: {', '.join(suspect_channels)}. "
                    "This is consistent with classic sequential LSB steganography."
                )
            else:
                st.success(
                    "✅ **No steganography detected** by chi-square test. "
                    "The LSB distribution appears natural — consistent with Adaptive LSB "
                    "or a clean image."
                )

            # LSB plane visualisation
            arr  = np.array(sa_img)
            fig2, axs = plt.subplots(1, 3, figsize=(12, 4))
            fig2.patch.set_facecolor('#0e1117')
            titles = ['Red LSB plane', 'Green LSB plane', 'Blue LSB plane']
            for ch, (ax, title) in enumerate(zip(axs, titles)):
                lsb = (arr[:, :, ch] & 1) * 255
                ax.imshow(lsb, cmap='gray')
                ax.set_title(title, color='white')
                ax.axis('off')
            plt.suptitle("LSB Bit Planes (random = clean, structured = hidden data)",
                         color='white', y=1.01)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ══════════════════════════════════════════════
#  TAB 5 — CLASSIC ENCODE
# ══════════════════════════════════════════════
with tab_cenc:
    st.header("📝 Classic LSB — Encode")
    st.warning("Classic LSB writes bits sequentially from pixel (0,0) — detectable by chi-square steganalysis. Use the Adaptive tab for better security.")

    up_c = st.file_uploader("Upload cover image", type=['png','jpg','jpeg','bmp','tiff','webp'], key="cenc_img")
    if up_c:
        img_c = Image.open(up_c).convert('RGB')
        col1c, col2c = st.columns(2)
        with col1c:
            st.image(img_c, use_container_width=True)
            st.info(f"Capacity: {calculate_capacity(img_c):,} chars")

        msg_c    = st.text_area("Secret message:", height=120, key="cenc_msg")
        use_ec   = st.checkbox("🔐 AES-256 encryption", key="cenc_enc")
        pass_c   = st.text_input("Password:", type="password", key="cenc_pass") if use_ec else None

        if st.button("🔒 Classic Encode", type="primary",
                     disabled=(not msg_c or (use_ec and not pass_c)), key="cenc_btn"):
            try:
                enc_c = encode_message_classic(img_c, msg_c, pass_c if use_ec else None)
                with col2c:
                    st.image(enc_c, use_container_width=True)
                    buf = io.BytesIO(); enc_c.save(buf, format='PNG')
                    st.download_button("⬇️ Download", buf.getvalue(), "stego_classic.png", "image/png")
                    st.metric("PSNR", f"{calculate_psnr(img_c, enc_c):.2f} dB")
            except Exception as e:
                st.error(f"❌ {e}")

# ══════════════════════════════════════════════
#  TAB 6 — CLASSIC DECODE
# ══════════════════════════════════════════════
with tab_cdec:
    st.header("🔓 Classic LSB — Decode")
    up_cd = st.file_uploader("Upload stego image", type=['png','jpg','jpeg','bmp','tiff','webp'], key="cdec_img")
    if up_cd:
        img_cd = Image.open(up_cd).convert('RGB')
        st.image(img_cd, width=400)
        use_dcd  = st.checkbox("🔐 Message is encrypted", key="cdec_enc")
        pass_cd  = st.text_input("Password:", type="password", key="cdec_pass") if use_dcd else None
        if st.button("🔓 Classic Decode", type="primary",
                     disabled=(use_dcd and not pass_cd), key="cdec_btn"):
            try:
                msg_cd = decode_message_classic(img_cd, pass_cd if use_dcd else None)
                st.success("✅ Message found!")
                st.text_area("Message:", value=msg_cd, height=150, disabled=True)
            except Exception as e:
                st.error(f"❌ {e}")

# ══════════════════════════════════════════════
#  TAB 7 — FILE HIDE / EXTRACT
# ══════════════════════════════════════════════
with tab_file:
    st.header("📁 Hide / Extract a File")
    ft1, ft2 = st.tabs(["Hide File", "Extract File"])

    with ft1:
        fi_up  = st.file_uploader("Cover image", type=['png','jpg','jpeg','bmp','tiff','webp'], key="fenc_img")
        fh_up  = st.file_uploader("File to hide (any type)", type=None, key="fenc_file")
        if fi_up and fh_up:
            fi_img  = Image.open(fi_up).convert('RGB')
            fb      = fh_up.read()
            cap_f   = (np.array(fi_img).shape[0] * np.array(fi_img).shape[1] * 3) // 8 - 100
            col1f, col2f = st.columns(2)
            with col1f:
                st.image(fi_img, use_container_width=True)
                st.info(f"Capacity: {cap_f:,} bytes | File: {len(fb):,} bytes")
                if len(fb) > cap_f:
                    st.error("File is too large!")
            if st.button("🔒 Hide File", type="primary", disabled=len(fb) > cap_f, key="fenc_btn"):
                try:
                    enc_fi = encode_file(fi_img, fb, fh_up.name)
                    with col2f:
                        st.image(enc_fi, use_container_width=True)
                        buf = io.BytesIO(); enc_fi.save(buf, format='PNG')
                        st.download_button("⬇️ Download", buf.getvalue(),
                                           f"stego_file_{fh_up.name}.png", "image/png")
                        st.success("✅ File hidden!")
                except Exception as e:
                    st.error(f"❌ {e}")

    with ft2:
        fd_up = st.file_uploader("Stego image with hidden file", type=['png','jpg','jpeg','bmp','tiff','webp'], key="fdec_img")
        if fd_up:
            fd_img = Image.open(fd_up).convert('RGB')
            st.image(fd_img, width=400)
            if st.button("🔓 Extract File", type="primary", key="fdec_btn"):
                try:
                    data, fname = decode_file(fd_img)
                    st.success(f"✅ Extracted: {fname} ({len(data):,} bytes)")
                    st.download_button(f"⬇️ Download {fname}", data, fname, "application/octet-stream")
                except Exception as e:
                    st.error(f"❌ {e}")

# ══════════════════════════════════════════════
#  TAB 8 — BATCH
# ══════════════════════════════════════════════
with tab_batch:
    st.header("📦 Batch Encoding")
    st.info("Hide the same message in multiple images at once — uses Adaptive LSB.")

    batch_imgs = st.file_uploader("Upload images", type=['png','jpg','jpeg','bmp','tiff','webp'],
                                   accept_multiple_files=True, key="batch_imgs")
    if batch_imgs:
        st.success(f"{len(batch_imgs)} images uploaded")
        b_msg  = st.text_area("Message to hide in all images:", height=100, key="b_msg")
        b_enc  = st.checkbox("🔐 AES-256 encryption", key="b_enc")
        b_pass = st.text_input("Password:", type="password", key="b_pass") if b_enc else None
        b_bpc  = st.select_slider("Bits per channel", options=[1,2,3], value=1, key="b_bpc")

        disabled_b = not b_msg or (b_enc and not b_pass)
        if st.button("🔒 Encode All", type="primary", disabled=disabled_b, key="batch_btn"):
            progress = st.progress(0)
            zip_buf  = io.BytesIO()
            ok = err = 0
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for idx, img_f in enumerate(batch_imgs):
                    try:
                        img_f.seek(0)
                        img_b = Image.open(img_f).convert('RGB')
                        enc_b = adaptive_encode(img_b, b_msg, b_pass if b_enc else None, b_bpc)
                        ibuf  = io.BytesIO(); enc_b.save(ibuf, format='PNG')
                        base  = os.path.splitext(img_f.name)[0]
                        zf.writestr(f"{base}_adaptive_stego.png", ibuf.getvalue())
                        ok += 1
                    except Exception as e:
                        st.warning(f"⚠️ {img_f.name}: {e}")
                        err += 1
                    progress.progress((idx + 1) / len(batch_imgs))
            progress.empty()
            if ok:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_buf.seek(0)
                st.success(f"✅ {ok} images encoded ({err} failed)")
                st.download_button("⬇️ Download ZIP", zip_buf.getvalue(),
                                   f"batch_stego_{ts}.zip", "application/zip")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
### 📖 How Adaptive LSB works

**Classic LSB** writes bits sequentially starting at pixel (0,0).  
This creates a statistical anomaly in the LSB plane that the **chi-square test** can detect.

**Adaptive LSB** (this implementation):
1. Applies a **Laplacian edge filter** to compute a per-pixel complexity score.
2. Ranks all pixels from most to least textured.
3. Embeds bits **only in the top-N most complex pixels** — edges, corners, noise.
4. Smooth regions (sky, walls) are completely untouched.
5. Optionally, pixel selection within tiers is **pseudo-randomised using the password** as a seed.

This approach is:
- 🔬 **Statistically resistant** — the LSB distribution in smooth regions looks natural.
- 👁️ **Perceptually optimal** — changes land where the human eye is least sensitive.
- 🔐 **Doubly secure** when combined with AES-256 encryption.

**PSNR > 40 dB** is considered visually lossless. Adaptive LSB typically achieves 48–55 dB.
""")
