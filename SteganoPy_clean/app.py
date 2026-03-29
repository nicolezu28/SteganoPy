"""
Diagnostic wrapper — deploy THIS as your app.py temporarily.
It will try to import and run your actual app code, and if anything
crashes, it will display the full traceback ON SCREEN so you can
see the real error on Streamlit Cloud.

Once you find and fix the error, replace this file with your real app.py.
"""
import streamlit as st
import sys
import traceback

st.set_page_config(page_title="SteganoPy Diagnostics", page_icon="🔧")
st.title("🔧 SteganoPy — Diagnostic Mode")

# Step 1: Test all imports one by one
st.header("1. Testing imports...")
imports_ok = True

test_imports = [
    ("streamlit", "import streamlit"),
    ("PIL / Pillow", "from PIL import Image"),
    ("numpy", "import numpy as np"),
    ("scipy.ndimage", "from scipy import ndimage"),
    ("matplotlib", "import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec"),
    ("cryptography", "from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes; from cryptography.hazmat.backends import default_backend; from cryptography.hazmat.primitives import padding"),
]

for name, stmt in test_imports:
    try:
        exec(stmt)
        st.success(f"✅ {name}")
    except Exception as e:
        st.error(f"❌ {name}: {e}")
        imports_ok = False

if imports_ok:
    st.success("All imports OK!")
else:
    st.error("Some imports failed — fix these first before deploying the full app.")
    st.stop()

# Step 2: Show Python and package versions
st.header("2. Environment info")
import numpy, scipy, matplotlib, PIL, cryptography
st.code(f"""
Python:        {sys.version}
NumPy:         {numpy.__version__}
SciPy:         {scipy.__version__}
Matplotlib:    {matplotlib.__version__}
Pillow:        {PIL.__version__}
Cryptography:  {cryptography.__version__}
Streamlit:     {st.__version__}
""")

# Step 3: Test core functionality
st.header("3. Testing core functions...")
try:
    from scipy import ndimage
    import numpy as np
    
    # Simulate what the app does: create a small test image and run the Laplacian
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = np.mean(test_img, axis=2).astype(np.float32)
    lap = ndimage.laplace(gray)
    magnitude = np.abs(lap)
    st.success(f"✅ Laplacian filter works (magnitude range: {magnitude.min():.2f} - {magnitude.max():.2f})")
except Exception as e:
    st.error(f"❌ Laplacian test failed: {e}")
    st.code(traceback.format_exc())

# Step 4: Test matplotlib rendering
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test plot")
    st.pyplot(fig)
    plt.close(fig)
    st.success("✅ Matplotlib rendering works")
except Exception as e:
    st.error(f"❌ Matplotlib test failed: {e}")
    st.code(traceback.format_exc())

st.header("4. All diagnostics passed!")
st.info(
    "If you see this page, your environment is healthy. "
    "The crash is likely in your app.py code itself. "
    "Replace this diagnostic file with your real app.py and check "
    "for any differences between your local file and the GitHub version."
)
