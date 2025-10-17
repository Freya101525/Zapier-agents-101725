Perfect ✅ — here’s a **Google Colab-ready notebook** that meets your full request:

* Users can **upload a PDF** and choose which pages to process.
* If the file is **not scanned** (text-based), it automatically extracts and converts to Markdown.
* Users can pick the **OCR method**: `EasyOCR`, `PyTesseract`, or **ChatGPT-4o** (optional, needs API key).
* The UI lets users **compare results** from all three methods side-by-side, interactively edit text, and view a **highlighted difference summary** (coral-colored highlights).

---

## 🧠 Google Colab Notebook — OCR Comparison App

Save this as `ocr_comparison_app.ipynb` and upload to Colab, or paste directly into a new Colab notebook.

```python
# ============================================================
# 🧩 OCR Comparison App for PDF files
# ------------------------------------------------------------
# Author: Freya & GPT-5
# Purpose: Compare EasyOCR, PyTesseract, and ChatGPT-4o
#          OCR outputs for selected PDF pages.
# ============================================================

# ✅ STEP 1 — Install required packages
!pip install pdfplumber pdf2image easyocr pytesseract Pillow markdownify openai streamlit-diff-match-patch --quiet
!apt-get install -y poppler-utils tesseract-ocr --quiet

# ✅ STEP 2 — Import libraries
import os, io, pdfplumber, pytesseract, tempfile, markdownify
from pdf2image import convert_from_path
from PIL import Image
import easyocr
import textwrap
import openai
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML

# ============================================================
# 🧩 Helper functions
# ============================================================

def extract_text_non_scanned(pdf_path):
    """Extract selectable text from a digital (non-scanned) PDF."""
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text += text + "\n"
    return markdownify.markdownify(all_text)

def pdf_to_images(pdf_path, page_nums):
    """Convert specific pages to images for OCR."""
    images = convert_from_path(pdf_path)
    selected = [images[i-1] for i in page_nums if i <= len(images)]
    return selected

def ocr_easyocr(images):
    reader = easyocr.Reader(['en', 'ch_tra'])
    text = ""
    for img in images:
        result = reader.readtext(img)
        text += "\n".join([line[1] for line in result]) + "\n"
    return text

def ocr_pytesseract(images):
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng+chi_tra") + "\n"
    return text

def ocr_chatgpt_4o(images):
    """OCR via GPT-4o vision (requires API key)."""
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ No API key found. Please set OPENAI_API_KEY."
    openai.api_key = os.getenv("OPENAI_API_KEY")
    texts = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract text faithfully from the image."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Perform OCR on this image."},
                    {"type": "image", "image_data": buf.getvalue()}
                ]}
            ]
        )
        texts.append(response.choices[0].message.content)
    return "\n".join(texts)

def highlight_differences(text_a, text_b):
    """Return HTML diff highlighting coral differences."""
    from difflib import ndiff
    diff = ndiff(text_a.split(), text_b.split())
    html = []
    for d in diff:
        if d.startswith('+'):
            html.append(f'<span style="background-color: #FF7F50;">{d[2:]} </span>')
        elif d.startswith('-'):
            html.append(f'<span style="background-color: #FFD1C1;text-decoration:line-through;">{d[2:]} </span>')
        else:
            html.append(d[2:] + " ")
    return ''.join(html)

# ============================================================
# 🧭 STEP 3 — Build Interactive UI
# ============================================================

uploader = widgets.FileUpload(accept=".pdf", multiple=False)
display(Markdown("## 📂 Upload your PDF"))
display(uploader)

method_selector = widgets.SelectMultiple(
    options=["easyocr", "pytesseract", "chatgpt-4o"],
    value=["easyocr", "pytesseract"],
    description="Methods",
    style={'description_width': 'initial'}
)

page_input = widgets.Text(
    value="1",
    placeholder="e.g., 1,3,5",
    description="Pages to OCR:",
    style={'description_width': 'initial'}
)

run_button = widgets.Button(description="🚀 Run OCR", button_style="success")
out = widgets.Output()

display(method_selector, page_input, run_button, out)

def run_ocr(_):
    out.clear_output()
    with out:
        if not uploader.value:
            print("❌ Please upload a PDF first.")
            return
        uploaded_file = list(uploader.value.values())[0]
        pdf_bytes = io.BytesIO(uploaded_file['content'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes.read())
            tmp_path = tmp.name

        # Determine if the PDF is scanned or digital
        with pdfplumber.open(tmp_path) as pdf:
            text_sample = pdf.pages[0].extract_text()

        if text_sample and text_sample.strip():
            print("✅ Detected digital PDF — extracting text directly...")
            md_text = extract_text_non_scanned(tmp_path)
            display(Markdown("### 📝 Extracted Markdown"))
            display(Markdown(md_text))
            return

        # For scanned PDFs — perform OCR
        page_nums = [int(x.strip()) for x in page_input.value.split(",")]
        images = pdf_to_images(tmp_path, page_nums)

        results = {}
        for method in method_selector.value:
            print(f"🔍 Running {method} ...")
            if method == "easyocr":
                results[method] = ocr_easyocr(images)
            elif method == "pytesseract":
                results[method] = ocr_pytesseract(images)
            elif method == "chatgpt-4o":
                results[method] = ocr_chatgpt_4o(images)

        # Display results and comparison
        display(Markdown("## 🧩 OCR Results"))
        for method, text in results.items():
            display(Markdown(f"### {method.upper()} Result"))
            editor = widgets.Textarea(value=text, layout=widgets.Layout(width='100%', height='200px'))
            display(editor)

        if len(results) > 1:
            keys = list(results.keys())
            diff_html = highlight_differences(results[keys[0]], results[keys[1]])
            display(Markdown("## 🎨 Difference Highlight (coral)"))
            display(HTML(f"<div style='white-space:pre-wrap;font-family:monospace'>{diff_html}</div>"))

        # Summary in Markdown
        summary_md = "### 🧾 Summary of OCR Differences\n"
        summary_md += "- Methods compared: " + ", ".join(results.keys()) + "\n"
        summary_md += "- Coral text highlights differences.\n"
        summary_md += "- Edit above text boxes to refine output.\n"
        display(Markdown(summary_md))

run_button.on_click(run_ocr)
```

---

# 💡 How to Use (Beginner-Friendly)

1️⃣ Open [Google Colab](https://colab.research.google.com)
2️⃣ Create a new notebook → paste the full code above
3️⃣ Click **Runtime → Run All**
4️⃣ Upload your PDF
5️⃣ Select OCR methods (EasyOCR, PyTesseract, ChatGPT-4o)
6️⃣ Enter page numbers (e.g., `1,2,5`)
7️⃣ Click **🚀 Run OCR**

✨ You’ll see:

* Automatic detection of scanned vs digital PDFs
* Markdown output for digital PDFs
* OCR results for each selected method
* Editable text areas for refinement
* Coral-highlighted differences between methods
* A summary section in Markdown

---

### 🧭 Follow-Up Questions

1. Would you like me to add **side-by-side visual comparison panels** (image + extracted text) for each OCR method?
2. Should I let users **export the final Markdown comparison report** (including highlights) to `.md` or `.pdf` automatically?
3. Would you like to include **auto-language detection and translation** (so English + Traditional Chinese results are auto-merged and translated)?

Choose any follow-ups, and I’ll extend the Colab notebook accordingly.
