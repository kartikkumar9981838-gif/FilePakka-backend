import io
import json
import logging
from typing import List, Dict, Any, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pikepdf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filepakka-backend")

app = FastAPI(
    title="FilePakka Stateless Backend",
    description="Stateless high-performance PDF and Image processing endpoints",
    version="1.0.0"
)

# Enable CORS for all origins in development (can be restricted in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 25MB Max file size limit (25 * 1024 * 1024 bytes)
MAX_FILE_SIZE = 25 * 1024 * 1024

def validate_file_size(file: UploadFile):
    """Checks the size of the uploaded file and raises 400 if it exceeds the limit."""
    # We can read a portion or check the content length from headers,
    # but the most reliable way for multipart files is reading up to MAX_FILE_SIZE.
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of 25MB (Size: {size / (1024 * 1024):.2f}MB)"
        )

# Health endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FilePakka backend is running and healthy"}

# ENDPOINT 1: POST /add-password
@app.post("/add-password")
async def add_password(
    file: UploadFile = File(...),
    password: str = Form(...)
):
    validate_file_size(file)
    if not password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")
        
    try:
        file_bytes = await file.read()
        
        # Load PDF using pikepdf
        with pikepdf.open(io.BytesIO(file_bytes)) as pdf:
            # Set up AES-256 Encryption (R=6)
            enc = pikepdf.Encryption(
                user=password, 
                owner=password, 
                R=6,
                allow=pikepdf.Permissions(accessibility=True)
            )
            
            # Save encrypted to an in-memory buffer
            out_buf = io.BytesIO()
            pdf.save(out_buf, encryption=enc)
            out_buf.seek(0)
            
            return StreamingResponse(
                out_buf, 
                media_type="application/pdf", 
                headers={"Content-Disposition": f"attachment; filename=protected_{file.filename}"}
            )
            
    except pikepdf.PdfError as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in /add-password: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ENDPOINT 2: POST /remove-password
@app.post("/remove-password")
async def remove_password(
    file: UploadFile = File(...),
    password: str = Form(...)
):
    validate_file_size(file)
    
    try:
        file_bytes = await file.read()
        
        # Open the PDF with the provided password
        try:
            pdf = pikepdf.open(io.BytesIO(file_bytes), password=password)
        except pikepdf.PasswordError:
            raise HTTPException(status_code=400, detail="Incorrect password")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file, or password decryption failed: {str(e)}")
            
        with pdf:
            # Save it back out without encryption (encryption omitted)
            out_buf = io.BytesIO()
            pdf.save(out_buf)
            out_buf.seek(0)
            
            return StreamingResponse(
                out_buf, 
                media_type="application/pdf", 
                headers={"Content-Disposition": f"attachment; filename=unlocked_{file.filename}"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /remove-password: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ENDPOINT 3: POST /redact
@app.post("/redact")
async def redact_pdf(
    file: UploadFile = File(...),
    redactions: str = Form(...)  # Expected to be a JSON string of redaction box objects
):
    validate_file_size(file)
    
    try:
        # Parse JSON redactions
        try:
            redactions_list = json.loads(redactions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for 'redactions'")
            
        file_bytes = await file.read()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Group redactions by page
        redactions_by_page = {}
        for r in redactions_list:
            p = int(r.get("page", 0))
            if p not in redactions_by_page:
                redactions_by_page[p] = []
            redactions_by_page[p].append(r)
            
        for page_num, rects in redactions_by_page.items():
            if page_num < 0 or page_num >= len(doc):
                continue
                
            page = doc[page_num]
            for r in rects:
                x = float(r.get("x", 0))
                y = float(r.get("y", 0))
                w = float(r.get("width", 0))
                h = float(r.get("height", 0))
                
                # fitz.Rect takes (x0, y0, x1, y1)
                rect = fitz.Rect(x, y, x + w, y + h)
                
                # Add redaction annotation - fill color is black by default
                # fill=(0, 0, 0) is RGB for black
                page.add_redact_annot(rect, fill=(0, 0, 0))
                
            # Apply redactions to genuinely remove underlying text and image content
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
            
        # Save to memory buffer
        out_buf = io.BytesIO()
        out_buf.write(doc.tobytes())
        out_buf.seek(0)
        doc.close()
        
        return StreamingResponse(
            out_buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=redacted_{file.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /redact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to redact PDF: {str(e)}")

# ENDPOINT 4: POST /edit-text
@app.post("/edit-text")
async def edit_text(
    file: UploadFile = File(...),
    edits: str = Form(...)  # Expected to be a JSON string of edit objects
):
    validate_file_size(file)
    
    try:
        # Parse JSON edits
        try:
            edits_list = json.loads(edits)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for 'edits'")
            
        file_bytes = await file.read()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Group edits by page
        edits_by_page = {}
        for e in edits_list:
            p = int(e.get("page", 0))
            if p not in edits_by_page:
                edits_by_page[p] = []
            edits_by_page[p].append(e)
            
        for page_num, page_edits in edits_by_page.items():
            if page_num < 0 or page_num >= len(doc):
                continue
                
            page = doc[page_num]
            # Retrieve text dict with layout structures
            text_dict = page.get_text("dict")
            
            for edit in page_edits:
                orig_text = edit.get("original_text", "").strip()
                new_text = edit.get("new_text", "")
                ex = float(edit.get("x", 0))
                ey = float(edit.get("y", 0))
                
                matched_span = None
                best_distance = float("inf")
                
                # Traversal of blocks, lines, and spans to find matching text and position
                for block in text_dict.get("blocks", []):
                    if block.get("type") != 0:  # Only text blocks
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            bbox = span.get("bbox")  # (x0, y0, x1, y1)
                            
                            if not bbox:
                                continue
                                
                            # Calculate distance from edit coordinate to span bbox center
                            bx_mid = (bbox[0] + bbox[2]) / 2
                            by_mid = (bbox[1] + bbox[3]) / 2
                            dist = np.sqrt((ex - bx_mid)**2 + (ey - by_mid)**2)
                            
                            # Check if the content is matching, or if we're very close
                            is_content_match = (orig_text in span_text) or (span_text in orig_text)
                            
                            # If we have a content match, we prioritize proximity
                            if is_content_match and dist < best_distance:
                                matched_span = span
                                best_distance = dist
                
                # If we didn't find any direct text match, let's look for the absolute closest span
                # within a tolerance of 15pt
                if not matched_span:
                    for block in text_dict.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                bbox = span.get("bbox")
                                if not bbox:
                                    continue
                                bx_mid = (bbox[0] + bbox[2]) / 2
                                by_mid = (bbox[1] + bbox[3]) / 2
                                dist = np.sqrt((ex - bx_mid)**2 + (ey - by_mid)**2)
                                if dist < 15 and dist < best_distance:
                                    matched_span = span
                                    best_distance = dist
                
                if matched_span:
                    bbox = matched_span["bbox"]
                    # Step 1: Redact original span bounding box to wipe it out cleanly
                    rect = fitz.Rect(bbox)
                    page.add_redact_annot(rect, fill=(1, 1, 1))  # Fill with white background
                    page.apply_redactions()
                    
                    # Step 2: Extract attributes to match visual style
                    fontsize = matched_span.get("size", 10)
                    color_int = matched_span.get("color", 0)
                    
                    # Decode RGB from integer color
                    r = ((color_int >> 16) & 0xFF) / 255.0
                    g = ((color_int >> 8) & 0xFF) / 255.0
                    b = (color_int & 0xFF) / 255.0
                    color_tuple = (r, g, b)
                    
                    # Insert new text at the baseline of the redacted box
                    # bbox: (x0, y0, x1, y1). The baseline is near y1
                    # A small offset upwards keeps it aligned properly
                    insert_point = fitz.Point(bbox[0], bbox[3] - (bbox[3] - bbox[1]) * 0.15)
                    
                    page.insert_text(
                        insert_point,
                        new_text,
                        fontsize=fontsize,
                        color=color_tuple,
                        fontname="helv"  # Safe standard Helvetica
                    )
                else:
                    # Fallback: if we can't find a matching span, we just insert the text at the specified coordinates
                    page.insert_text(
                        fitz.Point(ex, ey),
                        new_text,
                        fontsize=11,
                        color=(0, 0, 0),
                        fontname="helv"
                    )
                    
        # Save to memory buffer
        out_buf = io.BytesIO()
        out_buf.write(doc.tobytes())
        out_buf.seek(0)
        doc.close()
        
        return StreamingResponse(
            out_buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=edited_{file.filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /edit-text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to edit PDF text: {str(e)}")

# Auxiliary function to order corner points correctly (top-left, top-right, bottom-right, bottom-left)
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates: minimum is top-left, maximum is bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    
    # Difference of coordinates (y - x): minimum is top-right, maximum is bottom-left
    # Note: np.diff is x - y (since it is col2 - col1), so we calculate y - x manually or adjust:
    # y - x = pts[:, 1] - pts[:, 0]
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]    # TR (lowest y - x means high x, low y)
    rect[3] = pts[np.argmax(diff)]    # BL (highest y - x means low x, high y)
    
    return rect

# ENDPOINT 5: POST /scan-document
@app.post("/scan-document")
async def scan_document(
    file: UploadFile = File(...),
    corners: str = Form(...),  # Expected to be JSON string: [{"x": 10, "y": 20}, ...]
    mode: str = Query("color", description="Enhancement mode: 'color' or 'bw'")
):
    validate_file_size(file)
    
    try:
        # Parse JSON corners
        try:
            corners_list = json.loads(corners)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for 'corners'")
            
        if len(corners_list) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 corner points must be provided")
            
        # Load image using OpenCV
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image")
            
        # Format corners to numpy array
        pts = np.array([[float(pt.get("x", 0)), float(pt.get("y", 0))] for pt in corners_list], dtype="float32")
        
        # Order points
        ordered_rect = order_points(pts)
        (tl, tr, br, bl) = ordered_rect
        
        # Compute maximum width and height to check aspect ratio
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Enforce A4 proportions (1240 x 1754 or 1754 x 1240)
        is_landscape = max_width > max_height
        if is_landscape:
            dest_w, dest_h = 1754, 1240
        else:
            dest_w, dest_h = 1240, 1754
            
        # Destination coordinate mapping
        dst_pts = np.array([
            [0, 0],
            [dest_w - 1, 0],
            [dest_w - 1, dest_h - 1],
            [0, dest_h - 1]
        ], dtype="float32")
        
        # Perform Perspective Warp
        M = cv2.getPerspectiveTransform(ordered_rect, dst_pts)
        warped = cv2.warpPerspective(img, M, (dest_w, dest_h))
        
        # Enhance document depending on mode
        if mode == "bw":
            # Grayscale & bilateral filter for noise cleanup
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            # Adaptive Thresholding for sharp scanner effect
            enhanced = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 10
            )
        else:  # mode == "color" (enhanced color scan)
            # Convert to LAB for luminance extraction
            lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            # Apply local contrast enhancement CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b_ch))
            enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            # Increase contrast/brightness to make backgrounds whiter
            enhanced = cv2.convertScaleAbs(enhanced_color, alpha=1.1, beta=10)
            
        # Encode back to JPEG
        _, encoded_img = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out_buf = io.BytesIO(encoded_img.tobytes())
        
        return StreamingResponse(
            out_buf,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=scanned_{file.filename if file.filename else 'doc.jpg'}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /scan-document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to scan and process document: {str(e)}")

# Run command to start the application when executed directly
if __name__ == "__main__":
    import uvicorn
    import os
    env_port = int(os.environ.get("PORT", 9000))
    port = env_port if env_port != 3000 else 9000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
