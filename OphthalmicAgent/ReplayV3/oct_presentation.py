"""Label-blind OCT rendering and acquisition framing for glaucoma replay V3."""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACQUISITION_CONTEXT = (
    "Glaucoma-task OCT, expected optic-nerve-head context based on offline visual "
    "inspection of this dataset. This is not verified acquisition metadata. "
    "Scan location, laterality, anatomical orientation and physical pixel spacing "
    "are not supplied in these NPZ files."
)

SYSTEM_PROMPT = """You are an ophthalmic image analysis specialist reviewing OCT B-scans.

Provide objective visual observations that may help another agent assess glaucoma.
This is a glaucoma-task volume with expected optic-nerve-head context, not a
presumed macular scan. The context comes from dataset image inspection, not
verified acquisition metadata. Identify the anatomy supported by the visible
images; explicitly state when scan location or a structure cannot be established.
Do not assume a central depression is a foveal pit or macular lesion. When an
optic-nerve-head configuration is visible, describe its visible contour and the
surrounding retinal tissue. Do not infer glaucoma from an excavation alone.

One central slice is enlarged above four sampled context slices from the same
volume. Labels are zero-based indices along array axis 0, not anatomical sectors
or physical distances. The context slices are spread across the volume, not
adjacent slices. Their spacing does not establish continuity between them.

Important instructions:
* Do not provide a final diagnosis or estimate glaucoma probability.
* Base observations only on visible OCT findings.
* Describe retinal layer organization, visible inner-layer continuity, visible
  tissue loss, structural irregularities, and differences across the shown slices.
* Describe visible optic-nerve-head tissue if identifiable, without inventing
  quantitative rim thickness, cup-to-disc ratio or RNFL measurements.
* Do not invent micrometre values, laterality, anatomical sectors, findings outside
  the displayed scans, or comparisons with a normative database.
* Distinguish visible findings from structures that cannot be assessed.

Structure your response as:
Glaucoma-Relevant Features:
...
Overall Impression:
Provide a brief summary of your structural observations and any uncertain scan identity.
"""


def slice_metadata(shape):
    if len(shape) != 3 or shape[0] < 8 or min(shape[1:]) < 32:
        raise ValueError("Expected an OCT volume [depth >= 8, height >= 32, width >= 32]")
    sampled = np.linspace(0, shape[0] - 1, 8, dtype=int).tolist()
    return dict(volume_shape=list(shape), index_base=0, slice_axis=0,
                central_index=shape[0] // 2, classifier_sampled_indices=sampled,
                context_sample_slots=[1, 2, 4, 5],
                context_indices=[sampled[i] for i in (1, 2, 4, 5)],
                acquisition_context=ACQUISITION_CONTEXT,
                acquisition_metadata_verified=False,
                preprocessing="Legacy per-slice scaling; CLAHE clip=1.5, tiles=8x8; central 2x cubic")


def captions(meta):
    return (f"Central slice {meta['central_index']} | {meta['volume_shape'][0]} slices | zero-based indices",
            [f"Slice {i}" for i in meta["context_indices"]])


def enhance(array):
    # Preserve the existing specialist transform, including its scaling condition.
    image = (array * 255).astype(np.uint8) if array.max() <= 1 else array.astype(np.uint8)
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(image)


def render(volume):
    meta = slice_metadata(volume.shape)
    if volume.dtype.kind not in "uif" or not np.isfinite(volume).all():
        raise ValueError("OCT must be a finite real numeric array")
    if volume.min() < 0 or volume.max() > 255:
        raise ValueError("Unsupported OCT intensity range; do not silently change preprocessing")
    height, width = volume.shape[1:]
    gap, title_height, separator = 12, 35, 20
    total_width = 4 * width + 3 * gap
    middle = cv2.resize(enhance(volume[meta["central_index"]]), (2 * width, 2 * height),
                        interpolation=cv2.INTER_CUBIC)
    image = Image.new("L", (total_width, 2 * title_height + 3 * height + separator), 255)
    image.paste(Image.fromarray(middle), ((total_width - 2 * width) // 2, title_height))
    context_y = title_height + 2 * height + separator
    for slot, index in enumerate(meta["context_indices"]):
        image.paste(Image.fromarray(enhance(volume[index])), (slot * (width + gap), context_y + title_height))
    draw = ImageDraw.Draw(image)
    title, labels = captions(meta)

    def label(text, x, y, available):
        # Pillow's bundled font keeps rendering portable without a system font path.
        for size in range(20, 7, -1):
            font = ImageFont.load_default(size=size)
            if draw.textbbox((0, 0), text, font=font)[2] <= available:
                break
        if draw.textbbox((0, 0), text, font=font)[2] > available:
            raise ValueError("Image too narrow for truthful slice captions")
        draw.text((x, y), text, fill=0, font=font)

    label(title, 8, 8, total_width - 16)
    for slot, text in enumerate(labels):
        label(text, slot * (width + gap) + 4, context_y + 8, width - 8)
    return image, meta


def jpeg_bytes(image):
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def request(image_bytes, meta, deployment):
    # No case ID, reference label, probability, prior diagnosis, SLO or CDR is exposed.
    indices = ", ".join(map(str, meta["context_indices"]))
    text = (f"Analyze the following OCT. {ACQUISITION_CONTEXT}\n"
            f"Array shape: {meta['volume_shape']}. Central slice: {meta['central_index']}. "
            f"Context slices, left to right: {indices}. All indices are zero-based along array axis 0.")
    return dict(model=deployment, max_completion_tokens=500,
                messages=[dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=[
                    dict(type="text", text=text), dict(type="image_url", image_url=dict(
                        url="data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")))])])
