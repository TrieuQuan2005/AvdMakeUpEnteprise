class EyeLandMark:
    # LEFT EYE
    # --- Core eye (mí chính) ---
    LEFT_EYE_CORNER = [33, 133]

    LEFT_EYE_UPPER = [159, 158, 157]
    LEFT_EYE_LOWER = [145, 153, 154]

    # --- Outer contour ---
    LEFT_EYE_OUTER = [161, 163]

    # --- Under eye (bọng mắt) ---
    LEFT_UNDER_EYE = [24, 23, 22]

    # --- Eyebrow ---
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]

    # --- Full (gộp lại nếu cần) ---
    LEFT_EYE_FULL = (
        LEFT_EYE_CORNER +
        LEFT_EYE_UPPER +
        LEFT_EYE_LOWER +
        LEFT_EYE_OUTER +
        LEFT_UNDER_EYE +
        LEFT_EYEBROW
    )

    # RIGHT EYE

    RIGHT_EYE_CORNER = [362, 263]

    RIGHT_EYE_UPPER = [387, 386, 385, 384, 398]
    RIGHT_EYE_LOWER = [373, 374, 380, 381, 382]

    RIGHT_EYE_OUTER = [466, 388, 390, 249]

    RIGHT_UNDER_EYE = [359, 255, 339, 254, 253, 252, 256]

    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295]

    RIGHT_EYE_FULL = (
        RIGHT_EYE_CORNER +
        RIGHT_EYE_UPPER +
        RIGHT_EYE_LOWER +
        RIGHT_EYE_OUTER +
        RIGHT_UNDER_EYE +
        RIGHT_EYEBROW
    )

    # =========================
    # IRIS
    # =========================

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # =========================
    # EYEBALL (tròng trắng + iris region)
    # =========================

    LEFT_EYEBALL_CORNER = [33, 133]

    LEFT_EYEBALL_UPPER = [160, 159, 158, 157, 173]
    LEFT_EYEBALL_LOWER = [144, 145, 153, 154, 155]

    LEFT_EYEBALL = (
        LEFT_EYEBALL_CORNER +
        LEFT_EYEBALL_UPPER +
        LEFT_EYEBALL_LOWER
    )

    RIGHT_EYEBALL_CORNER = [362, 263]

    RIGHT_EYEBALL_UPPER = [387, 386, 385, 384, 398]
    RIGHT_EYEBALL_LOWER = [373, 374, 380, 381, 382]

    RIGHT_EYEBALL = (
        RIGHT_EYEBALL_CORNER +
        RIGHT_EYEBALL_UPPER +
        RIGHT_EYEBALL_LOWER
    )