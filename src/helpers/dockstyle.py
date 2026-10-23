"""Shared style of the Fig. 6C-F interaction drawings (paper 2, twelfth version): interaction types in priority order (a residue
disc and its close-up dashed line take the first type the residue has), Chinese legend labels and colours modelled on the
BIOVIA Discovery Studio two-dimensional diagrams used by Li et al. (2026, Food Res. Int. 235, 119168, Fig. 8B-E).
No third-party imports, so that both the md venv (PyMOL/RDKit scripts) and the figure environment can import it."""
TYPES = ["常规氢键", "碳氢键", "π–σ", "π–烷基", "烷基", "范德华"]
LABEL = {"常规氢键": "常规氢键", "碳氢键": "碳氢键（C–H…O）", "π–σ": "π–σ 作用", "π–烷基": "π–烷基作用", "烷基": "烷基疏水作用",
         "范德华": "范德华作用"}
LINE = {"常规氢键": "#1E9E3A", "碳氢键": "#3DAE7C", "π–σ": "#8456C2", "π–烷基": "#E59BDA", "烷基": "#D9579D", "范德华": None}
FILL = {"常规氢键": "#4CC862", "碳氢键": "#BCE9D2", "π–σ": "#CDB5EE", "π–烷基": "#F8D6F2", "烷基": "#F4B3D3", "范德华": "#DDF4D6"}
EDGE = {"常规氢键": "#1E9E3A", "碳氢键": "#3DAE7C", "π–σ": "#8456C2", "π–烷基": "#D88BCB", "烷基": "#C9468D", "范德华": "#8CCB86"}
# three-dimensional close-ups (PyMOL colours as 0xRRGGBB)
C_RES, C_LIG, C_CARTOON, C_SURF = "0x4DB8C4", "0xF0A030", "0xBFE3EA", "0xEAF5F8"
C_CARTOON_OV = "0x7DBACB"              # overview cartoon (inside the transparent surface)
FRAME = "#E08A3C"                      # panel frames and zoom connectors (orange, as in the reference figure)
