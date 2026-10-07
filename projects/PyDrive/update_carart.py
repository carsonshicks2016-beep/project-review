import sys
import re

def main():
    with open("supra/carart.py", "r") as f:
        content = f.read()

    # Find the block for porsche_919evo
    match = re.search(r'( +elif name == "porsche_919evo":.*?)( +elif name == "skyline":)', content, re.DOTALL)
    if not match:
        print("Could not find porsche_919evo block")
        return
    
    original_block = match.group(1)
    skyline_block = match.group(2)
    
    # Create the legacy block by renaming
    legacy_block = original_block.replace('elif name == "porsche_919evo":', 'elif name == "porsche_919_legacy":')
    legacy_block = legacy_block.replace('Porsche 919 Hybrid Evo — modern LMP1 prototype:', 'Legacy Porsche 919 Evo (Original visual proxy)')
    
    # Create the NEW highly detailed porsche_919evo block
    new_block = """    elif name == "porsche_919evo":
        # ------------------------------------------------------------------
        # Porsche 919 Hybrid Evo — HERITAGE RACING EDITION (Option 2)
        #   * Detailed multi-color racing stripes (red and black)
        #   * High-detail LED headlight arrays
        #   * Refined aerodynamic splitters, dive planes, and louvers
        # ------------------------------------------------------------------
        HW_STRIPE = 0.15 * HW
        
        vertices = [
            # 0..7: Centerline (Y=0)
            (1.0000 * HL, 0.0, 0.05),   # 0: nose tip
            (0.8826 * HL, 0.0, 0.14),   # 1: nose crown
            (0.2000 * HL, 0.0, 0.36),   # 2: screen base
            (-0.0200 * HL, 0.0, 0.68),  # 3: canopy peak
            (-0.2400 * HL, 0.0, 0.60),  # 4: canopy trailing edge
            (-0.4600 * HL, 0.0, 0.44),  # 5: engine deck
            (-0.9182 * HL, 0.0, 0.40),  # 6: tail deck end
            (-0.9364 * HL, 0.0, 0.10),  # 7: tail floor

            # 8..15: Left Stripe Edge (Y = HW_STRIPE)
            (1.0000 * HL, HW_STRIPE, 0.05),
            (0.8826 * HL, HW_STRIPE, 0.14),
            (0.2000 * HL, HW_STRIPE, 0.36),
            (-0.0200 * HL, HW_STRIPE, 0.68),
            (-0.2400 * HL, HW_STRIPE, 0.60),
            (-0.4600 * HL, HW_STRIPE, 0.44),
            (-0.9182 * HL, HW_STRIPE, 0.40),
            (-0.9364 * HL, HW_STRIPE, 0.10),

            # 16..23: Right Stripe Edge (Y = -HW_STRIPE)
            (1.0000 * HL, -HW_STRIPE, 0.05),
            (0.8826 * HL, -HW_STRIPE, 0.14),
            (0.2000 * HL, -HW_STRIPE, 0.36),
            (-0.0200 * HL, -HW_STRIPE, 0.68),
            (-0.2400 * HL, -HW_STRIPE, 0.60),
            (-0.4600 * HL, -HW_STRIPE, 0.44),
            (-0.9182 * HL, -HW_STRIPE, 0.40),
            (-0.9364 * HL, -HW_STRIPE, 0.10),
            
            # 24..35: Left side outer
            (0.9478 * HL, 0.8400 * HW, 0.06),   # 24: splitter corner
            (0.8400 * HL, 0.6400 * HW, 0.16),   # 25: nose shoulder
            (0.5800 * HL, 0.9769 * HW, 0.32),   # 26: front arch crown
            (0.2000 * HL, 0.5000 * HW, 0.34),   # 27: screen base outer
            (-0.0200 * HL, 0.2400 * HW, 0.60),  # 28: canopy side front
            (-0.2400 * HL, 0.2400 * HW, 0.54),  # 29: canopy side rear
            (-0.4600 * HL, 0.6400 * HW, 0.42),  # 30: sidepod / deck shoulder
            (-0.6600 * HL, 0.9769 * HW, 0.38),  # 31: rear arch crown
            (-0.9091 * HL, 0.9308 * HW, 0.37),  # 32: tail corner
            (-0.9273 * HL, 0.9385 * HW, 0.09),  # 33: tail floor corner
            (0.4000 * HL, 0.9923 * HW, 0.06),   # 34: front skirt
            (-0.3000 * HL, 1.0000 * HW, 0.06),  # 35: rear skirt
            
            # 36..47: Right side outer (will be mirrored)
        ]
        
        # Mirror Left side outer to Right side outer
        for idx in range(24, 36):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))
            
        vertices += [
            # 48..55: Left Headlight LEDs
            (0.7000 * HL, 0.7000 * HW, 0.20), (0.7500 * HL, 0.6500 * HW, 0.18),
            (0.7000 * HL, 0.8500 * HW, 0.22), (0.7500 * HL, 0.8000 * HW, 0.20),
            (0.6000 * HL, 0.7500 * HW, 0.26), (0.6500 * HL, 0.7000 * HW, 0.24),
            (0.6000 * HL, 0.9000 * HW, 0.28), (0.6500 * HL, 0.8500 * HW, 0.26),

            # 56..63: Right Headlight LEDs
            (0.7000 * HL, -0.7000 * HW, 0.20), (0.7500 * HL, -0.6500 * HW, 0.18),
            (0.7000 * HL, -0.8500 * HW, 0.22), (0.7500 * HL, -0.8000 * HW, 0.20),
            (0.6000 * HL, -0.7500 * HW, 0.26), (0.6500 * HL, -0.7000 * HW, 0.24),
            (0.6000 * HL, -0.9000 * HW, 0.28), (0.6500 * HL, -0.8500 * HW, 0.26),
            
            # Shark fin (64..67 L face, 68..71 R face)
            (-0.2400 * HL, 0.0450 * HW, 0.60),  # 64: fin base front L
            (-0.3400 * HL, 0.0450 * HW, 0.86),  # 65: fin top front L
            (-0.8909 * HL, 0.0450 * HW, 0.72),  # 66: fin top rear L
            (-0.9091 * HL, 0.0450 * HW, 0.42),  # 67: fin base rear L
            (-0.2400 * HL, -0.0450 * HW, 0.60), # 68: fin base front R
            (-0.3400 * HL, -0.0450 * HW, 0.86), # 69: fin top front R
            (-0.8909 * HL, -0.0450 * HW, 0.72), # 70: fin top rear R
            (-0.9091 * HL, -0.0450 * HW, 0.42), # 71: fin base rear R

            # Rear wing (72..83)
            (-0.8818 * HL, 0.9077 * HW, 0.46),   # 72: L plate lower front
            (-0.8818 * HL, 0.9077 * HW, 1.00),   # 73: L plate upper front
            (-1.0000 * HL, 0.9077 * HW, 1.04),   # 74: L plate upper rear
            (-1.0000 * HL, 0.9077 * HW, 0.52),   # 75: L plate lower rear
            (-0.8818 * HL, -0.9077 * HW, 0.46),  # 76: R plate lower front
            (-0.8818 * HL, -0.9077 * HW, 1.00),  # 77: R plate upper front
            (-1.0000 * HL, -0.9077 * HW, 1.04),  # 78: R plate upper rear
            (-1.0000 * HL, -0.9077 * HW, 0.52),  # 79: R plate lower rear
            (-0.8636 * HL, 0.3000 * HW, 0.42),   # 80: L pylon base
            (-0.9182 * HL, 0.3000 * HW, 0.96),   # 81: L pylon top
            (-0.8636 * HL, -0.3000 * HW, 0.42),  # 82: R pylon base
            (-0.9182 * HL, -0.3000 * HW, 0.96),  # 83: R pylon top
            
            # Front dive planes (Canards) (84..87 L, 88..91 R)
            (0.7400 * HL, 0.7000 * HW, 0.19),    # 84
            (0.7400 * HL, 0.9538 * HW, 0.23),    # 85
            (0.5600 * HL, 0.9538 * HW, 0.23),    # 86
            (0.5600 * HL, 0.7000 * HW, 0.19),    # 87
            (0.7400 * HL, -0.7000 * HW, 0.19),   # 88
            (0.7400 * HL, -0.9538 * HW, 0.23),   # 89
            (0.5600 * HL, -0.9538 * HW, 0.23),   # 90
            (0.5600 * HL, -0.7000 * HW, 0.19),   # 91
        ]

        c_white = (238, 240, 242)       # Porsche Motorsport white
        c_red = (196, 30, 42)           # Porsche red flash
        c_black = (18, 20, 24)
        c_carbon = (12, 14, 17)
        c_led = (245, 250, 255)         # Bright LED white
        c_stripe_black = (20, 20, 20)

        faces = [
            # Splitter
            ("splitter_center", [0, 8, 16], (0.9, 0.0, 0.22), c_carbon),
            ("splitter_L", [8, 24, 0], (0.9, 0.2, 0.22), c_carbon),
            ("splitter_R", [16, 0, 36], (0.9, -0.2, 0.22), c_carbon),

            # Nose Stripe (Red & Black)
            ("nose_stripe_red", [0, 1, 9, 8], (0.45, 0.0, 0.86), c_red),
            ("nose_stripe_red_R", [0, 16, 17, 1], (0.45, 0.0, 0.86), c_red),
            
            # Nose Body (White)
            ("nose_L", [8, 9, 25, 24], (0.45, 0.22, 0.86), c_white),
            ("nose_R", [16, 36, 37, 17], (0.45, -0.22, 0.86), c_white),

            # Hood Stripe
            ("hood_stripe_black_L", [1, 2, 10, 9], (0.30, 0.0, 0.94), c_stripe_black),
            ("hood_stripe_black_R", [1, 17, 18, 2], (0.30, 0.0, 0.94), c_stripe_black),

            # Hood Body (White)
            ("hood_L", [9, 10, 27, 25], (0.30, 0.2, 0.94), c_white),
            ("hood_R", [17, 37, 39, 18], (0.30, -0.2, 0.94), c_white),

            # Canopy Glass (Center stripe is glass here)
            ("glass_C_L", [2, 3, 11, 10], (0.50, 0.0, 0.82), "glass"),
            ("glass_C_R", [2, 18, 19, 3], (0.50, 0.0, 0.82), "glass"),
            ("glass_L", [10, 11, 28, 27], (0.50, 0.2, 0.82), "glass"),
            ("glass_R", [18, 39, 40, 19], (0.50, -0.2, 0.82), "glass"),

            ("canopy_C_L", [3, 4, 12, 11], (0.0, 0.0, 0.95), "glass"),
            ("canopy_C_R", [3, 19, 20, 4], (0.0, 0.0, 0.95), "glass"),
            ("canopy_L", [11, 12, 29, 28], (0.0, 0.28, 0.95), "glass"),
            ("canopy_R", [19, 40, 41, 20], (0.0, -0.28, 0.95), "glass"),
            
            # Side Glass
            ("side_glass_left", [27, 28, 29, 30], (0.0, 0.90, 0.40), "glass"),
            ("side_glass_right", [39, 42, 41, 40], (0.0, -0.90, 0.40), "glass"),
            
            # Rear Glass / Engine Deck Start
            ("rear_glass_L", [4, 5, 13, 12], (-0.50, 0.1, 0.80), "glass"),
            ("rear_glass_R", [4, 20, 21, 5], (-0.50, -0.1, 0.80), "glass"),
            ("deck_stripe_red_L", [5, 6, 14, 13], (-0.10, 0.0, 0.96), c_red),
            ("deck_stripe_red_R", [5, 21, 22, 6], (-0.10, 0.0, 0.96), c_red),
            
            # Engine Deck Outer (White)
            ("deck_L", [13, 14, 32, 30], (-0.10, 0.25, 0.96), c_white),
            ("deck_R", [21, 42, 44, 22], (-0.10, -0.25, 0.96), c_white),
            
            # Front Fenders (White)
            ("fender_top_left", [25, 27, 26], (0.20, 0.40, 0.89), c_white),
            ("fender_top_right", [37, 38, 39], (0.20, -0.40, 0.89), c_white),
            ("fender_side_left", [24, 25, 26, 34], (0.25, 0.95, 0.0), c_white),
            ("fender_side_right", [36, 46, 38, 37], (0.25, -0.95, 0.0), c_white),
            
            # Flanks & Sidepods (White body, Black lower skirt)
            ("flank_upper_left", [26, 27, 30, 31], (0.05, 0.98, 0.15), c_white),
            ("flank_upper_right", [38, 43, 42, 39], (0.05, -0.98, 0.15), c_white),
            ("flank_lower_left", [34, 26, 31, 35], (0.0, 1.0, 0.0), c_carbon),
            ("flank_lower_right", [46, 47, 43, 38], (0.0, -1.0, 0.0), c_carbon),
            ("haunch_left", [35, 31, 32, 33], (-0.25, 0.95, 0.0), c_white),
            ("haunch_right", [47, 45, 44, 43], (-0.25, -0.95, 0.0), c_white),

            # LED Headlights (Highly Detailed)
            ("led_fl_1", [48, 49, 51, 50], (0.6, 0.6, 0.8), c_led),
            ("led_fl_2", [52, 53, 55, 54], (0.6, 0.6, 0.8), c_led),
            ("led_fr_1", [56, 57, 59, 58], (0.6, -0.6, 0.8), c_led),
            ("led_fr_2", [60, 61, 63, 62], (0.6, -0.6, 0.8), c_led),

            # Kamm Tail / Diffuser
            ("tail_C_L", [6, 7, 15, 14], (-0.96, 0.0, 0.0), c_black),
            ("tail_C_R", [6, 22, 23, 7], (-0.96, 0.0, 0.0), c_black),
            ("tail_L", [14, 15, 33, 32], (-0.96, 0.20, 0.0), c_carbon),
            ("tail_R", [22, 44, 45, 23], (-0.96, -0.20, 0.0), c_carbon),

            # Shark fin
            ("fin_left", [64, 65, 66, 67], (0.0, 1.0, 0.05), c_stripe_black),
            ("fin_right", [68, 69, 70, 71], (0.0, -1.0, 0.05), c_stripe_black),
            ("fin_top", [65, 66, 70, 69], (0.0, 0.0, 1.0), c_red),
            ("fin_front", [64, 65, 69, 68], (0.6, 0.0, 0.5), c_stripe_black),

            # Front dive planes (Canards)
            ("canard_left", [84, 85, 86, 87], (0.1, 0.2, 0.95), c_carbon),
            ("canard_right", [88, 91, 90, 89], (0.1, -0.2, 0.95), c_carbon),

            # Rear wing (Red edges, carbon blades, white endplates)
            ("wing_L_plate", [72, 73, 74, 75], (0.0, 1.0, 0.0), c_white),
            ("wing_R_plate", [76, 79, 78, 77], (0.0, -1.0, 0.0), c_white),
            ("wing_blade_top", [73, 74, 78, 77], (0.0, 0.0, 1.0), c_carbon),
            ("wing_blade_front", [72, 73, 77, 76], (0.85, 0.0, 0.25), c_red),
            ("wing_blade_rear", [74, 75, 79, 78], (-0.85, 0.0, 0.25), c_red),
            ("wing_pylon_left", [80, 81, 73, 72], (0.0, 1.0, 0.0), c_carbon),
            ("wing_pylon_right", [82, 76, 77, 83], (0.0, -1.0, 0.0), c_carbon),
        ]
"""
    
    new_content = content[:match.start()] + legacy_block + "\n" + new_block + skyline_block + content[match.end():]
    
    with open("supra/carart.py", "w") as f:
        f.write(new_content)
    
    print("Successfully updated supra/carart.py!")

if __name__ == "__main__":
    main()
