from s2sphere import LatLng, CellId

def latlng_to_s2_tokens(lat, lng, level):
    """
    Convert a (lat, lng) pair to a full S2 token sequence.
    Returns a list of length (level+1):
      - Token 0: Cube face (an integer in [0,5])
      - Tokens 1..level: Each token is in [0,3] representing the child index at that level.
    """
    # Use s2sphere’s native conversion.
    latlng = LatLng.from_degrees(lat, lng)
    cell = CellId.from_lat_lng(latlng).parent(level)
    face = cell.face()
    tokens = [face]
    cell_id_int = cell.id()
    # Extract the 2*k child tokens.
    # The Hilbert position is encoded in the next 2*level bits.
    shift = 61 - (2 * level)
    pos_bits = (cell_id_int >> shift) & ((1 << (2 * level)) - 1)
    for i in range(level):
        shift_i = 2 * (level - i - 1)
        token = (pos_bits >> shift_i) & 0x3
        tokens.append(token)
    return tokens

def group_s2_tokens(full_tokens, group_size=5):
    """
    Group the child tokens into groups of size `group_size`. The first token (face) remains unchanged.
    For example, if full_tokens has length 21 and group_size=4, then the grouped tokens
    will have length 1 + (20/4) = 6.
    """
    if len(full_tokens) < 1 or (len(full_tokens) - 1) % group_size != 0:
        raise ValueError("Full token sequence length minus one must be divisible by group_size")
    grouped = [full_tokens[0]]
    for i in range(1, len(full_tokens), group_size):
        group = full_tokens[i:i+group_size]
        val = 0
        for digit in group:
            val = val * 4 + digit
        grouped.append(val)
    return grouped

def ungroup_s2_tokens(grouped_tokens, group_size=5):
    """
    Reconstruct the full token sequence from the grouped token sequence.
    The first token is unchanged; each subsequent grouped token is converted back into group_size tokens.
    """
    full = [grouped_tokens[0]]
    for val in grouped_tokens[1:]:
        group = []
        for i in range(group_size):
            digit = (val // (4 ** (group_size - i - 1))) % 4
            group.append(digit)
        full.extend(group)
    return full

def s2_tokens_to_latlng(tokens, level):
    """
    Reconstruct the S2 cell from a full token sequence and return its center (lat, lng).
    The S2 cell id for a cell at level k is constructed as:
    
       cell_id = (face << 61) | (pos << (61 - 2*k)) | (1 << ((61 - 2*k) - 1))
    
    where tokens is a list of length (level+1) with:
      - tokens[0]: cube face (0–5)
      - tokens[1:] (k tokens): each in [0,3]
    
    Returns:
       (lat, lng) in degrees.
    """
    if len(tokens) != level + 1:
        print(len(tokens))
        raise ValueError("Token sequence length must be level+1")
    face = tokens[0]
    pos = 0
    for token in tokens[1:]:
        pos = (pos << 2) | (token & 0x3)
    POS_BITS = 61
    shift = POS_BITS - 2 * level
    lsb = 1 << (shift - 1)  # The lowest bit of the cell at this level.
    cell_id_int = (face << POS_BITS) | (pos << shift) | lsb
    cell = CellId(cell_id_int)
    latlng = cell.to_lat_lng()
    lat_val = latlng.lat().degrees
    lng_val = latlng.lng().degrees
    return lat_val, lng_val

if __name__ == "__main__":
    # Demo: test with an example location.
    lat, lng = 83.7249, -92.4194
    level = 20
    full_tokens = latlng_to_s2_tokens(lat, lng, level)
    print("Full S2 tokens (length {}):".format(len(full_tokens)))
    print(full_tokens)
    grouped = group_s2_tokens(full_tokens, group_size=4)
    print("Grouped tokens (length {}):".format(len(grouped)))
    print(grouped)
    recovered = ungroup_s2_tokens(grouped, group_size=4)
    print("Recovered full tokens:")
    print(recovered)
    decoded_lat, decoded_lng = s2_tokens_to_latlng(recovered, level)
    print("Decoded lat/lng:")
    print(decoded_lat, decoded_lng)
