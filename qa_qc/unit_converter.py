from qa_qc.qaqc_utils_ import u2c_


def convert_density(source_unit, source_value: float, target_unit):
    # Normalize unit strings
    unit_map = {
        "kg/m^3": "kg/m³", "kg/m**3": "kg/m³", "kg.m-3": "kg/m³", "kg m-3": "kg/m³",
        "mg/L": "mg/L", "mg L-1": "mg/L",
        "ug/l": "ug/L", "ug/L": "ug/L", "ug l-1": "ug/L",
        "mg/m^3": "mg/m³"
    }

    # Conversion to base unit: kg/m³
    to_kg_m3 = {
        "kg/m³": 1,
        "mg/L": 0.001,
        "ug/L": 0.000001,
        "mg/m³": 0.000001
    }

    # Normalize inputs
    src_unit = unit_map.get(source_unit.strip(), None)
    tgt_unit = unit_map.get(target_unit.strip(), None)

    if src_unit not in to_kg_m3 or tgt_unit not in to_kg_m3:
        raise ValueError("Unsupported unit provided.")

    # Convert source to kg/m³
    value_in_kg_m3 = source_value * to_kg_m3[src_unit]

    # Convert from kg/m³ to target unit
    converted_value = value_in_kg_m3 / to_kg_m3[tgt_unit]

    return converted_value


def convert_length(source_unit, source_value, target_unit):
    # Normalize unit strings
    unit_map = {
        "m": "m", "meters": "m",
        "mm": "mm",
        "cm": "cm",
        "km": "km",
        "nmi": "nmi",
        "inch": "in", "in": "in"
    }

    # Conversion to base unit: meters
    to_meters = {
        "m": 1,
        "mm": 0.001,
        "cm": 0.01,
        "km": 1000,
        "nmi": 1852,
        "in": 0.0254
    }

    # Normalize inputs
    src_unit = unit_map.get(source_unit.strip().lower(), None)
    tgt_unit = unit_map.get(target_unit.strip().lower(), None)

    if src_unit not in to_meters or tgt_unit not in to_meters:
        raise ValueError("Unsupported unit provided.")

    # Convert source to meters
    value_in_m = source_value * to_meters[src_unit]

    # Convert from meters to target unit
    converted_value = value_in_m / to_meters[tgt_unit]

    return converted_value


def convert_pressure(source_unit, source_value, target_unit):
    # Normalize unit strings
    unit_map = {
        "dbar": "dbar", "decibar": "dbar", "decibars": "dbar",
        "bar": "bar", "bars": "bar",
        "pa": "pa",
        "hpa": "hpa", "mbar": "hpa",
        "inhg": "inhg", "inhg": "inhg"
    }

    # Conversion to base unit: Pa
    to_pa = {
        "dbar": 10000,
        "bar": 100000,
        "pa": 1,
        "hpa": 100,
        "inhg": 3386.389
    }

    # Normalize inputs
    src_unit = unit_map.get(source_unit.strip().lower(), None)
    tgt_unit = unit_map.get(target_unit.strip().lower(), None)

    if src_unit not in to_pa or tgt_unit not in to_pa:
        raise ValueError("Unsupported unit provided.")

    # Convert source to Pa
    value_in_pa = source_value * to_pa[src_unit]

    # Convert from Pa to target unit
    converted_value = value_in_pa / to_pa[tgt_unit]

    return converted_value


def convert_photon_flux(source_unit, source_value, target_unit):
    # Normalize and map units
    unit_map = {
        "µeinsteins/s/m^2": "umol",
        "ueinsteins/s/m^2": "umol",
        "ueinsteins/s/m2": "umol",
        "umol-photons m-2 s-1": "umol",
        "mol m-2 s-1": "mol"
    }

    # Conversion to base unit: mol m⁻² s⁻¹
    to_mol = {
        "umol": 1e-6,
        "mol": 1.0
    }

    # Normalize units
    src_unit = unit_map.get(source_unit.strip().lower(), None)
    tgt_unit = unit_map.get(target_unit.strip().lower(), None)

    if src_unit not in to_mol or tgt_unit not in to_mol:
        raise ValueError("Unsupported unit provided.")

    # Convert source to mol m⁻² s⁻¹
    value_in_mol = source_value * to_mol[src_unit]

    # Convert from mol to target
    converted_value = value_in_mol / to_mol[tgt_unit]

    return converted_value


def unit_convert(unit_1, value, unit_2):
    cat_ = u2c_[unit_1.lower()]
    di__ = { "length": convert_length,
             "density":convert_density,
             "pressure":convert_pressure,
             "light_flux":convert_photon_flux
             }
    func_to_call = di__.get(cat_)
    if func_to_call:
        return func_to_call(unit_1, value, unit_2)
    else:
        raise Exception(f" [{cat_}] not found in conversion ")


if __name__ == '__main__':
    print(unit_convert("km", 1, "m"))  # Output: 1000.0
    print(unit_convert("inch", 12, "cm"))  # Output: 30.48
    print(unit_convert("nmi", 1, "km")) # Output: 1.852

    print(unit_convert("bar", 1, "pa"))  # 100000.0
    print(unit_convert("hpa", 1013.25, "inhg"))  # ~29.92
    print(unit_convert("decibar", 10, "bar"))  # 1.0
