from microsim.schema.sample import Fluorophore

def fetch_FPs(fp_names: list[str]) -> list[Fluorophore]:
    return [Fluorophore.from_fpbase(name=fp_name) for fp_name in fp_names]