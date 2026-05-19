def split_thin_thick(eos: list, threshold=3):
    thin = [eo for eo in eos if len(eo["drivers"]) < threshold]
    thick = [eo for eo in eos if len(eo["drivers"]) >= threshold]

    return thin, thick
