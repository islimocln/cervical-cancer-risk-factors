# Veri

Ham veri depoda tutulmaz (`.gitignore`: `data/*.csv`). Defterleri çalıştırmadan önce dosyayı UCI'den indirip bu klasöre koyun.

## İndirme

1. Sayfayı açın: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
2. **Download** ile gelen zip dosyasını açın.
3. İçindeki `risk_factors_cervical_cancer.csv` dosyasını bu klasöre kopyalayın:

```
data/risk_factors_cervical_cancer.csv
```

Dosya adı değiştirilmemelidir; `src/utils.py` içindeki `load_raw()` bu yolu bekler.

## Beklenen içerik

| | |
|---|---|
| Satır | 858 (her satır bir hasta) |
| Sütun | 36 |
| Eksik değer | `?` ile gösterilir |
| Hedef | `Biopsy` (55 pozitif, %6.4) |

Ham dosya hiçbir adımda değiştirilmez. Veri hataları her çalıştırmada bellekte, `src/utils.py` içindeki `clean()` ile düzeltilir.

## Kaynak ve lisans

Fernandes K, Cardoso JS, Fernandes J. *Transfer learning with partial observability applied to cervical cancer screening.* IbPRIA, 2017.
Veri: Hospital Universitario de Caracas, Venezuela. Lisans: CC BY 4.0.
