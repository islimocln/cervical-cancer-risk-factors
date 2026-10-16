# Serviks Kanseri Risk Faktörleri ve Biyopsi Sonucu

**Ders:** Biyomedikal Verilerin Analizi ve Yapay Zeka
**Öğrenci:** İslim
**Veri seti:** [Cervical Cancer (Risk Factors) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
(858 hasta · 36 öznitelik · eksik değerler `?` ile gösterilmiş · lisans: CC BY 4.0)

## Araştırma sorusu

Hastaların cevaplamadığı (eksik) sorular rastgele mi; eksikliğin kendisi riskle ilişkili mi?
Bu yapı altında **kalibre ve dürüst** bir risk modeli kurulabilir mi?

**Hedef değişken:** `Biopsy` (biyopsi sonucu; tanıda altın standart, 55 pozitif, %6.4).
Aynı muayenede yapılan `Hinselmann`, `Schiller` ve `Citology` testleri **veri sızıntısı** yaratacağı için girdi olarak kullanılmadı.

## Proje durumu

| Adım | Notebook | Durum |
|---|---|---|
| Veri inceleme ve öznitelik seçimi | [`01_veri_inceleme.ipynb`](notebooks/01_veri_inceleme.ipynb) | ✅ |
| Eksik veri mekanizması (MCAR/MAR/MNAR) | [`02_eksik_veri_analizi.ipynb`](notebooks/02_eksik_veri_analizi.ipynb) | ✅ |
| Göstergeli ve göstergesiz imputasyon (Pipeline) | [`03_imputasyon_modelleme.ipynb`](notebooks/03_imputasyon_modelleme.ipynb) | ✅ |
| PR-AUC, kalibrasyon, karar eğrisi, bootstrap | `04_degerlendirme.ipynb` | ⏳ |
| SHAP ve etik tartışma | `05_shap_yorumlama.ipynb` | ⏳ |
| LaTeX rapor | `report/` | ⏳ |

## Şu ana kadarki bulgular

**01 · Veri inceleme**
- 36 sütundan **15 öznitelik** seçildi. 20 sütun çıkarıldı: sabit (2), çok nadir (6), türetilmiş/tekrar eden (6), yapısal eksik (2), gizli eksik (1), sızıntı (3).
- Üç tür eksiklik bulundu: **gerçek** (hasta cevaplamamış), **yapısal** (soru o hasta için anlamsız) ve **gizli** (eksik değer yerine `0` yazılmış).
- 2 imkânsız değer (ilk ilişki yaşı > şimdiki yaş) ve 5 tutarsız `Dx` kaydı tespit edildi.

**02 · Eksik veri analizi**
- Hassas sorular (hormonal kontraseptif, RİA, CYBH) **blok halinde** cevapsız bırakılmış (phi ≈ 0.9).
- Cevapsız bırakanlar daha genç (medyan 23 / 26 yaş, Holm sonrası p < 0.01) → veri **MCAR değil**, en az MAR.
- Hormonal kontraseptif sorusunu cevaplamayan 108 hastanın **hiçbirinde** biyopsi pozitif değil (cevaplayanlarda %7.3). Fisher p ≈ 0.001, Holm sonrası p ≈ 0.008, yaşa göre düzeltilmiş Mantel-Haenszel p ≈ 0.01.
- Little MCAR testi tek başına zayıf kaldı (p ≈ 0.48). Hedefe yönelik testler daha bilgilendirici oldu.

**03 · İmputasyon ve modelleme** (5 katlı katmanlı CV × 10 tekrar)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Rastgele (herkese %6.4) | 0.064 | 0.50 |
| Sadece yaş | 0.127 | 0.56 |
| Lojistik regresyon (en iyi düzen) | 0.170 | 0.62 |
| **Rastgele orman · medyan + grup eksiklik göstergesi** | **0.194** | **0.69** |

- Risk faktörleri gerçek ama **zayıf** bir sinyal taşıyor.
- Eksiklik göstergeleri ROC-AUC'u artırma eğiliminde (+0.03, p ≈ 0.055), PR-AUC'u neredeyse değiştirmiyor.
- MICE, basit medyan imputasyonundan daha iyi sonuç vermedi.

## Dizin yapısı

```
cervical_cancer_risk_factor/
├── data/                  ham veri (depoda yok; UCI'den indirilir, bkz. Çalıştırma)
├── notebooks/             analiz adımları (01 → 05 sırayla çalıştırılır)
├── src/utils.py           veri temizleme, Pipeline, Little MCAR testi, istatistik fonksiyonları
├── results/
│   ├── figures/           rapora girecek grafikler
│   └── tables/            metrik ve test tabloları
├── report/                LaTeX rapor
├── requirements.txt
└── README.md
```

## Yöntem notları

- **Sızıntı önleme:** İmputasyon, ölçekleme ve eksiklik göstergeleri `Pipeline` içinde; her çapraz doğrulama katında yalnızca eğitim verisinden öğrenilir.
- **Ham veri değiştirilmez:** Veri hataları `src/utils.py` içindeki `clean()` fonksiyonuyla her çalıştırmada bellekte düzeltilir.
- **Satır silinmez:** Complete-case analizi pozitif örneklerin %18'ini ve eksikliğin taşıdığı bilgiyi kaybettirir.
- **Model karşılaştırması:** Tekrarlı CV'de katlar bağımsız olmadığı için Nadeau-Bengio düzeltilmiş t-testi kullanıldı.

## Çalıştırma

Python 3.12 ile test edildi.

**1. Veri seti:** https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

İndirilen `risk_factors_cervical_cancer.csv` dosyasını `data/` klasörüne koy:

```
data/risk_factors_cervical_cancer.csv
```

**2. Kütüphaneleri kur ve notebook'ları aç:**

```bash
pip install -r requirements.txt
jupyter lab
```

Notebook'lar `notebooks/` klasörü içinden çalıştırılmalıdır (veri yolları `../data/` şeklindedir).

## Kaynaklar

- Fernandes K, Cardoso JS, Fernandes J. *Transfer learning with partial observability applied to cervical cancer screening.* Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA), 2017.
- Sterne JAC ve ark. *Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.* BMJ 2009;338:b2393.
- Little RJA. *A test of missing completely at random for multivariate data with missing values.* JASA 1988;83(404):1198–1202.
- Nadeau C, Bengio Y. *Inference for the generalization error.* Machine Learning 2003;52:239–281.
- Rubin DB. *Inference and missing data.* Biometrika 1976;63(3):581–592.
