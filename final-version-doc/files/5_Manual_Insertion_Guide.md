<div dir="rtl" lang="ar">

# دليل إضافة الباقي من الـ Figures يدوياً في Word

## بعد ما أبنيلك الـ docx بالـ 5 figures الأهم، عندك 14 figure آخرين تقدر تضيفهم في 10 دقايق.

---

## 📍 خريطة الـ Placeholders في الـ docx

افتح الـ docx واستخدم `Ctrl+F` (Find) للبحث عن النصوص دي، وحط الصورة المقابلة في كل مكان.

### Chapter 4 (Proposed Solution)

| Placeholder للبحث عنه | الصورة اللي تحطها | الموضع |
|----------------------|---------------------|---------|
| `[Insert Figure 4.1 — Hybrid Machine Learning Pipeline architecture diagram (fig_11_hybrid_architecture_diagram.png)]` | `fig_11_hybrid_architecture_diagram.png` | ✅ أنا هحطها |

### Chapter 5 (System Design)

| Placeholder | الصورة | الموضع |
|-------------|--------|---------|
| `[Insert Figure 5.1 — Data flow from API response through prepared train and test sets]` | اعمل diagram يدوي بسيط (PowerPoint shapes) أو **اشيل الـ placeholder تماماً** | optional |
| `[Insert Figure 5.2 — Component diagram of the project codebase]` | اعمل diagram يدوي بسيط أو **شيل الـ placeholder** | optional |
| `[Insert Figure 5.3 — Data flow diagram for a single inference]` | اعمل diagram يدوي بسيط أو **شيل الـ placeholder** | optional |
| `[Insert Figure 5.4 — Sequence diagram of the full pipeline orchestration]` | اعمل diagram يدوي بسيط أو **شيل الـ placeholder** | optional |

⚠️ **نصيحة:** الـ Chapter 5 figures مش موجودة عندنا. عندك خيارين:
- **شيل الـ placeholders كاملة** (الـ chapter لسه كامل بدونها)
- اعملها بسرعة في **draw.io** أو **PowerPoint shapes** (15 دقيقة)

### Chapter 7 (Testing and Evaluation)

الـ figures المتاحة عندك في `cicd-failure-prediction/figures/`:

| الصورة | الموضع المقترح في Chapter 7 | كيف تضيفها |
|--------|----------------------------|------------|
| `fig_01_conclusion_distribution.png` | بعد Section 7.4 introduction | Find: "The results of the full evaluation are presented in three parts" → ضع تحته |
| `fig_06_conclusion_vs_repository_heatmap.png` | بعد Table 7.1 | Find: "Table 7.1" → ضع بعد الجدول |
| `fig_12_confusion_matrices_grid.png` | في Section 7.4.1 | Find: "Section 7.4.4" → ضع قبل |
| `fig_13_roc_curves_per_target.png` | في Section 7.4.1 | بعد الـ confusion matrices |
| `fig_14_metrics_comparison_bars.png` | في Section 7.4.1 | بعد الـ ROC curves |
| `fig_15_ablation_study.png` | في Section 7.4.3 | ✅ أنا هحطها |
| `fig_17_feature_importance_global.png` | في Section 7.4.3 (نهاية) | ✅ أنا هحطها |
| `fig_20_threshold_optimization.png` | في Section 7.4.4 | ✅ أنا هحطها |
| `fig_21_metrics_before_after_threshold.png` | في Section 7.4.4 | ✅ أنا هحطها |

---

## 🛠️ كيف تضيف صورة في Word

### الخطوات (تأخذ 30 ثانية لكل صورة):

1. **افتح الـ docx**
2. **اضغط `Ctrl+F`** → اكتب الـ placeholder text
3. **اختار الـ placeholder** بالكامل (highlight)
4. **اضغط Delete** عشان تشيله
5. **اضغط `Insert` → `Pictures` → `This Device`**
6. **اختار الصورة** من مجلد `figures/`
7. **اضغط على الصورة** بعد ما تتدخل
8. **اضغط `Ctrl+E`** (Center the image)
9. **تحت الصورة، اكتب الـ caption** بصيغة:
   ```
   Figure 7.X: [الكلام اللي في caption مذكور في الورقة]
   ```

### Caption template:

اكتب الـ captions بالشكل ده تحت كل صورة:

> *Figure X.Y: Brief descriptive title of what the figure shows.*

**مثال:**
> *Figure 7.5: Ablation study showing the contribution of each feature modality. The structured-only configuration outperforms the hybrid configuration at the default threshold.*

---

## ✅ Checklist نهائي قبل التسليم

- [ ] الـ 5 figures الأهم متدخلة (هعملها أنا)
- [ ] شيلت كل `[Insert Figure ...]` placeholders اللي لسه باقية
- [ ] كل صورة في مكانها الصح
- [ ] كل صورة عندها caption تحتها
- [x] اسم المشرف مكتوب في الرسالة: Dr. Maged Mamdouh
- [ ] فحصت الـ Table of Contents لتحديثها (Right-click → Update Field)
- [ ] كل الـ tables واضحة وقابلة للقراءة
- [ ] قريت الـ Abstract وتأكدت من دقته
- [ ] فحصت الـ References (20 reference موجودة)
- [ ] حفظت بـ File → Save As → اسم نهائي زي: `MSc_Thesis_Moamen_Aly_Final.docx`

---

## 💡 نصايح إضافية

### نصيحة 1: التحكم في حجم الصور

بعد ما تدخل الصورة:
- اضغط عليها مرة واحدة
- في الـ corner، اسحب لتغيير الحجم
- **اقترح ضبط العرض على 6 inches** (يملأ الصفحة بالعرض مع margins)

### نصيحة 2: Page Breaks

لو الصورة كبيرة وقاطعة بين paragraphs:
- اضغط قبل الصورة
- اضغط `Ctrl+Enter` (Page break)
- كده الصورة هتبدأ في صفحة جديدة وتبقى واضحة

### نصيحة 3: Captions Style

لو عاوز الـ captions موحدة وأنيقة:
1. اختار النص اللي تحت الصورة
2. اضغط `Home` → اختار `Caption` من الـ Styles
3. ده هيخلي كل الـ captions نفس الشكل

### نصيحة 4: قبل التسليم

اعمل **Print Preview** (`Ctrl+P`):
- شوف الورقة كأنها مطبوعة
- اتأكد إن مفيش paragraph مقطوع بين صفحتين بشكل بشع
- اتأكد إن الصور كلها في مكانها

</div>
