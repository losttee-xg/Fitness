# OCR benchmark evaluation plan

## Current state

- Project tests currently exercise OCR payload construction and upload/candidate staging with a mocked provider. They do not measure image transcription accuracy and do not call GPT-4o.
- The existing research corpus in `data/research` is wearable/sleep data, not report-image OCR data.

## Candidate dataset

ClinOCR-Bench: https://github.com/ClinOCR-Bench/ClinOCR-Bench and https://huggingface.co/datasets/ClinOCR-Bench/ClinOCR-Bench

The public documentation describes 384 synthetic clinical-document scans, six subsets (`normal`, `handwriting`, `poor`, `rotated`, `tables`, `mixed`), paired ground-truth transcriptions, MIT license, and a roughly 114 MB release ZIP. Download and inspect the license, archive manifest, and checksums before extracting. Treat all downloaded files as untrusted; extract only into a new empty directory and do not execute archive contents.

## Dataset status and inventory

- Downloaded the official v1.0 release ZIP (119,227,531 bytes), checked the ZIP integrity (`testzip` returned no bad member), checked archive paths before extraction, and extracted into `data/research/ocr/clinocr-bench-v1.0/`.
- SHA-256: `ce1d231138050abf7f458ba5e6bd75c6ee2f3832b72f22296843da4c5ba45457`.
- Inventory: `data/research/ocr/clinocr-bench-inventory.json`; 384 scan/ground-truth pairs found. Split counts: normal 8 train/56 test; handwriting 8/56; poor 8/56; rotated 8/56; tables 8/56; mixed 16/48. 328 total test images and 56 exemplars.
- This inventory step did not call an OCR engine and therefore reports no accuracy score.

1. Keep OCR model prompt/config frozen and record a version/hash for every run.
2. Use the benchmark test split only. Never tune prompt or thresholds against the test split.
3. Upload image bytes through the report OCR adapter; retain raw OCR text/candidates, source quotes, model, response metadata and request settings in a test-only output directory.
4. Compare transcription to ground truth with character error rate and word error rate. Separately measure exact match for analyte, value, unit and date; report coverage and rejection rate as well as precision/recall.
5. Test safety cases: unsupported analyte, reference range vs patient result, absent/unclear date, ambiguous decimal/comma, cropped value, mismatch between quote and numeric value, and no hallucinated candidate for blank/non-result areas.
6. Run live requests only with an approved, newly issued API key and explicit cost budget. Use only synthetic benchmark images for external API tests. Do not transmit real user reports during this evaluation.
7. Keep the complete application suite separate from the OCR benchmark score. A passing unit suite is not an OCR accuracy claim.

## Current block

The benchmark is downloaded, integrity-checked, split and inventory-indexed. No OCR model scores exist yet: the local environment has no OCR engine and no authorized new credential was used for live GPT-4o. The live runner is capped, and inventory mode does not make billable provider calls.
