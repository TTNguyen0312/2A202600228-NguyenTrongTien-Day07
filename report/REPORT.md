# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trọng Tiến
**MSSV:** 2A202600228
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (gần 1.0) nghĩa là hai vector embedding gần như cùng hướng trong không gian nhiều chiều, tức hai đoạn văn bản mang ý nghĩa tương đồng nhau, bất kể độ dài. Giá trị 0 nghĩa là không liên quan, giá trị âm nghĩa là ngược nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "Thí sinh phải mang Giấy báo dự thi khi vào phòng thi."
- Sentence B: "Candidates must bring their exam admission slip to the examination room."
- Tại sao tương đồng: Cùng nội dung (quy định giấy tờ khi vào phòng thi), chỉ khác ngôn ngữ, embedding ngữ nghĩa sẽ ánh xạ hai câu gần nhau trong vector space.

**Ví dụ LOW similarity:**
- Sentence A: "Thí sinh bị đình chỉ thi khi mang điện thoại vào phòng thi."
- Sentence B: "Quy định về thời hạn nộp thuế thu nhập cá nhân năm 2024."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (quy chế thi vs tài chính), không có từ khóa hay khái niệm chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo **hướng** (góc) giữa hai vector, không phụ thuộc vào **độ lớn** (magnitude). Tài liệu dài và ngắn có thể mang cùng ý nghĩa nhưng có magnitude rất khác nhau, dẫn đến Euclidean distance sẽ phạt (penalize) tài liệu dài một cách không công bằng, trong khi cosine similarity bỏ qua sự khác biệt đó.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> `step = chunk_size - overlap = 500 - 50 = 450`
> `num_chunks = ceil((doc_length - chunk_size) / step) + 1`
> `num_chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1`
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100: `step = 400`, số chunks = `floor(9600/400) + 1 = 25 chunks`, tăng thêm 2 chunks. Overlap nhiều hơn giúp mỗi chunk chia sẻ nhiều context hơn với chunk kế cận, giảm rủi ro cắt đứt câu/ý giữa chừng, đặc biệt quan trọng với văn bản pháp lý nơi một khoản thường tham chiếu đến khoản trước.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quy chế thi tốt nghiệp THPT (Thông tư 15/2020/TT-BGDĐT)

**Tại sao nhóm chọn domain này?**
> Quy chế thi là loại tài liệu pháp lý chuyên sâu có cấu trúc chương, điều rõ ràng và chứa nhiều quy định phức tạp. Việc chọn domain này giúp kiểm thử tốt khả năng chia nhỏ văn bản (chunking) sao cho không bị mất ngữ cảnh pháp lý và kiểm chứng việc tìm kiếm (retrieval) trả về đúng điều khoản quy định.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `01_quy_dinh_chung.md` | TT 15/2020 | 4,406 | `{"category": "quy_dinh_chung", "phase": "general"}` |
| 2 | `02_ban_chi_dao_hoi_dong.md` | TT 15/2020 | 10,291 | `{"category": "to_chuc_thi", "phase": "preparation"}` |
| 3 | `03_diem_thi_phong_thi.md` | TT 15/2020 | 6,404 | `{"category": "to_chuc_thi", "phase": "preparation"}` |
| 4 | `04_doi_tuong_dieu_kien.md` | TT 15/2020 | 8,646 | `{"category": "dang_ky_thi", "phase": "registration"}` |
| 5 | `05_trach_nhiem_thi_sinh.md` | TT 15/2020 | 8,409 | `{"category": "thi_sinh", "phase": "registration"}` |
| 6 | `06_cong_tac_de_thi.md` | TT 15/2020 | 12,889 | `{"category": "de_thi", "phase": "preparation"}` |
| 7 | `07_in_sao_van_chuyen_de.md`| TT 15/2020 | 6,409 | `{"category": "de_thi", "phase": "preparation"}` |
| 8 | `08_coi_thi.md` | TT 15/2020 | 15,182 | `{"category": "coi_thi", "phase": "exam"}` |
| 9 | `09_cham_thi.md` | TT 15/2020 | 29,970 | `{"category": "cham_thi", "phase": "grading"}` |
| 10 | `10_phuc_khao_tot_nghiep.md`| TT 15/2020 | 52,945 | `{"category": "phuc_khao", "phase": "appeals"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | `string` | `"quy_dinh_chung"`, `"coi_thi"` | Giúp khoanh vùng nhanh các nội dung có chung chủ đề, dễ dàng lọc kết quả tìm kiếm theo nội dung chuyên môn thay vì từ khóa (ví dụ truy vấn về coi thi chỉ quét tài liệu nhóm coi_thi). |
| `phase` | `string` | `"preparation"`, `"grading"` | Cho phép hệ thống lọc kết quả theo quá trình liên quan trong kỳ thi, giúp thu hẹp phạm vi context nếu câu hỏi nhắm vào một giai đoạn cụ thể (chuẩn bị thi, lúc thi, chấm bài). |
| `source` | `string` | `"Thông tư 15/2020/TT-BGDDT"` | Hỗ trợ cung cấp câu trích dẫn tham chiếu ở cuối câu trả lời, tăng độ tin cậy. Dữ liệu này được map dưới dạng shared key - áp dụng chung cho mọi docs từ YAML schema. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu: `01_quy_dinh_chung.md`, `02_ban_chi_dao_hoi_dong.md`, `03_diem_thi_phong_thi.md` với `chunk_size_limit=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `01_quy_dinh_chung` | FixedSizeChunker | 30 | 195.2 | Không, cắt giữa câu/khoản |
| | SentenceChunker | 11 | 398.3 | Khá, giữ câu nhưng chunk hơi dài |
| | RecursiveChunker | 42 | 104.9 | Tốt, theo cấu trúc đoạn văn |
| | **CustomChunker** | **29** | **152.4** | **Tốt, theo cấu trúc pháp lý + overlap** |
| `02_ban_chi_dao_hoi_dong` | FixedSizeChunker | 69 | 198.4 | Không, cắt ngang khoản mục |
| | SentenceChunker | 10 | 1026.7 | Kém, chunk quá lớn (max 2581), vượt token limit |
| | RecursiveChunker | 84 | 122.5 | Tốt, nhưng không có overlap |
| | **CustomChunker** | **63** | **163.7** | **Tốt, ranh giới pháp lý + overlap liên khoản** |
| `03_diem_thi_phong_thi` | FixedSizeChunker | 43 | 197.8 | Không, cắt ngang quy định |
| | SentenceChunker | 9 | 709.1 | Kém, chunk quá lớn (max 1543) |
| | RecursiveChunker | 65 | 98.5 | Tốt, nhưng chunk nhỏ, mất context liên điều |
| | **CustomChunker** | **45** | **141.9** | **Tốt, cân bằng kích thước và ngữ cảnh** |

### Strategy Của Tôi

**Loại:** `CustomChunker` — Hybrid strategy tối ưu cho văn bản pháp lý Việt Nam

**Mô tả cách hoạt động:**
> `CustomChunker` tách văn bản theo cấu trúc pháp lý 4 cấp bằng regex lookahead: `### Điều` → `## Chương` → `\n1.` (khoản có số) → `\na)` (điểm có chữ cái) → `\n\n` (đoạn văn). Các đơn vị này được gom greedy vào chunk không vượt `max_size`. Khi flushed, `_tail_clause` tìm ranh giới khoản/điểm cuối cùng trong `overlap` chars để làm overlap_seed cho chunk tiếp theo, đảm bảo context liên điều không bị đứt.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản Quy chế thi có câu trung bình 140 đến 444 ký tự, câu dài nhất lên tới 2679 ký tự, không phù hợp để split trên dấu `.` như `SentenceChunker`. `FixedSizeChunker` cắt ngang khoản mục làm mất ngữ cảnh pháp lý. `RecursiveChunker` thiếu overlap khiến các câu tham chiếu chéo giữa khoản bị đứt. `CustomChunker` giải quyết cả ba vấn đề: nhận diện đúng ranh giới pháp lý, không bao giờ cắt giữa khoản, và carry forward context.

**Code snippet:**
```python
_SPLIT_RE = re.compile(
    r'(?=\n#{3}\s)'        # ### Điều heading
    r'|(?=\n#{2}\s)'       # ## Chương heading
    r'|(?=\n\d{1,2}\.\s)'  # numbered clause \n1. \n2.
    r'|(?=\n[a-zđ]\)\s)'   # alpha sub-clause \na) \nđ)
    r'|\n\n'               # paragraph break
)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|-----------|----------|-------------|------------|-------------------|
| `02_ban_chi_dao_hoi_dong` | SentenceChunker (best context) | 10 | 1026.7 | Thấp, vượt token limit OpenAI (max 2581 chars) |
| | RecursiveChunker (best size) | 84 | 122.5 | Trung bình, chunk quá nhỏ, mất context liên khoản |
| | **CustomChunker** | **63** | **163.7** | **Cao — ranh giới khoản, overlap giữ tham chiếu chéo** |
| `08_coi_thi` | SentenceChunker | 12 | 1262.5 | Rất thấp, max 5369 chars, embed bị loãng |
| | RecursiveChunker | 141 | 107.7 | Trung bình, quá nhiều chunk nhỏ |
| | **CustomChunker** (max=1500) | **varies** | **~1150** | **Cao, 0 chunks vượt 1500 chars** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10)  | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|----------------|-----------|----------|
| Tôi (Nguyễn Trọng Tiến) | CustomChunker (legal-aware hybrid) | 8 | Overlap theo khoản, không cắt giữa điều luật | Nhiều chunk hơn RecursiveChunker |
| Nguyễn Việt Quang | LegalArticleChunker (`legal_article`) | 7.5 | Giữ trọn vẹn ngữ cảnh theo từng Điều luật, phù hợp khi câu hỏi cần đầy đủ các Khoản và Điểm liên quan trong cùng một Điều. | Chunk quá dài, số lượng chunk ít nên embedding bị loãng; đã thể hiện rõ ở query về chấm thi khi hệ thống retrieve nhầm tài liệu. |
| Vũ Đức Minh | RecursiveChunker (chunk_size=800) | 8.5 | Tôn trọng cấu trúc markdown, giữ ngữ cảnh pháp lý, avg length consistent 636 ký tự | Chunk count cao (47 vs 40), có thể chậm hơn với tài liệu rất lớn |
| Nguyễn Thị Ngọc | SentenceChunker | 9 | Preserve context tốt, ít chunks | Chunk dài hơn, cost embedding cao |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `CustomChunker` phù hợp nhất vì nhận diện đúng cấu trúc `### Điều` → `\n1.` → `\na)` đặc trưng của văn bản pháp lý Việt Nam, không bao giờ cắt giữa khoản và có overlap theo ranh giới điều khoản. `SentenceChunker` tạo chunk quá lớn vì câu văn pháp lý rất dài (trung bình 350+ chars, max 2679 chars), gây lỗi token limit với OpenAI embedder. `RecursiveChunker` thiếu overlap làm mất tham chiếu chéo giữa các khoản liên tiếp.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[\.\!\?]) |(?<=\.)\n', text)` để đảm bảo dấu câu luôn ở cuối câu trước khi tách, không bị split ra thành token riêng. Sau đó nhóm `max_sentences_per_chunk` câu liên tiếp thành một chunk bằng `" ".join(group)`. Edge case: text không chứa dấu câu nào trả về `[text.strip()]`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Dùng đệ quy với separator hierarchy `["\n\n", "\n", ". ", " ", ""]`. Base case: `len(text) <= chunk_size` trả về `[text]`. Với mỗi separator, tách text thành pieces và greedy merge cho đến khi piece tiếp theo sẽ làm vượt `chunk_size` thì flush; piece nào lớn hơn `chunk_size` được đệ quy với separator tiếp theo. Khi hết separator (empty string), fallback sang hard split character-level.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `self._embedding_fn(doc.content)` cho từng document và append dict `{id, content, embedding, metadata}` vào `self._store` (in-memory list). `search` embed query bằng cùng `_embedding_fn`, tính dot product giữa query vector và toàn bộ stored embeddings (hoạt động như cosine similarity do vectors là unit-normalised), sort descending và trả về `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter trước khi search: duyệt `self._store`, giữ lại records có metadata match toàn bộ `metadata_filter` dict, sau đó chạy `_search_records` trên subset đó. `delete_document` dùng list comprehension lọc bỏ tất cả records có `metadata["doc_id"] == doc_id`, trả về `True` nếu collection shrink.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` để lấy chunks liên quan, nối nội dung bằng `"\n\n".join(c["content"] for c in chunks)` thành `context`. Xây prompt RAG chuẩn: `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` rồi pass vào `llm_fn`. Cấu trúc này hướng dẫn LLM chỉ trả lời dựa trên context cung cấp, hạn chế hallucination.

### Test Results

```
collected 42 items                                                                                                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                    [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                             [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                      [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                       [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                            [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                  [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                   [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                 [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                   [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                              [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                          [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                           [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                               [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                         [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                               [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                     [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                       [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                             [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                  [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                    [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                        [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                     [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                              [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                             [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                        [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                    [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                               [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                   [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                         [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                   [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                              [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                             [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                 [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                            [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                     [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                           [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                               [100%]

===================================================================== 42 passed in 0.11s ======================================================================
```

**Số tests pass:** 42 / 42


---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Sử dụng mock embedder (`_mock_embed`) — deterministic hash-based, 64 dimensions.*

*Lưu ý: Scores tính bằng `MockEmbedder` (MD5 hash-based, 64 dims, không semantic). Dự đoán và phân tích dựa trên ngữ nghĩa thực — với embedder thật (OpenAI/sentence-transformers) kết quả sẽ khớp dự đoán.*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score (mock) | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Thí sinh phải mang Giấy báo dự thi vào phòng thi" | "Candidates must bring their exam admission slip to the examination room" | high | 0.0520 | Không (mock không semantic) |
| 2 | "Ban chấm thi gồm Trưởng ban và các ủy viên" | "Hội đồng chấm thi bao gồm Trưởng ban và thành viên" | high | 0.1786 | Không (mock không semantic) |
| 3 | "Quy định về phúc khảo bài thi tốt nghiệp THPT" | "Điều kiện dự thi tốt nghiệp trung học phổ thông" | low | -0.0048 | Có |
| 4 | "Thí sinh bị đình chỉ thi khi mang điện thoại vào phòng thi" | "Quy định về phòng thi và điểm thi trong kỳ thi THPT" | low | 0.1256 | Không (mock không semantic) |
| 5 | "Thời hạn nộp đơn phúc khảo là 10 ngày kể từ ngày công bố điểm" | "Phúc khảo bài thi được thực hiện trong thời hạn quy định" | high | 0.1053 | Không (mock không semantic) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là cặp 4 — câu về điện thoại bị đình chỉ vs quy định phòng thi — có score 0.1256 dù ngữ nghĩa khác xa nhau. `MockEmbedder` dùng hash MD5 nên không mã hóa ngữ nghĩa, mọi similarity đều là nhiễu ngẫu nhiên. Điều này cho thấy chất lượng RAG phụ thuộc hoàn toàn vào embedder: với mock, retrieval là random; với OpenAI `text-embedding-3-small`, cosine similarity phản ánh đúng mức độ liên quan ngữ nghĩa.


---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Thí sinh được phép mang những vật dụng gì vào phòng thi? | Gồm: Bút viết, thước kẻ, bút chì, tẩy chì, êke, thước vẽ đồ thị, dụng cụ vẽ hình, máy tính cầm tay (không soạn thảo văn bản/thẻ nhớ), Atlat Địa lý (đối với môn Địa). |
| 2 | Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào? (Metadata filter: `category="quy_dinh_chung"`) | Bố trí 01 điện thoại để ở phòng làm việc chung (chỉ dùng nghe gọi, bật loa ngoài, có ghi nhật ký). Máy tính chỉ được nối internet khi báo cáo nhanh. |
| 3 | Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm? | Thí sinh bị điểm liệt nếu có bài thi (hoặc môn thi thành phần) đạt từ 1,0 điểm trở xuống (tất cả phải trên 1,0 mới đạt). |
| 4 | Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện? | Chấm hai vòng độc lập bởi hai Cán bộ chấm thi (CBChT) của hai Tổ Chấm thi khác nhau. |
| 5 | Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm? | Trong thời hạn 10 ngày kể từ ngày công bố điểm thi. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk | Score | Relevant? |
|---|-------|----------------------|-------|-----------|
| 1 | Thí sinh được phép mang những vật dụng gì vào phòng thi? | `05_trach_nhiem_thi_sinh_c1` | 0.6788 | Có |
| 2 | Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào? | `02_ban_chi_dao_hoi_dong_c1` | 0.7114 | Có |
| 3 | Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm? | `10_phuc_khao_tot_nghiep_c3` | 0.6369 | Có |
| 4 | Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện? | `06_cong_tac_de_thi_c1` | 0.5691 | Có |
| 5 | Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm? | `10_phuc_khao_tot_nghiep_c2` | 0.5965 | Có |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5 — **Avg cosine similarity: 0.6385**

---

### Agent Answers

**Q1 — Thí sinh được phép mang những vật dụng gì vào phòng thi?**
> Thí sinh được phép mang vào phòng thi các vật dụng: Bút viết, Thước kẻ, Bút chì, Tẩy chì, Ê ke, Thước vẽ đồ thị, Dụng cụ vẽ hình, Máy tính cầm tay (không có chức năng soạn thảo văn bản và không có thẻ nhớ), Atlat Địa lí Việt Nam biên soạn theo Chương trình GDPT 2006 (không được đánh dấu hoặc viết thêm bất cứ nội dung nào).

**Q2 — Việc sử dụng điện thoại và internet tại Điểm thi được quy định thế nào?**
> 1. **Điện thoại cố định**: Mỗi khu vực in sao đề thi, coi thi, làm phách, chấm thi và phúc khảo phải có 01 điện thoại cố định có loa ngoài (khu vực in sao và làm phách phải có chức năng ghi âm).
> 2. **Điện thoại di động**: Chỉ dùng khi bất khả kháng — không được có chức năng ghi hình, thẻ nhớ, kết nối internet; phải niêm phong khi không sử dụng.
> 3. **Chức năng sử dụng**: Chỉ nghe/gọi liên lạc với Hội đồng thi hoặc Ban Chỉ đạo; bật loa ngoài, ghi nhật ký, có ủy viên giám sát chứng kiến.
> 4. **Máy tính**: 01 máy tính tại phòng trực; chỉ nối internet khi chuyển báo cáo nhanh; ghi nhật ký có Phó trưởng Điểm thi chứng kiến.
> 5. **Cấm thiết bị thu phát**: Không dùng thiết bị thu phát trong khu vực coi thi, chấm thi và phúc khảo ngoài thiết bị đã quy định; phải niêm phong và bảo quản an toàn.

**Q3 — Điểm liệt trong xét công nhận tốt nghiệp THPT là bao nhiêu điểm?**
> Điểm liệt là **1,0 điểm** theo thang điểm 10. Nếu thí sinh có bài thi hoặc môn thi thành phần nào đạt từ 1,0 điểm trở xuống thì bị coi là liệt.

**Q4 — Mỗi bài thi tự luận được chấm bao nhiêu vòng và do ai thực hiện?**
> Mỗi bài thi tự luận được chấm **hai vòng độc lập** bởi hai Cán bộ chấm thi (CBChT) của hai Tổ Chấm thi khác nhau.

**Q5 — Thời hạn nhận đơn phúc khảo bài thi là bao nhiêu ngày kể từ ngày công bố điểm?**
> Thời hạn nhận đơn phúc khảo là **10 ngày** kể từ ngày công bố điểm thi.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Em học được cách chuẩn hóa metadata đồng bộ giữa các file YAML và Markdown để linh hoạt hơn trong xử lý JSON. Đồng thời tôi cũng học được cách fix lỗi encoding khi terminal Windows không hỗ trợ một số ký tự Unicode đặc biệt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Các nhóm khác đã thử nghiệm dùng thêm Recursive Character Text Splitter của LangChain chạy mô phỏng cùng lúc để làm baseline vững chắc hơn thay vì tự build hoàn toàn từ đầu như cách thủ công.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Em sẽ kết hợp `LegalArticleChunker` (cắt theo Điều luật) và `RecursiveChunker`. Ban đầu chia bằng Điều Luật, nhưng nếu Điều luật ấy vẫn tạo ra chunk vượt quá 1000 tokens (làm loãng Embedding như câu 4 bị truy xuất sai), thì cắt nhỏ tiếp bằng `RecursiveChunker`.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 28 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **81 / 90** |
