# 1. Tóm tắt các Task đã thực hiện

- Task 1: Tải & sử dụng model pre-trained (Gensim)
Tải glove-wiki-gigaword-50, lấy vector 1 từ, tính cosine similarity giữa 2 từ, tìm các từ tương tự (most_similar).

- Task 2: Nhúng câu/văn bản
Triển khai embed_document(document) bằng cách tokenize → lấy vector từng từ → bỏ OOV → tính trung bình element-wise.

- Task 3: Huấn luyện Word2Vec (Gensim)
Stream file en_ewt-ud-train.txt, train Word2Vec, lưu model results/word2vec_ewt.model, load lại và dùng.

- Task 4: Huấn luyện Word2Vec (Spark)
Cài pyspark, đọc JSON lớn (C4 sample), tiền xử lý bằng Spark DataFrame, train pyspark.ml.feature.Word2Vec.

- Task 5: Trực quan hóa embedding
Lấy vectors → dùng PCA xuống 2D → scatter plot, annotate từ.

# 2. Hướng dẫn chạy code
- Chạy code với các file ở thư mục test:
## lab4_embedding_training_demo.py 
```
python lab4_embedding_training_demo.py
```
- Với kết quả chạy là:
```
Reading data from: Lab_01/data/UD_English-EWT/en_ewt-ud-train.txt
Training Word2Vec model...

Starting epoch 1...
Finished epoch 1
Average loss per token: 3.137100

Starting epoch 2...
Finished epoch 2
Average loss per token: 2.683720

Starting epoch 3...
Finished epoch 3
Average loss per token: 1.965788

Starting epoch 4...
Finished epoch 4
Average loss per token: 2.452111

Starting epoch 5...
Finished epoch 5
Average loss per token: 2.516643

Starting epoch 6...
Finished epoch 6
Average loss per token: 1.936986

Starting epoch 7...
Finished epoch 7
Average loss per token: 2.509860

Starting epoch 8...
Finished epoch 8
Average loss per token: 2.491723

Starting epoch 9...
Finished epoch 9
Average loss per token: 1.999369

Starting epoch 10...
Finished epoch 10
Average loss per token: 2.440353

Model saved to Lab_03/results/word2vec_ewt.model

Top 5 most similarity of king:
rotorua             : 0.9778
judy                : 0.9768
frightening         : 0.9749
marvelous           : 0.9749
retreat             : 0.9745

Analogy vec(woman) + vec(king) − vec(man):
[('nobody', 0.9277999401092529)]
```
## lab4_spark_word2vec_demo.py
```
python lab4_spark_word2vec_demo.py
```
- Với kết quả chạy là:
```
WARNING: Using incubator modules: jdk.incubator.vector
25/10/16 23:29:20 WARN Shell: Did not find winutils.exe: java.io.FileNotFoundException: java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset. -see https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/10/16 23:29:21 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Loaded 30000 documents
Training Word2Vec model...
25/10/16 23:29:39 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
Training completed.

Top 5 words similar to 'computer':
   computers        ->  similarity = 0.6773
   desktop          ->  similarity = 0.6286
   programming      ->  similarity = 0.6140
   laptop           ->  similarity = 0.6133
   software         ->  similarity = 0.5990
Spark session stopped.
SUCCESS: The process with PID 21928 (child process of PID 13664) has been terminated.
SUCCESS: The process with PID 13664 (child process of PID 14420) has been terminated.
SUCCESS: The process with PID 14420 (child process of PID 13884) has been terminated.
```
## lab4_test.py
```
python lab4_test.py
```
- Với kết quả là:
```
Loading model...
Model loaded successfully

Vector of king:
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]

Similarity of king and queen:
0.7839043

Similarity of king and man:
0.53093773

Top 10 most similarity of computer:
word = computers            score = 0.9165
word = software             score = 0.8815
word = technology           score = 0.8526
word = electronic           score = 0.8126
word = internet             score = 0.8060
word = computing            score = 0.8026
word = devices              score = 0.8016
word = digital              score = 0.7992
word = applications         score = 0.7913
word = pc                   score = 0.7883

Doc = The queen rules the country.
Doc vector shape: (50,)
Doc vector: [ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
 -0.33622834 -0.12495166 -0.01031508 -0.5006717   0.18690467  0.17482166
 -0.268985   -0.03096624  0.36686516  0.29983264  0.01397333 -0.06872118
 -0.3260683  -0.210115    0.16835399 -0.03151734 -0.06204716  0.04301083
 -0.06958768 -1.7792168  -0.54365396 -0.06104483 -0.17618     0.009181
  3.3916333   0.08742473 -0.4675417  -0.213435    0.02391887 -0.04470453
  0.20636833 -0.12902866 -0.28527132 -0.2431805  -0.3114423  -0.03833717
  0.11977985 -0.01418401 -0.37086335  0.22069354 -0.28848937 -0.36188802
 -0.00549529 -0.46997246]
```

# 3. Phân tích kết quả

## Nhận xét về độ tương đồng & most similar trên model pre-trained (GloVe)
- Pre-trained GloVe (glove-wiki-gigaword-50) thu được trên corpus lớn (Wikipedia + Gigaword). Vì vậy:
- Các từ mang nghĩa liên quan thường đứng gần nhau (high cosine): computer <=> computers, software, technology.

## Phân tích trực quan hóa (PCA)
- PCA 2D thường giữ được cấu trúc tổng quát:
- Từ cùng chủ đề (computer, software, programming, server) sẽ tạo cụm gần nhau.
- Từ biểu cảm xã hội/nhân vật (king, queen, man, woman) có cấu trúc "song song" thể hiện quan hệ analogy.

![]()
Quan sát thú vị (ví dụ mẫu từ kết quả chạy)

computer cụm với desktop, computers, software → phản ánh ngữ cảnh công nghệ.

Một số từ phi ngữ nghĩa hoặc stopwords có thể nằm rải rác, gây nhiễu; nên loại stopwords trước khi train/visualize.

Giải thích tại sao:

Embedding học dựa trên ngữ cảnh — từ xuất hiện trong các ngữ cảnh tương tự sẽ có vector gần nhau.

PCA giữ trục phương sai lớn nhất; nên chủ đề/trend xuất hiện nhiều trong dataset sẽ nằm gần nhau.

4.3 So sánh: Pre-trained vs Self-trained

Pre-trained (GloVe):

Học trên corpora rất lớn → captures semantics sâu, idiomatic uses.

Dùng ngay, không cần train.

Không phù hợp domain quá chuyên (cần fine-tune hoặc train lại).

Self-trained (en_ewt với Gensim):

Phù hợp với domain dữ liệu (nếu dataset đặc thù).

Kiểm soát tham số training (window, sg/cbow, negative, min_count).

Nếu dataset nhỏ → embedding kém/ít đa dạng, nhiều từ không đủ ngữ cảnh → chất lượng thấp.

Spark-trained trên C4 sample:

Có thể xử lý data lớn, phù hợp scale.

Spark MLlib Word2Vec có API hơi khác (DataFrame-based); performance & hyperparameters khác gensim.

Kết luận: Nếu bạn cần mô hình tổng quát, pre-trained lớn (GloVe/Word2Vec Google News) rất tiện. Nếu domain-specific (ví dụ log, y tế), nên train/fine-tune trên corpus domain.

# 4. Những khó khăn gặp phải & giải pháp
- Loại bỏ bớt thông tin log cho bớt rối:
```
spark.sparkContext.setLogLevel("WARN")
```
- Loss dao động khi train Gensim: Chuẩn hóa loss theo token (chia cho số token).
