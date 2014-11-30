echo -e "== 32 threads =="
./mmdriver -bin k6.bin  -block 32 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 32 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 16 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 32 512  -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 16 512  -thread 32 1 -mat 1024 -size 1024 -check

echo -e "== 64 threads =="
./mmdriver -bin k6.bin  -block 16 1024 -thread 64 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 16 1024 -thread 64 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 8  1024 -thread 64 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 16 512  -thread 64 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 8  512  -thread 64 1 -mat 1024 -size 1024 -check

echo -e "== 128 threads =="
./mmdriver -bin k6.bin  -block 8  1024 -thread 128 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 8  1024 -thread 128 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 4  1024 -thread 128 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 8  512  -thread 128 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 4  512  -thread 128 1 -mat 1024 -size 1024 -check

echo -e "== 256 threads =="
./mmdriver -bin k6.bin  -block 4  1024 -thread 256 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 4  1024 -thread 256 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 2  1024 -thread 256 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 4  512  -thread 256 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 2  512  -thread 256 1 -mat 1024 -size 1024 -check

echo -e "== 512 threads =="
./mmdriver -bin k6.bin  -block 2  1024 -thread 512 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 2  1024 -thread 512 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 1  1024 -thread 512 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 2  512  -thread 512 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 1  512  -thread 512 1 -mat 1024 -size 1024 -check
