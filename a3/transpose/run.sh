./mmdriver -bin k6.bin  -block 32 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k7.bin  -block 32 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k8.bin  -block 16 1024 -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k9.bin  -block 32 512  -thread 32 1 -mat 1024 -size 1024 -check
./mmdriver -bin k10.bin -block 16 512  -thread 32 1 -mat 1024 -size 1024 -check
