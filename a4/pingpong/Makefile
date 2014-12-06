SOURCES=$(wildcard *.c)

OBJS=$(SOURCES:.c=)

all: $(OBJS)

%: %.c
	mpicc -o $@ $<

clean:
	rm $(OBJS)
