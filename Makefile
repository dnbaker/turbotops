# SPDX-License-Identifier: MIT
# turbotops — build the CLI.
#
#   make                  optimized for this machine (-march=native)
#   make PORTABLE=1       generic build for the target architecture
#                         (x86-64: SSE2 baseline; arm64: NEON); SIMD code
#                         paths are translated by simde, results are identical
#   make NO_SIMD=1        scalar kernels only
#   make test             build and run the self-test (parity vs ripser if found)
#   make CXX=clang++      choose the compiler (C++20 required)

CXX      ?= c++
CXXFLAGS ?= -O3
STD      := -std=c++20
WARN     := -Wall -Wextra -Wno-unused-parameter
INCLUDES := -Iinclude -isystem third_party -isystem third_party/simde
LDFLAGS  += -pthread

ifeq ($(PORTABLE),1)
  ARCH := -Wno-psabi
else
  ARCH := -march=native
endif
ifeq ($(NO_SIMD),1)
  DEFS += -DTURBOTOPS_NO_SIMD
endif

HEADERS := $(wildcard include/turbotops/*.hpp)

all: turbotops

turbotops: src/turbotops.cpp $(HEADERS)
	$(CXX) $(STD) $(CXXFLAGS) $(ARCH) $(WARN) $(DEFS) $(INCLUDES) -o $@ $< $(LDFLAGS)

test: turbotops
	./turbotops test
	sh tests/test_cli.sh
	sh tests/check_headers.sh

clean:
	rm -f turbotops

.PHONY: all test clean
