CC      = clang
ARCH    = arm64
TARGET  = bayesian_opt
SRC     = bys.s

# -arch arm64          : emit ARM64 code
# -mmacosx-version-min : link against macOS 12 SDK minimum
# -lm                  : libm for sin/cos/exp/sqrt (part of libSystem on macOS,
#                        but -lm is harmless and documents intent)

CFLAGS  = -arch $(ARCH) -mmacosx-version-min=12.0

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean