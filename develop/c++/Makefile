# Makefile for C++ library
CC = g++
CFLAGS = -shared -fPIC -O2
TARGET = libmymath.so
SOURCE = my_math.cpp

# 默认目标
all: $(TARGET)

# 编译动态库
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# 清理
clean:
	rm -f $(TARGET)

# 安装到系统路径（可选）
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	ldconfig

# 显示帮助
help:
	@echo "可用的目标:"
	@echo "  all     - 编译动态库"
	@echo "  clean   - 清理编译文件"
	@echo "  install - 安装到系统路径"
	@echo "  help    - 显示此帮助信息"

.PHONY: all clean install help 