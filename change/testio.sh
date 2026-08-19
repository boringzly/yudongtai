#!/bin/bash

# 设置测试文件路径（可以自定义目录）
OUTFILE="./write_speed_test.tmp"

echo "➡️ 测试1：带缓存的写入（dd 默认行为）"
dd if=/dev/zero of="$OUTFILE" bs=1M count=1024 status=progress
sync
echo

echo "➡️ 测试2：关闭缓存的写入（oflag=direct，更真实）"
dd if=/dev/zero of="$OUTFILE" bs=1M count=1024 oflag=direct status=progress
sync
echo

rm -f "$OUTFILE"
echo "✅ 测试完成，临时文件已删除。"

