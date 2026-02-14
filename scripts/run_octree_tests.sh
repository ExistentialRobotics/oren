#!/bin/bash

# 按顺序执行octree分辨率测试
echo "开始执行octree分辨率测试..."

echo "================================"
echo "执行: 10-6-0.2.yaml"
echo "================================"
python grad_sdf/trainer.py --config configs/v2/test-octree-resolution/10-6-0.2.yaml
if [ $? -ne 0 ]; then
    echo "错误: 10-6-0.2.yaml 执行失败"
    exit 1
fi

echo "================================"
echo "执行: 11-7-0.1.yaml"
echo "================================"
python grad_sdf/trainer.py --config configs/v2/test-octree-resolution/11-7-0.1.yaml
if [ $? -ne 0 ]; then
    echo "错误: 11-7-0.1.yaml 执行失败"
    exit 1
fi

echo "================================"
echo "执行: 8-5-0.7.yaml"
echo "================================"
python grad_sdf/trainer.py --config configs/v2/test-octree-resolution/8-5-0.7.yaml
if [ $? -ne 0 ]; then
    echo "错误: 8-5-0.7.yaml 执行失败"
    exit 1
fi

echo "================================"
echo "执行: 9-6-0.4.yaml"
echo "================================"
python grad_sdf/trainer.py --config configs/v2/test-octree-resolution/9-6-0.4.yaml
if [ $? -ne 0 ]; then
    echo "错误: 9-6-0.4.yaml 执行失败"
    exit 1
fi

echo "================================"
echo "所有测试执行完成!"
echo "================================"

