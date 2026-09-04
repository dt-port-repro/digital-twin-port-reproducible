# 附录E：数据模型与接口规范

## E.1 统一数据模型

### 集装箱实体
```json
{
  "container_id": "MSCU1234567",
  "size_type": "40",
  "weight_kg": 28500,
  "pod": "HKHKG",
  "is_reefer": false,
  "is_dangerous": false,
  "yard_position": {
    "bay": 12,
    "row": 3,
    "tier": 2
  },
  "stow_position": {
    "bay": 28,
    "row": 6,
    "tier": 4
  },
  "status": "in_yard",
  "last_updated": "2024-06-01T08:30:00Z"
}
```

### 船舶实体
```json
{
  "vessel_code": "CNTIG",
  "vessel_name": "MSC SAMUEL",
  "max_teu": 2782,
  "n_containers": 2650,
  "berth_plan_no": "5830653246812",
  "arrival_time": "2024-06-01T06:00:00Z",
  "departure_time": "2024-06-01T22:24:00Z",
  "n_pod": 3,
  "pods": ["HKHKG", "SGSIN", "CNYTN"],
  "stowage_plan": {
    "fitness": 0.6717,
    "rehandle_rate": 0.072
  }
}
```

### 堆场箱位实体
```json
{
  "cell_code": "B012R03T02",
  "bay": 12,
  "row": 3,
  "tier": 2,
  "size_type": "20",
  "allow_sizes": "20",
  "max_weight": 30000,
  "zone": "A",
  "is_reserved": false,
  "occupied_by": null
}
```

## E.2 API接口规范

### 配载服务 REST API
```
POST /api/v1/stowage/optimize
Request: {
  "vessel_code": "CNTIG",
  "containers": [...],
  "config": "D"  // A|B|C|D
}
Response: {
  "stowage_plan": {...},
  "fitness": 0.6717,
  "time_s": 82.0
}
```

### 堆场预测服务 REST API
```
POST /api/v1/prediction/yard-forecast
Request: {
  "horizon_h": 72,
  "yard_state": {...}
}
Response: {
  "mean_arrivals": 1250,
  "std_arrivals": 180,
  "confidence": 0.87,
  "type_distribution": {"20": 0.35, "40": 0.60, "45": 0.05}
}
```

### 堆场选位服务 REST API
```
POST /api/v1/yard/allocate
Request: {
  "containers": [...],
  "yard_state": {...},
  "prediction": {...},
  "config": "D"
}
Response: {
  "positions": [...],
  "avg_penalty": 0.1121,
  "time_ms": 0.03
}
```

## E.3 Kafka消息主题

| 主题名 | 数据类型 | 生产者 | 消费者 |
|--------|---------|--------|--------|
| equipment-status | 设备状态JSON | 边缘节点 | 可视化/日志 |
| container-movement | 集装箱移动JSON | TOS | 孪生模型/优化 |
| vessel-schedule | 船舶计划JSON | 计划系统 | 配载服务 |
| optimization-result | 优化结果JSON | 优化服务 | 可视化/执行 |
