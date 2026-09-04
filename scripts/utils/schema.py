"""数据模式定义 — CTOS原始中文列名 → 标准英文列名映射。

每个表定义为 TableDef，包含 ColumnDef 列表。
每个 ColumnDef 定义 name（标准名）、raw_names（原始中文名变体）、dtype。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ColumnDef:
    """单列定义"""
    name: str                        # 标准化列名（英文下划线）
    raw_names: List[str] = field(default_factory=list)  # CTOS原始列名（含中文）
    dtype: str = "str"               # str / int / float / datetime / bool


@dataclass
class TableDef:
    """表定义"""
    name: str
    columns: List[ColumnDef]


# ══════════════════════════════════════════════════════════════
# 表一：船舶基本资料 (01_vessels.parquet)
# ══════════════════════════════════════════════════════════════
VESSELS = TableDef("vessels", [
    ColumnDef("berth_plan_no", ["BERTHPLANNO", "BERTH_PLAN_NO", "船舶编号"], "str"),
    ColumnDef("e_vessel_name", ["EVESSELNAME", "E_VESSEL_NAME"], "str"),
    ColumnDef("c_vessel_name", ["CVESSELNAME", "C_VESSEL_NAME"], "str"),
    ColumnDef("imo", ["IMO"], "str"),
    ColumnDef("terminal_codes", ["TERMINALCODES", "TERMINAL_CODE"], "str"),
    ColumnDef("max_teu", ["最大TEU容量", "MAX_TEU", "TEU容量"], "float"),
    ColumnDef("length", ["LENGTH", "船长"], "float"),
    ColumnDef("width", ["WIDTH", "船宽"], "float"),
    ColumnDef("height", ["HEIGHT", "船高"], "float"),
    ColumnDef("gross_weight", ["GROSSWEIGHT", "总吨"], "float"),
    ColumnDef("dead_weight", ["DEADWEIGHT", "载重吨"], "float"),
    ColumnDef("net_weight", ["NETWEIGHT", "净吨"], "int"),
    ColumnDef("max_bay", ["TEU长（最大贝）", "MAX_BAY", "最大贝位"], "int"),
    ColumnDef("max_row", ["TEU宽（最大列）", "MAX_ROW", "最大列"], "int"),
    ColumnDef("max_tier", ["层高", "MAX_TIER", "最大层"], "float"),
    ColumnDef("hatch_width", ["舱盖宽度(最大/米)", "HATCH_WIDTH"], "float"),
    ColumnDef("hatch_weight", ["舱盖重量(最大/吨)", "HATCH_WEIGHT"], "float"),
    ColumnDef("vessel_type", ["船舶类型。B-大船 S-驳船", "VESSEL_TYPE", "船舶类型"], "str"),
    ColumnDef("berth_time", ["靠泊时间"], "datetime"),
])

# ══════════════════════════════════════════════════════════════
# 表二：船舶贝位结构 (02_bay_structure.parquet)
# ══════════════════════════════════════════════════════════════
BAY_STRUCTURE = TableDef("bay_structure", [
    ColumnDef("VESSELTYPECODE", ["VESSELTYPECODE", "船型代码"], "str"),
    ColumnDef("custom_cell", ["CUSTOMCELL", "自定义箱位"], "str"),
    ColumnDef("custom_bay", ["CUSTOMBAY", "自定义贝位"], "int"),
    ColumnDef("custom_stack", ["CUSTOMSTACK", "自定义列"], "int"),
    ColumnDef("CUSTOMTIER", ["CUSTOMTIER", "自定义层"], "int"),
    ColumnDef("TYPEA", ["TYPEA", "类型A"], "str"),
    ColumnDef("iso_cell", ["ISOCELL", "ISO箱位"], "str"),
    ColumnDef("iso_bay", ["ISOBAY", "ISO贝位"], "int"),
    ColumnDef("iso_stack", ["ISOSTACK", "ISO列"], "int"),
    ColumnDef("iso_tier", ["ISOTIER", "ISO层"], "int"),
    ColumnDef("STARTTIERNO", ["STARTTIERNO", "起始层号"], "int"),
    ColumnDef("allow_sizes", ["ALLOWSIZES", "兼容箱型"], "str"),
    ColumnDef("BOOLATTR", ["BOOLATTR", "布尔属性"], "int"),
    ColumnDef("size_type", ["SIZETYPE", "尺寸类型"], "str"),
    ColumnDef("LASTUPDATEMAN", ["LASTUPDATEMAN", "更新人"], "str"),
    ColumnDef("LASTUPDATETIME", ["LASTUPDATETIME", "更新时间"], "datetime"),
])

# ══════════════════════════════════════════════════════════════
# 表三：靠离泊计划 (03_berth_plan.parquet)
# ══════════════════════════════════════════════════════════════
BERTH_PLAN = TableDef("berth_plan", [
    ColumnDef("berth_plan_no", ["BERTHPLANNO", "BERTH_PLAN_NO", "计划编号"], "int"),
    ColumnDef("inbound_voy", ["INBOUNDVOY", "进港航次"], "str"),
    ColumnDef("outbound_voy", ["OUTBOUNDVOY", "出港航次"], "str"),
    ColumnDef("eta", ["ETA_TIME", "ETA", "预计到港"], "datetime"),
    ColumnDef("etd", ["ETD_TIME", "ETD", "预计离港"], "datetime"),
    ColumnDef("actual_berth", ["实际靠泊时间", "ACTUAL_BERTH"], "datetime"),
    ColumnDef("actual_depart", ["实际离泊时间", "ACTUAL_DEPART"], "datetime"),
    ColumnDef("EVESSELNAME", ["EVESSELNAME", "英文船名"], "str"),
    ColumnDef("AVESSELNAME", ["AVESSELNAME", "船名缩写"], "str"),
    ColumnDef("CVESSELNAME", ["CVESSELNAME", "中文船名"], "str"),
    ColumnDef("IMO", ["IMO"], "str"),
    ColumnDef("TERMINALCODES", ["TERMINALCODES", "码头代码"], "str"),
])

# ══════════════════════════════════════════════════════════════
# 表六：堆场箱位定义 (06_yard_definition.parquet)
# ══════════════════════════════════════════════════════════════
YARD_DEFINITION = TableDef("yard_definition", [
    ColumnDef("block_no", ["BLOCK_NO", "场区号"], "str"),
    ColumnDef("bay_no", ["BAY_NO", "贝位号"], "int"),
    ColumnDef("stack_no", ["STACK_NO", "列号"], "int"),
    ColumnDef("tier_no", ["TIER_NO", "层号"], "int"),
    ColumnDef("is_useful", ["IS_USEFUL", "是否可用"], "bool"),
    ColumnDef("allow_sizes", ["ALLOW_SIZES", "兼容尺寸"], "str"),
    ColumnDef("allow_types", ["ALLOW_TYPES", "兼容箱型"], "str"),
    ColumnDef("yard_area", ["YARD_AREA", "堆场区域"], "str"),
])

# ══════════════════════════════════════════════════════════════
# 表四+五：出口集装箱清单+装船位置 (05_export_manifest.parquet)
# ══════════════════════════════════════════════════════════════
EXPORT_MANIFEST = TableDef("export_manifest", [
    ColumnDef("container_id", ["CONTAINER_ID", "箱ID"], "str"),
    ColumnDef("container_no", ["CONTAINER_NO", "箱号"], "str"),
    ColumnDef("berth_plan_no", ["BERTHPLANNO", "靠泊计划号"], "str"),
    ColumnDef("container_size", ["CONTAINER_SIZE", "箱尺寸"], "int"),
    ColumnDef("container_type", ["CONTAINER_TYPE", "箱类型"], "str"),
    ColumnDef("gross_weight", ["GROSS_WEIGHT", "毛重"], "float"),
    ColumnDef("pod", ["POD", "卸货港"], "str"),
    ColumnDef("pol", ["POL", "装货港"], "str"),
    ColumnDef("stow_position", ["STOW_POSITION", "装船位置"], "str"),
    ColumnDef("stow_bay", ["STOW_BAY", "装船贝位"], "int"),
    ColumnDef("stow_row", ["STOW_ROW", "装船列"], "int"),
    ColumnDef("stow_tier", ["STOW_TIER", "装船层"], "int"),
    ColumnDef("dangerous_class", ["DANGEROUS_CLASS", "危品等级"], "str"),
    ColumnDef("reefer", ["REEFER", "冷藏"], "bool"),
    ColumnDef("overseas", ["OVERSEAS", "外贸"], "bool"),
    ColumnDef("data_half", ["DATA_HALF", "数据半年"], "str"),
])

# ══════════════════════════════════════════════════════════════
# 表八：集装箱移动事件流 (06_movement_events.parquet)
# ══════════════════════════════════════════════════════════════
MOVEMENT_EVENTS = TableDef("movement_events", [
    ColumnDef("event_id", ["EVENT_ID", "事件ID"], "str"),
    ColumnDef("container_id", ["CONTAINER_ID", "箱ID"], "str"),
    ColumnDef("container_no", ["CONTAINER_NO", "箱号"], "str"),
    ColumnDef("op_type", ["OP_TYPE", "操作类型"], "str"),
    ColumnDef("op_time", ["OP_TIME", "操作时间"], "datetime"),
    ColumnDef("berth_plan_no", ["BERTHPLANNO", "靠泊计划号"], "str"),
    ColumnDef("container_size", ["CONTAINER_SIZE", "箱尺寸"], "int"),
    ColumnDef("from_pos", ["FROM_POS", "来源位置"], "str"),
    ColumnDef("to_pos", ["TO_POS", "目标位置"], "str"),
    ColumnDef("equipment", ["EQUIPMENT", "设备"], "str"),
    ColumnDef("in_time", ["IN_TIME", "进场时间"], "datetime"),
    ColumnDef("out_time", ["OUT_TIME", "出场时间"], "datetime"),
    ColumnDef("data_half", ["DATA_HALF", "数据半年"], "str"),
])

# ══════════════════════════════════════════════════════════════
# xce.dump 解析用：XCE3表 + 集装箱表
# ══════════════════════════════════════════════════════════════
XCE3 = TableDef("xce3", [
    ColumnDef("container_id", ["CONTAINER_ID", "箱ID"], "str"),
    ColumnDef("xce_type", ["XCE_TYPE", "XCE类型"], "str"),
    ColumnDef("xce_date", ["XCE_DATE", "XCE日期"], "datetime"),
    ColumnDef("block_no", ["BLOCK_NO", "场区号"], "str"),
    ColumnDef("bay_no", ["BAY_NO", "贝位号"], "int"),
    ColumnDef("stack_no", ["STACK_NO", "列号"], "int"),
    ColumnDef("tier_no", ["TIER_NO", "层号"], "int"),
])

COMPTAINERS = TableDef("comtainers", [
    ColumnDef("container_id", ["CONTAINER_ID", "箱ID"], "str"),
    ColumnDef("container_no", ["CONTAINER_NO", "箱号"], "str"),
    ColumnDef("container_size", ["CONTAINER_SIZE", "箱尺寸"], "int"),
    ColumnDef("container_type", ["CONTAINER_TYPE", "箱类型"], "str"),
    ColumnDef("gross_weight", ["GROSS_WEIGHT", "毛重"], "float"),
    ColumnDef("pod", ["POD", "卸货港"], "str"),
    ColumnDef("pol", ["POL", "装货港"], "str"),
])
