统一类别编号 GT：class_mapping.json 为编号依据。
原始Labelme标签保留，objects[].class_id 为明确类别编号。
此GT结构为成员3导出格式；与评价器对接时确认字段。
actual_counts 为真实标注计数，不是推理输入 expected_counts；不得将GT提供给推理模型。
划分保持现状，场景独立性尚待核查；包含测试真值，仅用于本地受控管理，不上传公开Git。
平垫片实物规格尚待确认；没有添加干扰件示例。
