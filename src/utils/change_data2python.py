


import json

entity_type_all = "Disease"
new_instruction_template = "def Bio_Named_Entity_Recognition(InputText: str):\n    \'\'\'\n    Task Definition: Please extract all relevant entities from the sentence and indicate their type. The types include: **{entity_type_all}**.\n    \'\'\'\n    return EntityList\n"

new_output_template = "EntityList.append({entity_type}(\"{entity_name}\"))\n"
no_entity_template = "EntityList.append(\"there is no {entity_type_all} entity\")"

##拿出原始数据
with open("bio_data/ncbi/ncbi_test.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

##修改新的数据内容
for data in all_data:
    past_instruction = data["instruction"]
    past_input = data["input"]
    past_output = data["output"]

    #新的指令
    data["instruction"] = new_instruction_template.format(entity_type_all=entity_type_all)
    #新的输入
    data["input"] = "InputText =" + f'"{past_input}"' + "\nEntityList = Named_Entity_Recognition(InputText)\n"

    #新的输出
    entities = past_output.split("##")
    new_output = ""
    for entity in entities:
        if entity.strip() == f"there is no {entity_type_all} entity":
            new_output = no_entity_template.format(entity_type_all=entity_type_all)
            break  # 负例直接跳过
        name, type_ = entity.rsplit("-", 1)  # 从右边分割，防止name中有-存在
        new_output += new_output_template.format(entity_type=type_, entity_name=name)
    data["output"] = new_output

# 保存到新的文件
with open("bio_data/ncbi_python/ncbi_python_test.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("处理完成，已保存到 bc2gm_python_test.json")
