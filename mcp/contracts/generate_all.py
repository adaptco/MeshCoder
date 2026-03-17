import yaml
import json
import os
from pathlib import Path
import subprocess

def generate_ts_types(tools):
    output = "/* Generated from tools.yaml */\n\n"
    for tool in tools:
        name = tool['name']
        schema = tool['input_schema']
        type_name = "".join(s.capitalize() for s in name.split("_")) + "Input"
        
        output += f"export interface {type_name} {{\n"
        props = schema.get('properties', {})
        required = schema.get('required', [])
        
        for prop_name, prop_details in props.items():
            prop_type = prop_details.get('type', 'any')
            if prop_type == 'integer': prop_type = 'number'
            elif prop_type == 'object': prop_type = 'any'
            
            is_optional = "?" if prop_name not in required else ""
            output += f"  {prop_name}{is_optional}: {prop_type};\n"
        
        output += "}\n\n"
    return output

def main():
    contracts_dir = Path(__file__).parent
    tools_path = contracts_dir / "tools.yaml"
    generated_dir = contracts_dir / "generated"
    generated_dir.mkdir(exist_ok=True)

    with open(tools_path, 'r') as f:
        data = yaml.safe_load(f)

    tools = data.get('tools', [])

    # Generate TS types
    ts_types = generate_ts_types(tools)
    with open(generated_dir / "types.ts", 'w') as f:
        f.write(ts_types)
    print("Generated types.ts")

    # Generate Python models using datamodel-code-generator
    # First, create a combined JSON schema
    combined_schema = {
        "title": "MCPTools",
        "type": "object",
        "definitions": {
            "".join(s.capitalize() for s in tool['name'].split("_")) + "Input": tool['input_schema']
            for tool in tools
        }
    }
    schema_path = generated_dir / "schema.json"
    with open(schema_path, 'w') as f:
        json.dump(combined_schema, f, indent=2)

    try:
        subprocess.run([
            "python3", "-m", "datamodel_code_generator",
            "--input", str(schema_path),
            "--input-file-type", "jsonschema",
            "--output", str(generated_dir / "models.py")
        ], check=True)
        print("Generated models.py")
    except Exception as e:
        print(f"Failed to run datamodel-code-generator: {e}")

if __name__ == "__main__":
    main()
