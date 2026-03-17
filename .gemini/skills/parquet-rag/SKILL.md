# Parquet-RAG Skill

A skill for multimodal RAG, docling-based chunking, and DMN-driven reasoning for the arbitration layer.

## Description

The `parquet-rag` skill provides the intelligence core for the Managing Agent. It handles the conversion of documents into Parquet-based RAG knowledge bases, executes DMN logic for decision-making, and lookups language translation mappings in XML.

## Tools

### docling_to_parquet
Converts documents to Parquet chunks.
- `input_path`: Path to input document.
- `output_parquet`: Target parquet file path.

### query_multimodal_rag
Retrieves reasoning chunks from Parquet.
- `query`: Natural language query.
- `parquet_path`: Path to knowledge base.

### reason_with_dmn
Executes the DMN reasoning matrix for arbitration.
- `context`: State object containing confidence and task info.

## Resources

- [translation_mapping.xml](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/parquet-rag/resources/translation_mapping.xml): XML-based mapping of compiler functions.
- [arbitration_dmn.json](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/parquet-rag/resources/arbitration_dmn.json): DMN decision table for the arbitration layer.
