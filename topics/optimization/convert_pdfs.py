#!/usr/bin/env python3
"""
Convert PDF files to Markdown format for analysis
"""
import pymupdf4llm
import os
from pathlib import Path

def convert_pdf_to_markdown(pdf_path, output_path):
    """Convert a single PDF to markdown"""
    print(f"Converting {pdf_path.name}...")
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        print(f"  ✓ Saved to {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    # Set up paths
    sources_dir = Path("sources")
    output_dir = Path("sources_markdown")
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(sources_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in sources directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Convert each PDF
    success_count = 0
    for pdf_path in sorted(pdf_files):
        output_path = output_dir / f"{pdf_path.stem}.md"
        if convert_pdf_to_markdown(pdf_path, output_path):
            success_count += 1
        print()
    
    print(f"Conversion complete: {success_count}/{len(pdf_files)} successful")

if __name__ == "__main__":
    main()
