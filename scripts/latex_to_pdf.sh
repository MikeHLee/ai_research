#!/bin/bash

# Check if a file path is provided
if [ $# -eq 0 ]; then
    echo "Error: No LaTeX file path provided"
    echo "Usage: $0 path/to/latex/file.tex"
    exit 1
fi

# Get the input file path
input_file="$1"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist"
    exit 1
fi

# Check if the file is a .tex file
if [[ ! "$input_file" =~ \.tex$ ]]; then
    echo "Error: File '$input_file' is not a .tex file"
    exit 1
fi

# Get the directory of the input file and the parent directory
dir=$(dirname "$input_file")
parent_dir=$(dirname "$dir")

# Get the base filename without path and extension
base_filename=$(basename "$input_file" .tex)

# Run pdflatex twice to resolve references
echo "Converting $input_file to PDF..."
pdflatex -interaction=nonstopmode -output-directory="$dir" "$input_file"
pdflatex -interaction=nonstopmode -output-directory="$dir" "$input_file"

# Check if PDF was created in the latex directory
temp_pdf_file="$dir/$base_filename.pdf"
if [ -f "$temp_pdf_file" ]; then
    # Move the PDF to the parent directory
    parent_pdf_file="$parent_dir/$base_filename.pdf"
    mv "$temp_pdf_file" "$parent_pdf_file"
    echo "Successfully created: $parent_pdf_file"
    
    # Clean up auxiliary files
    rm -f "$dir/$base_filename.aux" "$dir/$base_filename.log" "$dir/$base_filename.out"
    echo "Cleaned up auxiliary files"
else
    echo "Error: PDF generation failed"
    exit 1
fi
