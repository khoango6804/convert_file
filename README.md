# File Converter Tool

A simple utility to convert files between different formats.

## Features

- Convert between various file formats
- Easy-to-use command line interface
- Batch processing capabilities
- Preserves metadata during conversion

## Installation

```bash
# Clone the repository
git clone https://github.com/username/convert_file.git

# Navigate to the project directory
cd convert_file

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic conversion
python main.py --input file.pdf --output file.md

# Batch conversion (recommend)
python main.py --input folder/ --output output/ --format md
```

## Supported Formats

- Text to PDF
- PDF to Text
- Image to Text (OCR)
- And more...

## Requirements

- Python 3.7 or higher
- Required dependencies listed in requirements.txt

## Configuration

Before running the tool, you need to set up your Google API key.

1. Visit [Google API Console](https://makersuite.google.com/app/apikey) to create or obtain your API key.
2. Create a file named `api.json` in the project root directory with the following content:

```json
{
    "api_key": "YOUR_GOOGLE_API_KEY"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
