# Landmark Detection

A Python project for detecting face, hand, and pose landmarks from images or webcam video using OpenCV and MediaPipe.

## Features

- Detects face mesh, hand landmarks, and pose landmarks
- Works with webcam video or image files
- Saves annotated output images
- Clean CLI entry point for quick testing
- Small, beginner-friendly codebase

## Project Structure

```text
landmark-detection/
  src/landmark_detection/
    __init__.py
    cli.py
    detector.py
  .gitignore
  LICENSE
  README.md
  pyproject.toml
  requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run on webcam:

```bash
python -m landmark_detection.cli --webcam
```

Run on an image:

```bash
python -m landmark_detection.cli --image path/to/image.jpg --output output.jpg
```

Choose which landmark systems to run:

```bash
python -m landmark_detection.cli --image path/to/image.jpg --faces --hands --pose
```

## Notes

- Press `q` to quit webcam mode.
- Good lighting improves landmark quality.
- For best webcam performance, close other camera applications first.

## License

MIT License
