import os
import shutil
from pathlib import Path

def create_gallery(base_dirs, output_folder="data_review_gallery"):
    """
    Scans base_dirs for 'DIRTY' folders and creates a visual HTML gallery.
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    img_dir = output_path / "images"
    img_dir.mkdir(exist_ok=True)

    html_content = [
        "<!DOCTYPE html>",
        "<html><head><title>Weiqi Data Cleaning Gallery</title>",
        "<style>",
        "  body { font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 15px; }",
        "  .item { background: #333; padding: 10px; border-radius: 8px; text-align: center; font-size: 11px; }",
        "  img { width: 100%; height: auto; border: 1px solid #555; margin-bottom: 5px; }",
        "  .label { font-weight: bold; color: #4dabf7; }",
        "  .source { color: #888; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        "</style></head><body>",
        f"<h1>Data Cleaning Review Gallery</h1>",
        f"<p>Found in DIRTY folders of: {base_dirs}</p>",
        "<div class='grid'>"
    ]

    count = 0
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    for b_dir in base_dirs:
        p_path = Path(b_dir)
        # Look for DIRTY subfolders in B, W, E
        for cls in ["B", "W", "E"]:
            dirty_dir = p_path / cls / "DIRTY"
            if not dirty_dir.exists():
                continue
            
            print(f"Scanning {dirty_dir}...")
            for img_file in list(dirty_dir.glob("*.jpg")) + list(dirty_dir.glob("*.png")):
                # Copy image to gallery folder for easy relative pathing
                new_name = f"{cls}_{img_file.name}"
                shutil.copy(img_file, img_dir / new_name)
                
                html_content.append(
                    f"<div class='item'>"
                    f"<img src='images/{new_name}' alt='{new_name}'>"
                    f"<div class='label'>Class: {cls}</div>"
                    f"<div class='source'>{img_file.name}</div>"
                    f"</div>"
                )
                count += 1

    html_content.append("</div></body></html>")

    with open(output_path / "index.html", "w", encoding="utf-8") as f:
        f.writelines(html_content)

    print(f"\n[DONE] Gallery created with {count} images.")
    print(f"Open this file to review: {output_path.absolute()}\\index.html")

if __name__ == "__main__":
    # Specify your patch directories here
    paths = [
        r"D:\Codes\Data\Gomrade\dataset1\patches",
        r"D:\Codes\Data\Gomrade\dataset2\patches"
    ]
    create_gallery(paths)
