from pathlib import Path
import os

artifact_dir = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\589bf15f-b64a-41ee-a26b-d091cb2629e5").resolve()
print(f"Artifact Dir: {artifact_dir}")

base_path = r"Users\HP\.gemini\antigravity-ide\brain\589bf15f-b64a-41ee-a26b-d091cb2629e5\exp1_multimodal_roc.png"

test_formats = [
    f"/C:/{base_path}",
    f"/C:\\{base_path}",
    f"/C:/{base_path.replace('\\', '/')}",
    f"/C://{base_path.replace('\\', '/')}",
    f"/{base_path}",
    f"///C:/{base_path.replace('\\', '/')}",
    f"//./C:/{base_path.replace('\\', '/')}",
    f"//?/C:/{base_path.replace('\\', '/')}",
    f"/C|/{base_path.replace('\\', '/')}",
    f"/c:/{base_path.replace('\\', '/')}",
]

for p in test_formats:
    p_normalized = p.replace('\\', '/')
    print(f"\nPath: {p_normalized}")
    print(f"  Starts with '/': {p_normalized.startswith('/')}")
    try:
        resolved = Path(p_normalized).resolve()
        print(f"  Resolved: {resolved}")
        is_rel = resolved.is_relative_to(artifact_dir)
        print(f"  Is relative to artifact dir: {is_rel}")
        if is_rel:
            print("  >>> SUCCESS! <<<")
    except Exception as e:
        print(f"  Error: {e}")
