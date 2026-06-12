import argparse
import sys
from pathlib import Path
from src.services.reporting.json_publisher import JSONPublisher
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("publisher-main")

def main():
    parser = argparse.ArgumentParser(description="JSON Data Publisher")
    parser.add_argument("--base-dir", "--base_dir", required=True, help="Base directory for equity data")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        logger.error("BASE_DIR_NOT_FOUND", extra={"path": str(base_dir)})
        sys.exit(1)

    output_file = Path(args.output).resolve()

    try:
        publisher = JSONPublisher(base_dir, output_file)
        publisher.publish()
    except Exception as e:
        logger.error("JSON_PUBLISH_FAILED", extra={"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
