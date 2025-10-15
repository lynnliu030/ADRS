#!/usr/bin/env python3
"""Launch the ADRS full evaluator on a remote SkyPilot cluster.

The script uses the SkyPilot Python SDK to:
  1. Load the SkyPilot task definition from ``skypilot/config.yaml``.
  2. Launch or reuse a cluster with high-end resources.
  3. Stream logs while ``full_eval.py`` runs on the remote machine.
  4. Rsync the generated JSON summary back to the local machine.

Example
-------
```
python scripts/run_skypilot_full_eval.py \
  --cluster adrs-eval \
  --config skypilot/config.yaml \
  --local-output openevolve/examples/ADRS/cant-be-late/full_eval_remote.json
```
"""

import argparse
import subprocess
import json
from pathlib import Path

import sky

SCRIPT_PATH = Path(__file__).resolve()


def _find_default_config() -> Path:
    """Search up from both CWD and script for skypilot/config.yaml."""
    search_roots = [Path.cwd(), SCRIPT_PATH.parent]
    search_roots.extend(SCRIPT_PATH.parents)
    seen = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / 'skypilot/config.yaml'
        if candidate.exists():
            return candidate
    return Path('skypilot/config.yaml')

DEFAULT_CONFIG = _find_default_config()
DEFAULT_REMOTE_OUTPUT = '~/sky_workdir/skypilot_artifacts/full_eval_remote.json'
DEFAULT_LOCAL_OUTPUT = Path('openevolve/examples/ADRS/cant-be-late/full_eval_remote.json')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cluster',
        default='adrs-eval',
        help='Name of the SkyPilot cluster to launch/reuse (default: adrs-eval).',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to the SkyPilot task YAML (default: skypilot/config.yaml).',
    )
    parser.add_argument(
        '--remote-output',
        default=DEFAULT_REMOTE_OUTPUT,
        help='Remote path of the JSON summary (default matches the YAML).',
    )
    parser.add_argument(
        '--local-output',
        type=Path,
        default=DEFAULT_LOCAL_OUTPUT,
        help='Local path to save the JSON summary (default: openevolve/examples/ADRS/cant-be-late/full_eval_remote.json).',
    )
    parser.add_argument(
        '--teardown',
        action='store_true',
        help='Tear down the cluster after the run completes.',
    )
    return parser.parse_args()


def ensure_local_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    config_path = (args.config or DEFAULT_CONFIG).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f'SkyPilot config not found: {config_path}')

    ensure_local_directory(args.local_output.resolve())

    task = sky.Task.from_yaml(str(config_path))

    request_id = sky.launch(task, cluster_name=args.cluster)
    job_id, _ = sky.stream_and_get(request_id)
    exit_code = sky.tail_logs(args.cluster, job_id=job_id, follow=True)
    if exit_code != 0:
        raise RuntimeError(f"Remote SkyPilot job failed with exit code {exit_code}")

    # Ensure ssh config/host alias is up-to-date before using plain rsync.
    subprocess.run(['sky', 'status', args.cluster], check=True)

    rsync_target = f'{args.cluster}:{args.remote_output}'
    subprocess.run(
        [
            'rsync',
            '-Pavz',
            rsync_target,
            str(args.local_output.resolve()),
        ],
        check=True,
    )

    if args.teardown:
        sky.down(cluster_name=args.cluster)

    local_path = args.local_output.resolve()
    print(f'📄 JSON report saved to {local_path}')

    try:
        with local_path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        summary = data.get('summary', {})
        baselines = data.get('baselines', [])
        if summary:
            print(
                f"🎯 Target avg cost ${summary.get('average_cost'):.2f}"
                f" (std ${summary.get('cost_std'):.2f})"
            )
        for record in baselines:
            path = record.get('path', '')
            label = Path(path).name if path else 'baseline'
            if label == 'initial_greedy.py':
                continue
            comp = record.get('comparison', {})
            avg_ratio = comp.get('avg_improvement_ratio')
            if avg_ratio is not None:
                print(
                    f"💡 Improvement vs {label}: avg {avg_ratio*100:.2f}%"
                )
    except Exception as exc:  # noqa: BLE001
        print(f'⚠️  Warning: failed to summarize JSON ({exc})')


if __name__ == '__main__':
    main()
