# Execution

You can execute a verification pipeline via a Python interface, or via the Command-Line. When running from Python, an OutputDataset object is returned.

`````{tab-set}
````{tab-item} Python
```python
from veriflow import run_pipeline
from pathlib import Path

path_to_config = Path("./config.yaml")
output_dataset = run_pipeline((path_to_config, "yaml"))
```
````

````{tab-item} Command Line
```bash
veriflow run --config ./config.yaml
```
````
`````
