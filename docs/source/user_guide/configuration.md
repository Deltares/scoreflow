# Configuration

The pipeline configuration can be provided in either in a Python object, or via a YAML configuration file. For new users, we recommend starting with a YAML configuration file.

`````{tab-set}
````{tab-item} YAML
```yaml
dummy_pipeline:
  dummy_a: true
  dummy_b: strict
```
````

````{tab-item} Python
```python
from veriflow import configure

configure(
    dummy_pipeline={
        "dummy_a": True,
        "dummy_b": "strict",
    }
)
```
````
`````

## JSON schema for editor validation

A JSON Schema describing the YAML configuration is published alongside the
documentation on GitHub Pages. Editors that support the
[yaml-language-server](https://github.com/redhat-developer/yaml-language-server)
modeline (e.g. VS Code with the Red Hat YAML extension, JetBrains IDEs) will
then provide auto-completion and real-time validation while you edit.

Add the following modeline at the top of your YAML config:

```yaml
# yaml-language-server: $schema=https://deltares.github.io/veriflow/v0/config.schema.json
```

The schema is versioned by major version (`v0`, `v1`, ...). Reference the
version that matches the `veriflow` release you target. The same file is
also committed to the repository under `schemas/<version>/config.schema.json`
for offline use.
