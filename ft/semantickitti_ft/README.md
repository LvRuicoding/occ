# SemanticKITTI Fine-tuning Structure

This experiment is split so the lifting method and occupancy head can be
changed independently.

- `train.py`: training/evaluation entry point. Select modules with
  `--lift` and `--head`.
- `dataset.py`: SemanticKITTI SSC dataset and collate function.
- `losses.py`: SSC losses.
- `interfaces.py`: shared dataclasses passed between lift modules and heads.
- `lifting/`: feature lifting modules. The current default is
  `occany_render_tokens`, which reproduces the previous OccAny recon +
  novel-render token pipeline.
- `heads/`: SSC head modules. The current default is `monoscene`.
- `models/`: model assembly that wires a lifter to a head.
- `occ_wrapper.py`: compatibility shim for older imports.

To add a new lifting method, create a module under `lifting/`, decorate its
class with `@register_lifter("your_name")`, return `LiftedFeatures`, and import
the module in `lifting/__init__.py` so it is registered.

To add a new head, create a module under `heads/`, decorate its class with
`@register_head("your_name")`, implement `forward(features) -> logits`, and
import the module in `heads/__init__.py` so it is registered.
