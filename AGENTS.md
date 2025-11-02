# AGENTS.md - Development Guidelines for RNK Project

## Build/Run Commands
```bash
# Training
python -m integrated_pipeline.main --mode train --config config/cfg_pretrain.yaml

# Evaluation  
python -m integrated_pipeline.main --mode eval --checkpoint <path> --config config/cfg_pretrain.yaml

# Inference
python -m integrated_pipeline.main --mode infer --checkpoint <path> --input <text>

# Run example usage
python integrated_pipeline/example_usage.py
```

## Code Style Guidelines
- **Imports**: Standard lib → Third-party → Local (`.config`, `.components`)
- **Naming**: PascalCase classes, snake_case functions/variables, UPPER_SNAKE_CASE constants
- **Types**: Use comprehensive type hints, `Optional[type]` for nullable, forward refs for class methods
- **Formatting**: 4 spaces, ~100 char lines, 2 blank lines between top-level functions/classes
- **Error Handling**: Try-catch with fallbacks, `raise ... from e` for context, clear validation messages
- **Docstrings**: Google-style (Args/Returns), brief module/class descriptions, strategic inline comments
- **PyTorch**: Inherit from `nn.Module`, explicit `.to(device)`, gradient control with `torch.set_grad_enabled()`

## Testing
No formal test framework exists. Use assertion-based validation and manual evaluation scripts. Run evaluation through main.py with eval mode.