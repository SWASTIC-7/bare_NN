# Charts API Example

This example uses the charts API in `include/charts_api.h` and `src/charts/charts_api.cpp`.

Generated chart files:

- `examples/charts/bar_chart.svg`
- `examples/charts/pie_chart.svg`
- `examples/charts/line_chart.svg`
- `examples/charts/stacked_bar_chart.svg`
- `examples/charts/area_line_chart.svg`
- `examples/charts/multi_line_chart.svg`
- `examples/charts/grouped_bar_chart.svg`
- `examples/charts/horizontal_progress_chart.svg`
- `examples/charts/horizontal_ranked_bar_chart.svg`
- `examples/charts/theme_showcase.svg`

## Build

From repository root:

```bash
g++ -std=c++17 -Iinclude examples/charts/main.cpp src/charts/charts_api.cpp -o build/charts_example
```

## Run

```bash
./build/charts_example
```

On PowerShell:

```powershell
.\build\charts_example.exe
```

The output SVGs are styled with a cool gray analytics dashboard theme inspired by your reference image and include synthetic datasets for each chart type.
