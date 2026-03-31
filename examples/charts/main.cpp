#include <iostream>
#include <vector>

#include "charts_api.h"

int main() {
    using namespace bare_nn::charts;

    ChartConfig cfg;
    cfg.width = 980;
    cfg.height = 620;
    cfg.title = "Operations Dashboard";

    const std::vector<std::string> months = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN"};
    const std::vector<double> sales = {62, 88, 54, 79, 68, 92};

    if (!create_bar_chart("examples/charts/bar_chart.svg", months, sales, cfg)) {
        std::cerr << "bar chart failed\n";
        return 1;
    }

    const std::vector<std::string> pie_labels = {"Group 1", "Group 2", "Group 3", "Group 4"};
    const std::vector<double> pie_values = {25, 42, 18, 50};
    if (!create_pie_chart("examples/charts/pie_chart.svg", pie_labels, pie_values, cfg)) {
        std::cerr << "pie chart failed\n";
        return 1;
    }

    const std::vector<std::string> x_labels = {"10", "20", "30", "40", "50", "60"};
    const std::vector<double> trend = {38, 42, 31, 50, 47, 58};
    if (!create_line_chart("examples/charts/line_chart.svg", x_labels, trend, cfg)) {
        std::cerr << "line chart failed\n";
        return 1;
    }

    const std::vector<std::string> categories = {"Q1", "Q2", "Q3", "Q4"};
    const std::vector<std::string> stack_labels = {"Ops", "Sales", "R&D"};
    const std::vector<std::vector<double>> stacks = {
        {20, 28, 24, 30},
        {18, 16, 22, 19},
        {14, 19, 17, 20}
    };
    if (!create_stacked_bar_chart(
            "examples/charts/stacked_bar_chart.svg", categories, stack_labels, stacks, cfg)) {
        std::cerr << "stacked bar chart failed\n";
        return 1;
    }

    ChartConfig showcase_cfg = cfg;
    showcase_cfg.title = "Theme Reference Panel";
    if (!create_theme_showcase("examples/charts/theme_showcase.svg", showcase_cfg)) {
        std::cerr << "theme showcase failed\n";
        return 1;
    }

    std::cout << "Generated charts in examples/charts/" << std::endl;
    return 0;
}
